import os
import sys
import json
import re
from fpgaconvnet.optimiser.cli import main

def fpgaconvnet_runner(model_name):
    onnx_path = "/home/zy18/codeDev/sparseCNN/models/{}.onnx".format(model_name)

    output_path = os.path.join(os.environ['FPGACONVNET_OPTIMISER_SPARSE'], 'outputs/sparse/{}'.format(model_name))

    platform_path = os.path.join(os.environ['FPGACONVNET_OPTIMISER_SPARSE'], 'examples/platforms/u250.toml')

    if model_name in ["resnet18", "resnet50", "mobilenet_v2", "resnet18_sparse", "resnet50_sparse", "mobilenet_v2_sparse"]:
        optimiser_config_path = os.path.join(os.environ['FPGACONVNET_OPTIMISER_SPARSE'], 'examples/greedy_partition_throughput_residual.toml')
    elif model_name in ["repvgg-a0", "vgg11", "vgg16", "alexnet", "repvgg-a0_sparse", "vgg11_sparse", "vgg16_sparse", "alexnet_sparse"]:
        optimiser_config_path = os.path.join(os.environ['FPGACONVNET_OPTIMISER_SPARSE'], 'examples/greedy_partition_throughput.toml')

    saved_argv = sys.argv
    sys.argv  = ['cli.py']
    sys.argv += ['--name', model_name]
    sys.argv += ['--model_path', onnx_path]
    sys.argv += ['--platform_path', platform_path]
    sys.argv += ['--output_path', output_path]
    sys.argv += ['-b', '256']
    sys.argv += ['--objective', 'throughput']
    sys.argv += ['--optimiser', 'greedy_partition']
    sys.argv += ['--optimiser_config_path', optimiser_config_path]

    main()
    sys.argv = saved_argv

    with open(os.path.join(output_path, 'report.json'), 'r') as f:
        report = json.load(f)
        return report

def unit_test():
    from fpgaconvnet.models.layers.ConvolutionLayer import ConvolutionLayer
    import numpy as np
    b=np.load("/home/zy18/codeDev/sparseCNN/runlog/sparsity_run_ma_window_size1_2023_02_03_16_57_36_500547/resnet18_layer1.0.conv2_mean.npy")
    a=ConvolutionLayer(64,56,56,64,1,1,1,3,3,1,1,1,1,1,1,1,has_bias=True,sparsity=b/9)
    a.coarse_in = 4
    a.coarse_out = 64
    a.update()
    print(min(a.get_stream_sparsity()))
    print(min(a.get_stream_sparsity(interleave=False)))
if __name__ == "__main__":
    results = {}
    for model_name in ["resnet18", "resnet18_sparse", "resnet50", "resnet50_sparse", "mobilenet_v2", "mobilenet_v2_sparse", "repvgg-a0", "repvgg-a0_sparse", "vgg11", "vgg11_sparse", "vgg16", "vgg16_sparse", "alexnet", "alexnet_sparse"]:
        report = fpgaconvnet_runner(model_name)
        results[model_name] = report["network"]["performance"]["throughput"]
        print(results)
    #unit_test()