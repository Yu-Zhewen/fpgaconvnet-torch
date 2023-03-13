import os
import sys
import json
import re
from fpgaconvnet.optimiser.cli import main

def fpgaconvnet_runner():
    model_name = "alexnet"
    onnx_path = "/home/zy18/codeDev/sparseCNN/models/{}.onnx".format(model_name)

    output_path = os.path.join(os.environ['FPGACONVNET_OPTIMISER_SPARSE'], 'outputs/sparse/{}'.format(model_name))

    platform_path = os.path.join(os.environ['FPGACONVNET_OPTIMISER_SPARSE'], 'examples/platforms/u250.toml')

    if model_name in ["resnet18", "resnet50", "mobilenet_v2"]:
        optimiser_config_path = os.path.join(os.environ['FPGACONVNET_OPTIMISER_SPARSE'], 'examples/greedy_partition_throughput_residual.toml')
    elif model_name in ["repvgg-a0", "vgg11", "vgg16", "alexnet"]:
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
    fpgaconvnet_runner()
    #unit_test()