import datetime
import os
import subprocess
import argparse




parser = argparse.ArgumentParser(description='Low rank approximation experiment')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument("--model_path",  default=None, type=str,
                    help='Path to sparse .onnx model')

parser.add_argument("--platform_path", default=None, type=str,
                    help='Path to platform specs (.toml)')

"../../examples/platforms/zc706.toml"
parser.add_argument("--optimised_config_path",  default=None, type=str,
                    help='Path to optimised configuration (.json)')

parser.add_argument("--accuracy_output",  default=None, type=str,
                    help='Path to csv file to write accuracy to')


args = parser.parse_args()

#python relu_run.py --gpu 1 --model_path ../fpgaconvnet-optimiser/fpgaconvnet/optimiser/onnx_models/resnet18_sparse.onnx --platform_path ../fpgaconvnet-optimiser/examples/platforms/u250.toml --optimised_config_path ../fpgaconvnet-optimiser/fpgaconvnet/optimiser/outputs/sparse/resnet18_sparse_hetero/config.json

'''
sweep_range = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]#[131072, 262144, 524288]
for window_size in sweep_range:
    test_name = "sparsity_run_ma_window_size" + str(window_size)

    start_time = datetime.datetime.now()
    log_dir= test_name + "_" + str(start_time).replace(" ","_").replace(".","_").replace(":","_").replace("-", "_")

    os.makedirs("runlog/" + log_dir)
    log_file="runlog/" + log_dir + "/log.txt"

    regsys_cmd="python3 -u imagenet_main.py --output_path " + "runlog/" + log_dir + " --ma_window_size " + str(window_size) + " --gpu " + str(args.gpu)

    with open(log_file, "w") as log_fp:
        log_fp.write(regsys_cmd + '\n')

    os.system(regsys_cmd + " 2>&1 | tee -a " + log_file)
'''
'''
for model_name in ["resnet18"]:
    test_name = "{}_sparsity_run_50k".format(model_name)

    start_time = datetime.datetime.now()
    log_dir= test_name + "_" + str(start_time).replace(" ","_").replace(".","_").replace(":","_").replace("-", "_")

    os.makedirs("runlog/" + log_dir)
    log_file="runlog/" + log_dir + "/log.txt"

    regsys_cmd="python3 -u imagenet_main.py --output_path " + "runlog/" + log_dir + " --gpu " + str(args.gpu) + " -a " + model_name + f" --data /data/imagenet -b 4"

    with open(log_file, "w") as log_fp:
        log_fp.write(regsys_cmd + '\n')

    os.system(regsys_cmd + " 2>&1 | tee -a " + log_file)
'''

def relu_run(args):

    sweep_range = [0.1, 0.15, 0.2]
    for model_name in ["resnet18"]:
        for relu_threshold in sweep_range:
            test_name = model_name + "_sparsity_run_50K_relu_" + str(relu_threshold)

            start_time = datetime.datetime.now()
            log_dir= test_name + "_" + str(start_time).replace(" ","_").replace(".","_").replace(":","_").replace("-", "_")

            os.makedirs("runlog/" + log_dir)
            log_file="runlog/" + log_dir + "/log.txt"

            regsys_cmd="python imagenet_main.py --calibration-size 50000 --output_path " + "runlog/" + log_dir + " --relu_threshold " + str(relu_threshold) + " --gpu " + str(args.gpu) + \
                " --optimised_config_path " + args.optimised_config_path + " --platform_path " + args.platform_path + " --model_path " + args.model_path + " --accuracy_output " + args.accuracy_output

            with open(log_file, "w") as log_fp:
                log_fp.write(regsys_cmd + '\n')
            os.system(regsys_cmd + " 2>&1 | tee -a " + log_file)


if __name__ == "__main__":
    args = parser.parse_args()
    relu_run(args)
    # models_run(args)



