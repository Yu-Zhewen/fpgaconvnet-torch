import datetime
import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description='Low rank approximation experiment')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
args = parser.parse_args()

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


sweep_range = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
for relu_threshold in sweep_range:
    test_name = "sparsity_run_relu_threshold" + str(relu_threshold)

    start_time = datetime.datetime.now()
    log_dir= test_name + "_" + str(start_time).replace(" ","_").replace(".","_").replace(":","_").replace("-", "_")

    os.makedirs("runlog/" + log_dir)
    log_file="runlog/" + log_dir + "/log.txt"

    regsys_cmd="python3 -u imagenet_main.py --output_path " + "runlog/" + log_dir + " --relu_threshold " + str(relu_threshold) + " --gpu " + str(args.gpu)

    with open(log_file, "w") as log_fp:
        log_fp.write(regsys_cmd + '\n')

    os.system(regsys_cmd + " 2>&1 | tee -a " + log_file)

