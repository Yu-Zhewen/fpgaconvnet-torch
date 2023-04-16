import os
import glob

if __name__ == "__main__":
    for data_dir in glob.glob("runlog/resnet18_sparsity_run_50K_relu*"):
        relu_threshold = data_dir.split("_")[5]
        os.system("python onnx_sparsity_attribute.py --arch resnet18 --dense_onnx_path onnx_models/resnet18.onnx --data " + data_dir + " --sparse_onnx_path onnx_models/resnet18_sparse_relu_" + relu_threshold + ".onnx ")