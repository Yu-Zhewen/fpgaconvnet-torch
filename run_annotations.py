import os
import glob

if __name__ == "__main__":
    for model in ["resnet18", "resnet50", "vgg11", "vgg16", "alexnet", "mobilenet_v2"]:
        data = "runlog/" + model + "_sparsity_run_50K_relu_0_*"
        dir = glob.glob(data)[0]
        dense = "../fpgaconvnet-optimiser/fpgaconvnet/optimiser/onnx_models/" + model + ".onnx"
        sparse = "../fpgaconvnet-optimiser/fpgaconvnet/optimiser/onnx_models/" + model + "_full.onnx"
        os.system("python onnx_sparsity_attribute_full.py --arch " + model + " --data " + dir + " --dense_onnx_path " + dense + " --sparse_onnx_path " + sparse)