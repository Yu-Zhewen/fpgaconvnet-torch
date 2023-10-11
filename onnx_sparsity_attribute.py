import os
import numpy as np
import torch
import torch.nn as nn
import onnx
import argparse
import csv
import toml

from utils import load_model, model_names, replace_modules

def torch_onnx_exporter(model, model_name, random_input, output_path):
    if model_name == "mobilenet_v2":
        replace_dict = {}
        for name, module in model.named_modules():
            # todo: relu6 creates clip node
            if isinstance(module, nn.ReLU6):
                replace_dict[module] = nn.ReLU()
        replace_modules(model, replace_dict)
    torch.onnx.export(model, random_input, output_path, verbose=False, keep_initializers_as_inputs=True)

# https://github.com/Xilinx/finn-base/blob/7c2603a95e90e4de2575020e575c24eab6a15889/src/finn/custom_op/base.py
def set_nodeattr(node, attr_name, attr_value):
    print("annotate ", node.name, attr_name, attr_value)
    new_attr = onnx.helper.make_attribute(attr_name, attr_value)
    node.attribute.append(new_attr)

def annotate_quantisation(model, weight_width, data_width, acc_width, block_floating_point):
    for node in model.graph.node:
        if node.op_type in ["Conv", "Gemm"]:
            set_nodeattr(node, "weight_width", weight_width)
            set_nodeattr(node, "data_width", data_width)
            set_nodeattr(node, "acc_width", acc_width)
            set_nodeattr(node, "block_floating_point", block_floating_point)
        else:
            set_nodeattr(node, "data_width", data_width)

def annotate_sparsity_from_numpy(model_name, onnx_model, data_path):
    def _layer_name_translation(model_name, onnx_name):
        onnx_name = onnx_name.split("/")
        if model_name in ["resnet18", "resnet50"]:
            if len(onnx_name) == 3: # first conv
                torch_name = onnx_name[1]+ ".1"
            else:
                assert len(onnx_name) in [5,6] 
                torch_name = onnx_name[2] + "." +onnx_name[-2]+ ".1"
        elif model_name == "mobilenet_v2":
            if len(onnx_name) == 5: # first and last conv
                torch_name = onnx_name[-2] + ".1"
            else:
                assert len(onnx_name) in [6,7]
                torch_name = onnx_name[2] + "." + onnx_name[-2] + ".1"
        elif model_name in ["alexnet", "vgg11", "vgg16"]:
            torch_name = onnx_name[-2] + ".1"
        elif model_name == "repvgg-a0":
            torch_name = ".".join(onnx_name[1:-1]) + ".1"
        return torch_name

    for node in onnx_model.graph.node:
        if node.op_type == 'Conv':
            layer_name = _layer_name_translation(model_name, node.name)
            np_path = os.path.join(data_path, model_name + "_" + layer_name + "_mean.npy")
            num_of_zeros_mean = np.load(np_path)
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    kernel_shape = attr.ints
                    break
            sparsity_data = num_of_zeros_mean / np.prod(kernel_shape)
            set_nodeattr(node, "input sparsity", sparsity_data)

def annotate_sparsity_from_toml(model_name, onnx_model, data_path):
    def _layer_name_translation(model_name, onnx_name):
        onnx_name = onnx_name.split("/")
        if model_name in ["resnet18"]:
            if len(onnx_name) == 3: #first_conv
                torch_name = onnx_name[1]
            else:
                torch_name = onnx_name[2] + "." + onnx_name[-2] 
        return torch_name

    with open(data_path) as f:
        toml_data = toml.load(f)
    
    for node in onnx_model.graph.node:
        if node.op_type == 'Conv':
            layer_name = _layer_name_translation(model_name, node.name)
            sparsity_data = toml_data[layer_name]["avg"]
            set_nodeattr(node, "input sparsity", sparsity_data)

def annotate_histograms(model_name, onnx_model, data_path):
    for node in onnx_model.graph.node:
        if node.op_type == 'Conv':
            layer_name = layer_name_translation(model_name, node.name)
            np_path = os.path.join(data_path, model_name + "_" + layer_name + "_histograms.npy")
            channel_wise_sprasity = np.load(np_path)
            windows_data = channel_wise_sprasity[:, -1]/channel_wise_sprasity.sum(axis = 1)
            set_nodeattr(node, "window sparsity", windows_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export ONNX model with sparsity attribute')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',choices=model_names)
    parser.add_argument('--state_dict', metavar='DIR', default="/home/zy18/Downloads/Pruning Results-20230815T143526Z-001/Pruning Results/weight_sparse_50/resnet18_classification_imagenet_2023-08-12/software/transform/transformed_ckpt/state_dict.pt")
    parser.add_argument('--data_path', metavar='DIR', default="/home/zy18/Downloads/Pruning Results-20230815T143526Z-001/Pruning Results/weight_sparse_50/resnet18_classification_imagenet_2023-08-12/software/transform/prune/activation_report.toml")
    parser.add_argument('--export_path', metavar='DIR', default="models")
    args = parser.parse_args()

    torch_model = load_model(args.arch)
    if args.state_dict is not None:
        torch_model.load_state_dict(torch.load(args.state_dict, map_location="cpu"))
    dense_onnx_path = os.path.join(args.export_path, args.arch + ".onnx")
    sparse_onnx_path = os.path.join(args.export_path, args.arch + "_sparse.onnx")
    torch_onnx_exporter(torch_model, args.arch, torch.randn(1, 3, 224, 224), dense_onnx_path)
    onnx_model = onnx.load(dense_onnx_path)
    annotate_quantisation(onnx_model, 16, 16, 32, False)
    if args.data_path.endswith(".toml"):
        annotate_sparsity_from_toml(args.arch, onnx_model, args.data_path)
    else:
        annotate_sparsity_from_numpy(args.arch, onnx_model, args.data_path)
    # annotate_histograms(args.arch, onnx_model, args.data_path)
    onnx.save(onnx_model, sparse_onnx_path)
