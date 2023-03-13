import os
import numpy as np
import torch
import torch.nn as nn
import onnx
import argparse
import csv
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

def layer_name_translation(model_name, onnx_name):
    if model_name == "resnet18":
        onnx_name = onnx_name.split("/")[1:]
        if len(onnx_name) == 2: # first conv
            torch_name = onnx_name[0]
        else:
            assert len(onnx_name) in [4,5] # block conv or downsample conv
            torch_name = onnx_name[1] + "." +onnx_name[-2]

    return torch_name

def annotate_sparsity(model_name, onnx_model, data_path):
    conv_layer_index = 0
    for node in onnx_model.graph.node:
        if node.op_type == 'Conv':
            layer_name = layer_name_translation(model_name, node.name)
            np_path = os.path.join(data_path, model_name + "_" + layer_name + "_mean.npy")
            num_of_zeros_mean = np.load(np_path)
            for attr in node.attribute:
                if attr.name == "kernel_shape":
                    kernel_shape = attr.ints
                    break
            sparsity_data = num_of_zeros_mean / np.prod(kernel_shape)
            set_nodeattr(node, "input sparsity", sparsity_data)

parser = argparse.ArgumentParser(description='Export ONNX model with sparsity attribute')
parser.add_argument('-a', '--arch', metavar='ARCH', default='repvgg-a0',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names))
parser.add_argument('--data', metavar='DIR', default="runlog/sparsity_run_ma_window_size1_2023_02_03_16_57_36_500547",
                    help='path to onnx model')  
parser.add_argument('--dense_onnx_path', metavar='DIR', default="models/repvgg-a0.onnx",
                    help='path to onnx model')              
parser.add_argument('--sparse_onnx_path', metavar='DIR', default="models/resnet18_sparse.onnx",
                    help='path to onnx model')      

args = parser.parse_args()

torch_model = load_model(args.arch)
torch_onnx_exporter(torch_model, args.arch, torch.randn(1, 3, 224, 224), args.dense_onnx_path)
onnx_model = onnx.load(args.dense_onnx_path)
annotate_quantisation(onnx_model, 16, 16, 32, False)
#annotate_sparsity(args.arch, onnx_model, args.data)
onnx.save(onnx_model, args.dense_onnx_path)