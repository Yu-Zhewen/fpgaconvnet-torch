import os
import numpy as np
import torch
import torch.nn as nn
import onnx
import argparse
import csv
import toml
import json

from utils import load_model, model_names, replace_modules

def onnx_layer_name_cast(model_name, onnx_name, sep="/", end=".1"):
    onnx_name = onnx_name.split(sep)
    if model_name in ["resnet18", "resnet50"]:
        if len(onnx_name) == 3: # first conv
            torch_name = onnx_name[1]+ end
        else:
            torch_name = onnx_name[2] + "." + onnx_name[-2] + end
    elif model_name == "mobilenet_v2":
        if len(onnx_name) == 5: # first and last conv
            torch_name = onnx_name[-2] + end
        else:
            torch_name = onnx_name[2] + "." + onnx_name[-2] + end
    elif model_name in ["alexnet", "vgg11", "vgg16"]:
        torch_name = onnx_name[-2] + end
    elif model_name == "repvgg-a0":
        torch_name = ".".join(onnx_name[1:-1]) + end
    return torch_name

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
    #print("annotate ", node.name, attr_name, attr_value)
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
    for node in onnx_model.graph.node:
        if node.op_type == 'Conv':
            layer_name = onnx_layer_name_cast(model_name, node.name)
            np_path = os.path.join(data_path, model_name + f"_{layer_name}_histograms.npy")
            histograms_data = np.load(np_path)
            histograms_data = histograms_data/histograms_data.sum(axis = 1, keepdims=True) # normalize
            set_nodeattr(node, "channel_sparsity_hist", histograms_data.flatten())

def annotate_sparsity_from_toml(model_name, onnx_model, data_path):
    assert False, "used in MASE, todo: update with histogram data"
    with open(data_path) as f:
        toml_data = toml.load(f)
    
    for node in onnx_model.graph.node:
        if node.op_type == 'Conv':
            layer_name = onnx_layer_name_cast(model_name, node.name, end="")
            sparsity_data = toml_data[layer_name]["hist"]
            set_nodeattr(node, "channel_sparsity_hist", sparsity_data)

def export_threshold_relu(model_name, onnx_path, relu_thresholds_path):
    with open(relu_thresholds_path) as f:
        relu_thresholds = json.load(f)
    onnx_model = onnx.load(onnx_path)

    for index, node in enumerate(onnx_model.graph.node):
        if node.op_type != "Relu":
            continue
        onnx_model.graph.node.remove(node)
        torch_name = onnx_layer_name_cast(model_name, node.name)
        new_node_name = "/".join(node.name.split("/")[:-1] + ["ThresholdedReLU"])
        new_node = onnx.helper.make_node(
            "ThresholdedRelu",
            name= new_node_name,
            inputs=[*node.input],
            outputs=node.output,
            alpha = relu_thresholds[torch_name]
        )
        onnx_model.graph.node.insert(index, new_node)
        next_node = next(filter(lambda x: node.output[0] in x.input, onnx_model.graph.node))
        next_node.input[0] = new_node.output[0]

    onnx.save(onnx_model, onnx_path)

def export_main():
    parser = argparse.ArgumentParser(description='Export ONNX model with sparsity attribute')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',choices=model_names)
    parser.add_argument('--state_dict', metavar='DIR', default=None)
    parser.add_argument('--data_path', metavar='DIR', default=None)
    parser.add_argument('--export_path', metavar='DIR', default="models")
    parser.add_argument("--relu_thresholds_path",  metavar='DIR', default=None,
                        help='path to relu thresholds json model')
    args = parser.parse_args()

    torch_model = load_model(args.arch)
    if args.state_dict is not None:
        torch_model.load_state_dict(torch.load(args.state_dict, map_location="cpu"))
    dense_onnx_path = os.path.join(args.export_path, args.arch + ".onnx")
    sparse_onnx_path = os.path.join(args.export_path, args.arch + "_sparse.onnx")
    torch_onnx_exporter(torch_model, args.arch, torch.randn(1, 3, 224, 224), dense_onnx_path)
    if args.relu_thresholds_path is not None:
        export_threshold_relu(args.arch, dense_onnx_path, args.relu_thresholds_path)
    onnx_model = onnx.load(dense_onnx_path)
    annotate_quantisation(onnx_model, 16, 16, 32, False)
    if args.data_path.endswith(".toml"):
        annotate_sparsity_from_toml(args.arch, onnx_model, args.data_path)
    else:
        annotate_sparsity_from_numpy(args.arch, onnx_model, args.data_path)
    onnx.save(onnx_model, sparse_onnx_path)

if __name__ == "__main__":
    export_main()