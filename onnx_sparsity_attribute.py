import torch
import onnx
import argparse
import csv
from utils import load_model, model_names

def torch_onnx_exporter(model, model_name, random_input, output_path):
    if model_name == "mobilenet_v2":
        replace_dict = {}
        for name, module in model.named_modules():
            # todo: relu6 creates clip node
            if isinstance(module, nn.ReLU6):
                replace_dict[module] = nn.ReLU()
        _replace_modules(model, replace_dict)
    torch.onnx.export(model, random_input, output_path, verbose=False, keep_initializers_as_inputs=True)

# https://github.com/Xilinx/finn-base/blob/7c2603a95e90e4de2575020e575c24eab6a15889/src/finn/custom_op/base.py
def set_nodeattr(node, attr_name, attr_value):
    print("annotate ", node.name, attr_name, attr_value)
    for attr in node.attribute:
        if attr.name == attr_name:
            if attr.type == 1:
                # float
                attr.f = attr_value
            else:
                assert False
            return
    new_attr = onnx.helper.make_attribute(attr_name, attr_value)
    node.attribute.append(new_attr)

def annotate_sparsity(model, data_path):
    with open(data_path, 'r') as f:
        csv_read = csv.reader(f)
        start = False
        sparsity_data = []
        for row in csv_read:
            if start:
                sparsity_data.append(float(row[2]))
            if len(row) >= 3 and row[2] == 'Accum Input Sparsity (Zeros / im2col(NCHW))':
                start = True

    conv_layer_index = 0
    for node in model.graph.node:
        if node.op_type == 'Conv':
            set_nodeattr(node, "input sparsity", sparsity_data[conv_layer_index])
            conv_layer_index += 1

    assert conv_layer_index == len(sparsity_data)

parser = argparse.ArgumentParser(description='Export ONNX model with sparsity attribute')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names))
parser.add_argument('--data', metavar='DIR', default="output/resnet18_sparsity_log.csv",
                    help='path to onnx model')  
parser.add_argument('--dense_onnx_path', metavar='DIR', default="models/resnet18.onnx",
                    help='path to onnx model')              
parser.add_argument('--sparse_onnx_path', metavar='DIR', default="models/resnet18_sparse.onnx",
                    help='path to onnx model')      

args = parser.parse_args()

torch_model = load_model(args.arch)
torch_onnx_exporter(torch_model, args.arch, torch.randn(1, 3, 224, 224), args.dense_onnx_path)
onnx_model = onnx.load(args.dense_onnx_path)
annotate_sparsity(onnx_model, args.data)
onnx.save(onnx_model, args.sparse_onnx_path)