import onnx
import argparse
import csv

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

parser = argparse.ArgumentParser(description='Add Sparity Attribute to ONNX file')
parser.add_argument('--model', metavar='DIR', default="models/resnet18_residual_removed.onnx",
                    help='path to onnx model')
parser.add_argument('--data', metavar='DIR', default="output/resnet18_sparsity_log.csv",
                    help='path to onnx model')                
parser.add_argument('--output', metavar='DIR', default="models/resnet18_residual_removed_sparse.onnx",
                    help='path to onnx model')      

args = parser.parse_args()

with open(args.data, 'r') as f:
    csv_read = csv.reader(f)
    start = False
    sparsity_data = []
    for row in csv_read:
        if start:
            sparsity_data.append(float(row[2]))
        if len(row) >= 3 and row[2] == 'Accum Input Sparsity (Zeros / im2col(NCHW))':
            start = True

model = onnx.load(args.model)

conv_layer_index = 0
for node in model.graph.node:
    if node.op_type == 'Conv':
        set_nodeattr(node, "input sparsity", sparsity_data[conv_layer_index])
        conv_layer_index += 1

assert conv_layer_index == len(sparsity_data)

onnx.save(model, args.output)