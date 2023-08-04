from torch import nn
from utils import *
import fpgaconvnet.tools.graphs as graphs
from fpgaconvnet.tools.layer_enum import LAYER_TYPE
import os

class VariableReLUWrapper(nn.Module):
    def __init__(self, relu_threshold):
        super(VariableReLUWrapper, self).__init__()

        self.threshold = relu_threshold

    def forward(self, x):
        return (x > self.threshold)*x

def replace_layer_with_variable_relu(model, layer_name, threshold=0):

    replace_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU) and name == layer_name:#or isinstance(module, nn.Linear):
            new_module = VariableReLUWrapper(threshold)
            replace_dict[module] = new_module
            break
    replace_modules(model, replace_dict)

def replace_with_variable_relu(model, threshold=0):

    replace_dict = {}
    relu_thresholds = {}

    if isinstance(threshold, dict):
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):#or isinstance(module, nn.Linear):
                new_module = VariableReLUWrapper(threshold[name])
                replace_dict[module] = new_module

    else:
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):#or isinstance(module, nn.Linear):
                new_module = VariableReLUWrapper(threshold)
                replace_dict[module] = new_module
                relu_thresholds[name] = threshold


        print("Relus to replace:", relu_thresholds)
        for name, module in model.named_modules():
            if name in relu_thresholds:
                new_module = VariableReLUWrapper(threshold)
                replace_dict[module] = new_module
            elif isinstance(module, VariableReLUWrapper):
                new_module = VariableReLUWrapper(threshold)
                replace_dict[module] = new_module

    replace_modules(model, replace_dict)
    return relu_thresholds
    # for name, module in model.named_modules():
    #     print(type(module))

def output_accuracy_to_csv(arch, relu_threshold, top1, top5, sparsity, output_path):
    if not os.path.isfile(output_path):
        with open(output_path, mode='w') as f:
            row = "Network,ReLU_Threshold,Top1_Accuracy,Top5_AccuracyNetwork_Sparsity\n"
            f.write(row)
    with open(output_path, mode='a') as f:
        row =  ','.join([arch, str(relu_threshold), top1, top5, str(sparsity)]) + "\n"
        print("Writing to csv")
        f.write(row)