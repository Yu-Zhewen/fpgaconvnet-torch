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

def replace_with_variable_relu(model, threshold=0, net=None):

    replace_dict = {}
    if (net == None):
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):#or isinstance(module, nn.Linear):
                new_module = VariableReLUWrapper(threshold)
                replace_dict[module] = new_module

    else:
        relus_to_replace = []
        previous_relu = None
        max_rate = 1 #Krish TODO: Change in case of skipping windows

        #Get list of ReLUs that are followed by rate bounded convolutiosns
        for partition_index in range(len(net.partitions)):
            partition = net.partitions[partition_index]
            for layer in graphs.ordered_node_list(partition.graph):
                #Keep track of preceding relu layer
                if isinstance(partition.graph.nodes[layer]['type'], list):
                    if LAYER_TYPE.ReLU in partition.graph.nodes[layer]['type']:
                        previous_relu = layer
                elif (partition.graph.nodes[layer]['type'] == LAYER_TYPE.ReLU):
                    previous_relu = layer

                #Check if layer is a Convolution layer tha can benefit from sparsit
                if (partition.graph.nodes[layer]['type'] == LAYER_TYPE.Convolution):
                    if partition.graph.nodes[layer]['hw'].modules['vector_dot'].rate_kernel_sparsity() < max_rate:
                        if previous_relu != None:
                            print(previous_relu)
                            layer_words = previous_relu.split("_")
                            i = -1
                            print(layer_words)
                            for idx in range(len(layer_words)):
                                if "Relu" in layer_words[idx]:
                                    i =idx
                            if i >= 2:
                                if "relu" in layer_words[i-1]:
                                    append_relu = ".".join(layer_words[i-2:i])
                                else:
                                    append_relu = ".".join(layer_words[i-1:i])
                                relus_to_replace.append(append_relu)

        print("Relus to replace:", relus_to_replace)
        for name, module in model.named_modules():
            if name in relus_to_replace:
                print("Replacing layer:", name)
                new_module = VariableReLUWrapper(threshold)
                replace_dict[module] = new_module

    replace_modules(model, replace_dict)
    # for name, module in model.named_modules():
    #     print(type(module))

def output_accuracy_to_csv(arch, relu_threshold, layer_name, top1, top5, output_dir):
    output_path = output_dir + "/" + str(layer_name) + ".csv"
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        with open(output_path, mode='w') as f:
            row = "Network,ReLU_Threshold,Variable,Top1_Accuracy,Top5_Accuracy\n"
            f.write(row)
    with open(output_path, mode='a') as f:
        row =  ','.join([arch, str(relu_threshold), str(layer_name), top1, top5]) + "\n"
        print("Writing to csv")
        f.write(row)
