import torch
import torch.nn as nn

import csv

from utils import *

def sparsity_target_filter(module):
    return isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)

def regsiter_hooks(model):
    def log_conv_input_sparsity(m, input, output):
        conv_input = input[0].detach()
        batch_size = conv_input.size()[0]

        num_of_elements = torch.numel(conv_input)
        num_of_zeros = num_of_elements - torch.count_nonzero(conv_input)
        layer_sparsity = num_of_zeros / num_of_elements

        m.layer_sparsity.update(layer_sparsity, batch_size)

    handle_list = []
    for name, module in model.named_modules():
        if sparsity_target_filter(module):
            module.layer_sparsity = AverageMeter('Layer Sparsity', ':.4e')
            handle = module.register_forward_hook(log_conv_input_sparsity)
            handle_list.append(handle)

    return handle_list

def output_sparsity_to_csv(model_name, model):
    file_path = "{}_sparsity_log.csv".format(model_name)
    with open(file_path, mode='w') as f:
        csv_writer = csv.writer(f)
        csv_header = ["Layer Name", "Layer Type", "Layer Input Overall Sparsity (Zeros / NCHW)"]
        csv_writer.writerow(csv_header)

    for name, module in model.named_modules():
        if sparsity_target_filter(module):
            with open(file_path, mode='a') as f:
                csv_writer = csv.writer(f)
                new_row = [name, type(module), module.layer_sparsity.avg.item()]
                csv_writer.writerow(new_row)

def delete_hooks(model, handle_list):
    for handle in handle_list:
        handle.remove() 

    for name, module in model.named_modules():
        if sparsity_target_filter(module):
            del module.layer_sparsity