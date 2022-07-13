import torch
import torch.nn as nn

import csv

from utils import *
from functools import reduce

def get_factors(n):
    return list(set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))

def is_coarse_in_feasible(module, coarse_in):
    if isinstance(module, nn.Conv2d):
        in_channels = module.in_channels
    elif isinstance(module, nn.Linear):
        in_channels = module.in_features
    
    return coarse_in in get_factors(in_channels)

def regsiter_hooks(model, coarse_in=-1):
    hist_bin_edges = torch.linspace(0, 1, 11)

    def log_conv_input_sparsity(m, input, output):
        conv_input = input[0].detach()
        batch_size = conv_input.size()[0]

        if m.coarse_in == -1:
            num_of_elements = torch.numel(conv_input)
            num_of_zeros = num_of_elements - torch.count_nonzero(conv_input)
            layer_sparsity = num_of_zeros / num_of_elements
            m.layer_sparsity.update(layer_sparsity, batch_size)
        else:
            conv_input = conv_input.transpose(0,1)
            conv_input = conv_input.reshape((m.coarse_in, -1))
            num_of_zeros = m.coarse_in - torch.count_nonzero(conv_input, dim=0)
            coarse_in_sparsity = num_of_zeros / m.coarse_in
            m.coarse_in_sparsity_hist += torch.histogram(coarse_in_sparsity.cpu(), bins=hist_bin_edges)[0]

    handle_list = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.coarse_in = coarse_in
            if coarse_in == -1:
                module.layer_sparsity = AverageMeter('Layer Sparsity', ':.4e')
            elif is_coarse_in_feasible(module, coarse_in):
                module.coarse_in_sparsity_hist = torch.zeros(len(hist_bin_edges)-1)
            else:
                continue
            handle = module.register_forward_hook(log_conv_input_sparsity)
            handle_list.append(handle)

    return handle_list

def output_sparsity_to_csv(model_name, model):
    file_path = "{}_sparsity_log.csv".format(model_name)


    bFirst = True
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if bFirst:
                bFirst = False
                with open(file_path, mode='a') as f:
                    csv_writer = csv.writer(f)
                    csv_header= ["Layer Name", "Layer Type"]
                    if module.coarse_in == -1:
                        csv_header += ["Layer Input Overall Sparsity (Zeros / NCHW)"]
                    else:
                        csv_header += ["COARSE_IN", "Coarse In Sparsity Histogram (Zeros / COARSE_IN)"]
                    csv_writer.writerow(csv_header)

            with open(file_path, mode='a') as f:
                csv_writer = csv.writer(f)
                new_row = [name, type(module)]
                if module.coarse_in == -1:
                    new_row += [module.layer_sparsity.avg.item()]
                elif is_coarse_in_feasible(module, module.coarse_in):
                    new_row += [module.coarse_in, module.coarse_in_sparsity_hist.type(torch.int64).tolist()]
                csv_writer.writerow(new_row)

def delete_hooks(model, handle_list):
    for handle in handle_list:
        handle.remove() 

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if hasattr(module, "layer_sparsity"):
                del module.layer_sparsity
            if hasattr(module, "coarse_in_sparsity_hist"):
                del module.coarse_in_sparsity_hist