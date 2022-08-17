import torch
import torch.nn as nn

import csv
import copy

from utils import *
from functools import reduce
import torch.nn.functional as F

import numpy as np

def get_factors(n):
    return list(set(reduce(list.__add__,
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0))))

def is_coarse_in_feasible(module, coarse_in):
    if isinstance(module, nn.Conv2d):
        in_channels = module.in_channels
    #elif isinstance(module, nn.Linear):
    #    in_channels = module.in_features
    
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
            m.sparsity_hist += torch.histogram(coarse_in_sparsity.cpu(), bins=hist_bin_edges)[0]

    handle_list = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) :#or isinstance(module, nn.Linear):
            module.coarse_in = coarse_in
            if coarse_in == -1:
                module.layer_sparsity = AverageMeter('Layer Sparsity', ':.4e')
            elif is_coarse_in_feasible(module, coarse_in):
                module.sparsity_hist = torch.zeros(len(hist_bin_edges)-1)
            else:
                continue
            #handle = module.register_forward_hook(log_conv_input_sparsity)
            #handle_list.append(handle)

    return handle_list

def output_sparsity_to_csv(model_name, model, accum_input=False):
    file_path = "{}_sparsity_log.csv".format(model_name)

    bFirst = True
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) :#or isinstance(module, nn.Linear):
            if bFirst:
                bFirst = False
                with open(file_path, mode='a') as f:
                    csv_writer = csv.writer(f)
                    csv_header= ["Layer Name", "Layer Type"]
                    if accum_input:
                        if not module.kernel_distribution:
                            csv_header += ["Accum Input Sparsity (Zeros / im2col(NCHW))"]
                        else:
                            csv_header += ["KERNEL*KERNEL", "Accum Input Kernel Sparsity Histogram (Zeros / im2col(KK))"]
                    else:
                        if module.coarse_in == -1:
                            csv_header += ["Layer Input Overall Sparsity (Zeros / NCHW)"]
                        else:
                            csv_header += ["COARSE_IN", "Layer Input Coarse In Sparsity Histogram (Zeros / COARSE_IN)"]

                            
                    csv_writer.writerow(csv_header)

            with open(file_path, mode='a') as f:
                csv_writer = csv.writer(f)
                new_row = [name, type(module)]
                if accum_input:
                    if not module.kernel_distribution:
                        new_row += [module.layer_sparsity.avg.item()]
                    else:
                        new_row += [module.kk, module.sparsity_hist.type(torch.int64).tolist()]
                else:
                    if module.coarse_in == -1:
                        new_row += [module.layer_sparsity.avg.item()]
                    else:
                        new_row += [module.coarse_in, module.sparsity_hist.type(torch.int64).tolist()]
                    
                csv_writer.writerow(new_row)

def delete_hooks(model, handle_list):
    for handle in handle_list:
        handle.remove() 

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) :#or isinstance(module, nn.Linear):
            if hasattr(module, "layer_sparsity"):
                del module.layer_sparsity
            if hasattr(module, "sparsity_hist"):
                del module.sparsity_hist

class VanillaConvolutionWrapper(nn.Module):
    def __init__(self, conv_module, kernel_distribution=False):
        super(VanillaConvolutionWrapper, self).__init__()

        self.conv_module = conv_module
        self.conv_module.kernel_distribution = kernel_distribution
        self.run_reference = False
        self.conv_module.kk = np.prod(self.conv_module.kernel_size)
        self.hist_bin_edges = torch.linspace(0, self.conv_module.kk+1, self.conv_module.kk+2)
        self.conv_module.sparsity_hist = torch.zeros(len(self.hist_bin_edges)-1)

    def forward(self, x):

        # https://discuss.pytorch.org/t/make-custom-conv2d-layer-efficient-wrt-speed-and-memory/70175
        assert self.conv_module.padding_mode == 'zeros'
        x_padded = F.pad(input=x, pad=self.conv_module._reversed_padding_repeated_twice, mode='constant', value=0)

        dh, dw = self.conv_module.stride
        out_channels, in_channels, kh, kw = self.conv_module.weight.shape
        groups = self.conv_module.groups
        in_channels *= groups
        batch_size = x.shape[0]

        patches = x_padded.unfold(2, kh, dh).unfold(3, kw, dw)
        h_windows = patches.shape[2]
        w_windows = patches.shape[3]
        patches = patches.expand(out_channels//groups, *patches.shape)
        patches = patches.permute(1, 3, 4, 0, 2, 5, 6)
        num_of_elements = torch.numel(patches)

        num_of_nonzeros = 0
        y = torch.zeros((batch_size, h_windows, w_windows, out_channels))
        if torch.cuda.is_available():
            y = y.cuda()

        # roll the loop to reduce memory
        for hi, wi in np.ndindex(patches.shape[1:3]):
            patch = patches[:,hi,wi].reshape((batch_size, out_channels//groups, groups, in_channels//groups, kh, kw))
            patch = patch.permute(0, 2, 1, 3, 4, 5)
            weight = self.conv_module.weight.reshape((groups, out_channels//groups, in_channels//groups, kh, kw))
            patch = patch * weight

            if not self.conv_module.kernel_distribution:    
                num_of_nonzeros += torch.count_nonzero(patch)
            else:
                tmp = patch.reshape((-1, self.conv_module.kk))
                num_of_zeros = self.conv_module.kk - torch.count_nonzero(tmp, dim=1)
                self.conv_module.sparsity_hist += torch.histogram(num_of_zeros.float().cpu(), bins=self.hist_bin_edges)[0]

            patch = patch.sum(-1).sum(-1).sum(-1)
            patch = patch.reshape(batch_size, out_channels)

            y[:,hi,wi] = patch

        if not self.conv_module.kernel_distribution:    
            num_of_zeros = num_of_elements - num_of_nonzeros
            layer_sparsity = num_of_zeros / num_of_elements
            self.conv_module.layer_sparsity.update(layer_sparsity, batch_size)

        if self.conv_module.bias is not None:
            bias = self.conv_module.bias.expand(batch_size, h_windows, w_windows, out_channels)
            y = y + bias
        y = y.permute(0, 3, 1, 2)

        if self.run_reference:
            ref_output = self.conv_module(x)
            assert torch.allclose(ref_output, y, atol=1e-5)

        return y

def replace_with_vanilla_convolution(model):
    replace_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            replace_dict[module] = VanillaConvolutionWrapper(copy.deepcopy(module))

    replace_modules(model, replace_dict)