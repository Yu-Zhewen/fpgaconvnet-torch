import copy
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

SPARSE_MODULES = (nn.Conv2d) # (nn.Conv2d, nn.Linear)

def moving_average(a, n):
    ret = torch.cumsum(a, dim=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

# stats collection on n windows, each window has k_2 elements
# enable_hist: whether to collect histogram
# enable_mvcc: whether to collect mean, var, cov, cor
# ma_size: moving average window size, 0 means disabled
class WindowDataCollector():
    def __init__(self, window_num, window_size, enable_hist, enable_mvcc, ma_size):
        self.count = 0
        self.window_num = window_num
        self.window_size = window_size

        self.enable_hist = enable_hist
        self.enable_mvcc = enable_mvcc
        self.ma_size = ma_size

        if self.enable_hist:
            self.hist = torch.zeros(window_num, window_size + 1)
            if torch.cuda.is_available():
                self.hist = self.hist.cuda()
        if self.enable_mvcc > 0:
            self.mean = torch.zeros(window_num)
            self.var  = torch.zeros(window_num)
            self.cov  = torch.zeros(window_num, window_num)
            self.cor  = torch.zeros(window_num, window_num)
            if torch.cuda.is_available():
                self.mean = self.mean.cuda()
                self.var  = self.var.cuda()
                self.cov  = self.cov.cuda()
                self.cor  = self.cor.cuda()
        if self.ma_size > 0:
            self.ma_buffer = None

    def _update_hist(self, newValues):
        # update hist
        zero_hists = F.one_hot(self.window_size - torch.count_nonzero(newValues, dim = -1), num_classes = self.window_size + 1)
        zero_hists = zero_hists.sum(dim=0)
        self.hist += zero_hists

    def _update_mvcc(self, newValues):
        num_of_zeros = self.window_size - torch.count_nonzero(newValues.reshape((-1, self.window_size)), dim=1)
        newValues = num_of_zeros.reshape((-1, self.window_num))

        if self.ma_size > 0:
            if self.ma_buffer is None:
                self.ma_buffer = newValues
            else:
                self.ma_buffer = torch.cat((self.ma_buffer, newValues), dim=0)
            if self.ma_buffer.size()[0] > self.ma_size:
                newValues = moving_average(self.ma_buffer, self.ma_size)
                self.ma_buffer = self.ma_buffer[-(self.ma_size-1):]

        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        # update mean, var, cov, cor
        self.var = self.var * self.count
        self.cov = self.cov * (self.count - 1)
        self.count += newValues.size()[0]
        delta = torch.subtract(newValues, self.mean.expand_as(newValues))
        self.mean += torch.sum(delta / self.count, dim=0)
        delta2 = torch.subtract(newValues, self.mean.expand_as(newValues))
        self.var += torch.sum(delta * delta2, dim=0)
        self.cov += torch.matmul(delta.T, delta2)
        self.var = self.var / self.count # note torch,var uses N-1 by default
        assert self.count > 1
        self.cov = self.cov / (self.count - 1)
        self.cor = self.cov / torch.sqrt(torch.matmul(self.var.unsqueeze(1), self.var.unsqueeze(0))) * (self.count-1) / self.count

    def update(self, newValues):
        # newvalues - (..., n, k_2)
        assert newValues.size()[1] == self.window_num
        assert newValues.size()[2] == self.window_size

        if self.enable_hist:
            self._update_hist(newValues)
        if self.enable_mvcc:
            self._update_mvcc(newValues)


# convolution layers with slide window emulation
class SlideWindowConvolution(nn.Module):
    def __init__(self, nn_conv, moving_average=0):
        super(SlideWindowConvolution, self).__init__()
        self.nn_conv = nn_conv # original nn.Conv2d instance
        self.moving_average = moving_average
        self.statistics = WindowDataCollector(nn_conv.in_channels, np.prod(nn_conv.kernel_size), True, False, moving_average)
        self.moving_average_statistics = None
        if moving_average > 0:
            self.moving_average_statistics = WindowDataCollector(nn_conv.in_channels, np.prod(nn_conv.kernel_size), False, True, moving_average)

    def forward(self, x):
        # zero padding 
        assert self.nn_conv.padding_mode == 'zeros'
        x_padded = F.pad(input=x, pad=self.nn_conv._reversed_padding_repeated_twice, mode='constant', value=0)
        
        # unfold, reshape the input x to windows
        dh, dw = self.nn_conv.stride
        out_channels, in_channels, kh, kw = self.nn_conv.weight.shape        
        groups = self.nn_conv.groups
        in_channels *= groups
        batch_size = x.shape[0]
        patches = x_padded.unfold(2, kh, dh).unfold(3, kw, dw)
        h_windows = patches.shape[2]
        w_windows = patches.shape[3]
        patches = patches.expand(out_channels//groups, *patches.shape) # dims = (out_channels//groups, batch_size, in_channels, h_windows, w_windows, kh, kw)
        patches = patches.permute(1, 3, 4, 0, 2, 5, 6) # dims = ( batch_size, h_windows, w_windows, out_channels//groups, in_channels, kh, kw)

        # accumulation buffer
        #y = torch.zeros((batch_size, h_windows, w_windows, out_channels))
        #if torch.cuda.is_available():
        #    y = y.cuda()

        # roll the loop to reduce GPU memory
        roll_factor = 7
        if h_windows % roll_factor != 0:
            roll_factor = 1
        if w_windows % roll_factor != 0:
            roll_factor = 1

        for hi, wi in np.ndindex(roll_factor, roll_factor):
            hstart = hi * (h_windows // roll_factor)
            hend   = (hi+1) * (h_windows // roll_factor)
            wstart = wi * (w_windows // roll_factor)
            wend   = (wi+1) * (w_windows // roll_factor)
            patch = patches[:,hstart:hend,wstart:wend].reshape((batch_size, h_windows//roll_factor, w_windows//roll_factor, out_channels//groups, groups, in_channels//groups, kh, kw))
            patch = patch.permute(0, 1, 2, 4, 3, 5, 6, 7) #(batch_size, h_windows//self.roll_factor, w_windows//self.roll_factor, groups, out_channels//groups, in_channels//groups, kh, kw)
            weight = self.nn_conv.weight.reshape((groups, out_channels//groups, in_channels//groups, kh, kw))
            patch = patch * weight
            window_product = patch.permute(0, 1, 2, 4, 3, 5, 6, 7)
            window_product = window_product.reshape((-1, in_channels, int(kh*kw)))
            self.statistics.update(window_product)
            if self.moving_average_statistics is not None:
                self.moving_average_statistics.update(window_product)

            # patch = patch.sum(-1).sum(-1).sum(-1)
            # patch = patch.reshape(batch_size, h_windows//self.roll_factor, w_windows//self.roll_factor, out_channels)
            # y[:,hstart:hend,wstart:wend] = patch

        return self.nn_conv(x)

def measure_model_sparsity(model_wrapper):
    replace_dict = {}
    named_sparse_modules = {}
    for name, module in model_wrapper.named_modules():
        if isinstance(module, SPARSE_MODULES):
            new_module = SlideWindowConvolution(copy.deepcopy(module))
            replace_dict[module] = new_module
            named_sparse_modules[name] = new_module
    model_wrapper.replace_modules(replace_dict)
    model_wrapper.inference("calibrate")
    
    # add sideband information
    model_wrapper.sideband_info["sparsity"] = {}
    for name, module in named_sparse_modules.items():
        model_wrapper.sideband_info["sparsity"][name] = {"hist": module.statistics.hist}

    # avg sparsity
    zeros = 0
    non_zeros = 0
    for name, module in named_sparse_modules.items():
        hist = module.statistics.hist
        for i in range(hist.shape[1]):
            zeros += (hist[:, i] * i).sum()
            non_zeros += (hist[:, i] * (hist.shape[1] - i)).sum()
    avg_sparsity = zeros / (zeros + non_zeros)
    return avg_sparsity.item()