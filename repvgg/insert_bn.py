import torch
import torch.nn as nn
from hybrid_svd.common.repvgg.repvgg import RepVGGBlock

def update_running_mean_var(x, running_mean, running_var, momentum=0.9, is_first_batch=False):
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    if is_first_batch:
        running_mean = mean
        running_var = var
    else:
        running_mean = momentum * running_mean + (1.0 - momentum) * mean
        running_var = momentum * running_var + (1.0 - momentum) * var
    return running_mean, running_var

#   Record the mean and std like a BN layer but do no normalization
class BNStatistics(nn.Module):
    def __init__(self, num_features):
        super(BNStatistics, self).__init__()
        shape = (1, num_features, 1, 1)
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.zeros(shape))
        self.is_first_batch = True

    def forward(self, x):
        if self.running_mean.device != x.device:
            self.running_mean = self.running_mean.to(x.device)
            self.running_var = self.running_var.to(x.device)
        self.running_mean, self.running_var = update_running_mean_var(x, self.running_mean, self.running_var, momentum=0.9, is_first_batch=self.is_first_batch)
        self.is_first_batch = False
        return x

#   This is designed to insert BNStat layer between Conv2d(without bias) and its bias
class BiasAdd(nn.Module):
    def __init__(self, num_features):
        super(BiasAdd, self).__init__()
        self.bias = torch.nn.Parameter(torch.Tensor(num_features))
    def forward(self, x):
        return x + self.bias.view(1, -1, 1, 1)

def switch_repvggblock_to_bnstat(model):
    for n, block in model.named_modules():
        if isinstance(block, RepVGGBlock):
            #print('switch to BN Statistics: ', n)
            assert hasattr(block, 'rbr_reparam')
            stat = nn.Sequential()
            stat.add_module('conv', nn.Conv2d(block.rbr_reparam.in_channels, block.rbr_reparam.out_channels,
                                              block.rbr_reparam.kernel_size,
                                              block.rbr_reparam.stride, block.rbr_reparam.padding,
                                              block.rbr_reparam.dilation,
                                              block.rbr_reparam.groups, bias=False))  # Note bias=False
            stat.add_module('bnstat', BNStatistics(block.rbr_reparam.out_channels))
            stat.add_module('biasadd', BiasAdd(block.rbr_reparam.out_channels))  # Bias is here
            stat.conv.weight.data = block.rbr_reparam.weight.data
            stat.biasadd.bias.data = block.rbr_reparam.bias.data
            block.__delattr__('rbr_reparam')
            block.rbr_reparam = stat

def switch_bnstat_to_convbn(model):
    for n, block in model.named_modules():
        if isinstance(block, RepVGGBlock):
            assert hasattr(block, 'rbr_reparam')
            assert hasattr(block.rbr_reparam, 'bnstat')
            #print('switch to ConvBN: ', n)
            conv = nn.Conv2d(block.rbr_reparam.conv.in_channels, block.rbr_reparam.conv.out_channels,
                             block.rbr_reparam.conv.kernel_size,
                             block.rbr_reparam.conv.stride, block.rbr_reparam.conv.padding,
                             block.rbr_reparam.conv.dilation,
                             block.rbr_reparam.conv.groups, bias=False)
            bn = nn.BatchNorm2d(block.rbr_reparam.conv.out_channels)
            bn.running_mean = block.rbr_reparam.bnstat.running_mean.squeeze()  # Initialize the mean and var of BN with the statistics
            bn.running_var = block.rbr_reparam.bnstat.running_var.squeeze()
            std = (bn.running_var + bn.eps).sqrt()
            conv.weight.data = block.rbr_reparam.conv.weight.data
            bn.weight.data = std
            bn.bias.data = block.rbr_reparam.biasadd.bias.data + bn.running_mean  # Initialize gamma = std and beta = bias + mean

            convbn = nn.Sequential()
            convbn.add_module('conv', conv)
            convbn.add_module('bn', bn)
            block.__delattr__('rbr_reparam')
            block.rbr_reparam = convbn