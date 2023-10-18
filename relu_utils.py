import torch
from torch import nn
from utils import replace_modules

class ThresholdedReLULayer(nn.Module):
    def __init__(self, lower_threshold, upper_threshold=None):
        super(ThresholdedReLULayer, self).__init__()
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold 

    def forward(self, x):
        if self.upper_threshold is not None:
            x =  torch.clip(x, max = self.upper_threshold)
            return torch.where(x > self.lower_threshold, x, 0.0)
        else:
            return torch.where(x > self.lower_threshold, x, 0.0)

def replace_with_threshold_relu(model, threshold={}):   
    replace_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU) or isinstance(module, nn.ReLU6):
            new_module = ThresholdedReLULayer(threshold[name])
            replace_dict[module] = new_module
    replace_modules(model, replace_dict)