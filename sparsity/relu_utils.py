import torch
from torch import nn

RELU_MODULES = (nn.ReLU, nn.ReLU6)

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

def apply_threshold_relu(model_wrapper, threshold=None):
    info = model_wrapper.sideband_info['threshold_relu'] if threshold is None else {}
    replace_dict = {}
    for name, module in model_wrapper.named_modules():
        if isinstance(module, RELU_MODULES):
            if threshold is None:
                new_module = ThresholdedReLULayer(info[name])
            else:
                new_module = ThresholdedReLULayer(threshold)
                info[name] = threshold
            replace_dict[module] = new_module
    model_wrapper.replace_modules(replace_dict)
    model_wrapper.sideband_info['threshold_relu'] = info