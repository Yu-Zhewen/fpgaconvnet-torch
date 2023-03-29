import torch
import torch.nn as nn

from utils import replace_modules

class VariableReLUWrapper(nn.Module):
    def __init__(self, threshold):
        self.threshold = threshold

    def forward(self, x):
        """
        Parameters
        ----------
        x: tensor
            Data input to layer

        Returns
        -------
        tensor
            output after applying ReLU with variable threshold
        """

        return (x >= self.threshold)*x
        
def replace_with_variable_relu(model, threshold = 0):
    replace_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            new_module = VariableReLUWrapper(threshold)
            replace_dict[module] = new_module

    replace_modules(model, replace_dict)
