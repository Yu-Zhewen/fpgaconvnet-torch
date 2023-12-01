import torch
import torch.nn as nn

WEIGHT_PRUNE_MODULES = (nn.Conv2d, nn.Conv3d, nn.Linear, nn.ConvTranspose2d, nn.ConvTranspose3d)

def apply_weight_pruning(model_wrapper, threshold):
    for name, module in model_wrapper.named_modules():
        if isinstance(module, WEIGHT_PRUNE_MODULES):
            module.weight.data = torch.where(
                torch.abs(module.weight.data) < threshold, 
                torch.tensor(0.0).to(module.weight.device), 
                module.weight.data)
        