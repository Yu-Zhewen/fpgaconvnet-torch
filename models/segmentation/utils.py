import math
import torch

from models.utils import replace_modules
from torch import nn

CONV_TRANSP_MODULES = (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)

class ConvTranspApproxLayer(nn.Module):
    def __init__(self, parent_module, upsampling_mode, kernel_approx_strategy):
        super().__init__()
        self.device = parent_module.weight.device
        self.has_bias = True if isinstance(parent_module.bias, nn.Parameter) else False

        self.upsampling_mode = upsampling_mode
        self.kernel_approx_strategy = kernel_approx_strategy

        # TODO: Here we assume a random value for the input spatial dimension (1st dim). We also assume that the stride, padding, kernel_size and output_padding are the same for all dimensions. If this is not the case, we need to change the code below.
        rand_spatial_dim = 128
        self.scale_factor = math.ceil(((rand_spatial_dim - 1) * parent_module.stride[0] - 2 * parent_module.padding[0] +
                                      parent_module.kernel_size[0] + parent_module.output_padding[0]) / rand_spatial_dim)

        self.upsample = nn.Upsample(
            scale_factor=self.scale_factor, mode=self.upsampling_mode).to(self.device)
        match parent_module._get_name():
            case "ConvTranspose1d":
                self.pointwise_conv = nn.Conv1d(
                    in_channels=parent_module.in_channels, out_channels=parent_module.out_channels, kernel_size=1, bias=self.has_bias).to(self.device)

                if self.kernel_approx_strategy == "average":
                    weights = parent_module.weight.data.permute(1, 0, 2)
                    weights = torch.mean(weights, dim=(2), keepdim=True)
                else:
                    raise NotImplementedError

            case "ConvTranspose2d":
                self.pointwise_conv = nn.Conv2d(
                    in_channels=parent_module.in_channels, out_channels=parent_module.out_channels, kernel_size=1, bias=self.has_bias).to(self.device)

                if self.kernel_approx_strategy == "average":
                    weights = parent_module.weight.data.permute(1, 0, 2, 3)
                    weights = torch.mean(weights, dim=(2, 3), keepdim=True)
                else:
                    raise NotImplementedError

            case "ConvTranspose3d":
                self.pointwise_conv = nn.Conv3d(
                    in_channels=parent_module.in_channels, out_channels=parent_module.out_channels, kernel_size=1, bias=self.has_bias).to(self.device)

                if self.kernel_approx_strategy == "average":
                    weights = parent_module.weight.data.permute(1, 0, 2, 3, 4)
                    weights = torch.mean(weights, dim=(2, 3, 4), keepdim=True)
                else:
                    raise NotImplementedError

        self.pointwise_conv.weight.data.copy_(weights)
        if self.has_bias:
            self.pointwise_conv.bias.data.copy_(parent_module.bias.data)

    def forward(self, x):
        x = self.upsample(x)
        x = self.pointwise_conv(x)

        return x

def apply_conv_transp_approx(model, upsampling_mode="bilinear", kernel_approx_strategy="average"):
    replace_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, CONV_TRANSP_MODULES):
            new_module = ConvTranspApproxLayer(
                parent_module=module, upsampling_mode=upsampling_mode, kernel_approx_strategy=kernel_approx_strategy)
            replace_dict[module] = new_module
    replace_modules(model, replace_dict)