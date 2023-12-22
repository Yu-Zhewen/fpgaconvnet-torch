import copy
import os
import torch
import torch.nn as nn

from enum import Enum

# todo: support more methods
class QuantMode(Enum):
    NETWORK_FP = 1
    LAYER_BFP = 2
    CHANNEL_BFP = 3

ACTIVA_QUANT_MODULES = (nn.Conv2d, nn.Conv3d, nn.Linear, 
                        nn.ConvTranspose2d, nn.ConvTranspose3d, 
                        nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.Hardswish,
                        nn.MaxPool2d, nn.MaxPool3d, 
                        nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d, 
                        nn.AvgPool2d, nn.AvgPool3d)
WEIGHT_QUANT_MODULES = (nn.Conv2d, nn.Conv3d, nn.Linear, 
                        nn.ConvTranspose2d, nn.ConvTranspose3d)

def linear_quantize(x, word_length, scaling_factor, zero_point):
    if len(x.shape) == 5:
        scaling_factor = scaling_factor.view(-1, 1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1, 1)
    elif len(x.shape) == 4:
        scaling_factor = scaling_factor.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    elif len(x.shape) == 2:
        scaling_factor = scaling_factor.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        assert False
    x_quant = torch.round(scaling_factor * x - zero_point)
    x_quant = saturate(x_quant, word_length)
    return x_quant

def linear_dequantize(x_quant, scaling_factor, zero_point):
    if len(x_quant.shape) == 5:
        scaling_factor = scaling_factor.view(-1, 1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1, 1)
    elif len(x_quant.shape) == 4:
        scaling_factor = scaling_factor.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    elif len(x_quant.shape) == 2:
        scaling_factor = scaling_factor.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        assert False
    x = (x_quant + zero_point) / scaling_factor
    return x

#Asymmetric Quantiation: x_q = round((x_f - min_xf) * (2^n - 1) / (max_xf - min_xf))
def asymmetric_linear_no_clipping(word_length, x_min, x_max):
    scaling_factor = (2**word_length - 1) / torch.clamp((x_max - x_min), min=1e-8) # Calculates scaling factor as shown in equation for function above
    zero_point = scaling_factor * x_min #Corresponds to most negative value represented by wlen-bit
    if isinstance(zero_point, torch.Tensor):
        zero_point = zero_point.round()
    else:
        zero_point = float(round(zero_point))
    zero_point += 2**(word_length - 1) #Corresponds to zero by adding 2^(wlen - 1)
    return scaling_factor, zero_point

def saturate(x_quant, word_length):
    n = 2**(word_length - 1)
    x_quant = torch.clamp(x_quant, -n, n - 1)
    return x_quant


class ModelParamQuantizer():
    def __init__(self, model_wrapper):
        weight_min = torch.tensor(float('inf'))
        weight_max = torch.tensor(float('-inf'))
        if torch.cuda.is_available():
            weight_min = weight_min.cuda()
            weight_max = weight_max.cuda()
        for module in model_wrapper.modules():
            if isinstance(module, WEIGHT_QUANT_MODULES):
                weight_min = torch.minimum(torch.min(module.weight), weight_min)
                weight_max = torch.maximum(torch.max(module.weight), weight_max)
        self.w_min = weight_min
        self.w_max = weight_max
        print("network weight min:", self.w_min)
        print("network weight max:", self.w_max)

    def apply(self, w, word_length, mode):
        if mode == QuantMode.CHANNEL_BFP:
            w_block = w.data.contiguous().view(w.size()[0], -1)
            w_min = w_block.min(dim=1)[0]
            w_max = w_block.max(dim=1)[0]
        elif mode == QuantMode.LAYER_BFP:
            w_min = w.data.min()
            w_max = w.data.max()
        elif mode == QuantMode.NETWORK_FP:
            w_min = torch.tensor([self.w_min])
            w_max = torch.tensor([self.w_max])
            w_min = w_min.to(w.device)
            w_max = w_max.to(w.device)
        scaling_factor, zero_point = asymmetric_linear_no_clipping(word_length, w_min, w_max)
        w_quant = linear_quantize(w, word_length, scaling_factor, zero_point)
        w_approx = linear_dequantize(w_quant, scaling_factor, zero_point)
        return w_approx, scaling_factor, zero_point

class QuantAct(nn.Module):
    def __init__(self,
                 word_length,
                 mode):

        super(QuantAct, self).__init__()
        self.word_length = word_length
        self.mode = mode
        self.calibrate = True
        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        self.register_buffer('scaling_factor', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))

    def get_scale_shift(self):
        self.scaling_factor, self.zero_point = asymmetric_linear_no_clipping(self.word_length, self.x_min, self.x_max)

    def forward(self, x):
        if self.calibrate: #Collects data about the x_min and x_max to quantise the input features
            if self.mode == QuantMode.CHANNEL_BFP:
                channel_num = x.size()[1]
                x_block = x.data.transpose(0, 1)
                x_block = x_block.contiguous().view(channel_num, -1)
                x_min = x_block.min(dim=1)[0]
                x_max = x_block.max(dim=1)[0]
                if self.x_min.size()[0] == 1:
                    self.x_min = torch.zeros(channel_num)
                    self.x_max = torch.zeros(channel_num)
                    self.x_min = self.x_min.to(self.scaling_factor.device)
                    self.x_max = self.x_max.to(self.scaling_factor.device)
            else:
                x_min = x.data.min()
                x_max = x.data.max()
            # in-place operation used on multi-gpus
            self.x_min += -self.x_min + torch.minimum(self.x_min, x_min)
            self.x_max += -self.x_max + torch.maximum(self.x_max, x_max)
            return x
        else:
            if self.mode == QuantMode.CHANNEL_BFP:
                x = x.transpose(0, 1)
            x_quant = linear_quantize(x, self.word_length, self.scaling_factor, self.zero_point)
            x_quant = linear_dequantize(x_quant, self.scaling_factor, self.zero_point)
            if self.mode == QuantMode.CHANNEL_BFP:
                x_quant = x_quant.transpose(0, 1)
            return x_quant

class ModelActQuantizer():
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper

    def apply(self, word_length, mode):
        # add activation quantisation module
        replace_dict = {}
        name_list = []
        for name, module in self.model_wrapper.named_modules():
            if isinstance(module, ACTIVA_QUANT_MODULES):
                module_quant = nn.Sequential(*[QuantAct(word_length, mode), copy.deepcopy(module), QuantAct(word_length, mode)])
                replace_dict[module] = module_quant
                name_list.append(name)
        self.model_wrapper.replace_modules(replace_dict)
        if torch.cuda.is_available():
            self.model_wrapper.cuda()

        # gather activation data
        self.model_wrapper.inference("calibrate")
        for name, module in self.model_wrapper.named_modules():
            if isinstance(module, QuantAct):
                module.calibrate = False
        act_min = torch.tensor(float('inf'))
        act_max = torch.tensor(float('-inf'))
        if torch.cuda.is_available():
            act_min = act_min.cuda()
            act_max = act_max.cuda()
        for module in self.model_wrapper.modules():
            if isinstance(module, QuantAct):
                act_min = torch.minimum(torch.min(module.x_min), act_min)
                act_max = torch.maximum(torch.max(module.x_max), act_max)
        print("activation min:", act_min)
        print("activation max:", act_max)
        
        # set scale and shift
        for i, seq in enumerate(replace_dict.values()):
            assert isinstance(seq, nn.Sequential)
            input_quant = seq[0]
            output_quant = seq[2]
            if mode == QuantMode.NETWORK_FP:
                input_quant.x_min = act_min
                input_quant.x_max = act_max
                output_quant.x_min = act_min
                output_quant.x_max = act_max
            input_quant.get_scale_shift()
            output_quant.get_scale_shift()
            
            # save to info
            name = name_list[i]
            if name not in self.model_wrapper.sideband_info['quantization']:
                self.model_wrapper.sideband_info['quantization'][name] = {}
            self.model_wrapper.sideband_info['quantization'][name]["input_scale"] = input_quant.scaling_factor
            self.model_wrapper.sideband_info['quantization'][name]["input_zero_point"] = input_quant.zero_point
            self.model_wrapper.sideband_info['quantization'][name]["output_scale"] = output_quant.scaling_factor
            self.model_wrapper.sideband_info['quantization'][name]["output_zero_point"] = output_quant.zero_point


def quantize_model(model_wrapper, info):
    model_wrapper.sideband_info['quantization'] = info
    weight_quantizer = ModelParamQuantizer(model_wrapper)
    for name, module in model_wrapper.named_modules():
        if isinstance(module, WEIGHT_QUANT_MODULES):
            if isinstance(module, nn.ConvTranspose2d):
                weight = module.weight.data.transpose(0, 1)
            else:
                weight = module.weight.data
            
            quantized_weight, w_scale, w_zero_point = weight_quantizer.apply(weight, info["weight_width"], info["mode"])
            model_wrapper.sideband_info['quantization'][name] = {"weight_scale": w_scale, "weight_zero_point": w_zero_point}

            if isinstance(module, nn.ConvTranspose2d):
                module.weight.data.copy_(quantized_weight.transpose(0, 1))
            else:
                module.weight.data.copy_(quantized_weight)
    
    activation_quantizer = ModelActQuantizer(model_wrapper)
    activation_quantizer.apply(info["data_width"], info["mode"])
