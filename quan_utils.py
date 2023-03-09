import torch
import torch.nn as nn
import copy
from enum import Enum

from utils import *

class QuanMode(Enum):
    NETWORK_FP = 1
    LAYER_BFP = 2
    CHANNEL_BFP = 3

QUAN_TARGET_MODULES = [nn.Conv2d, nn.ReLU, nn.ReLU6, nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.Linear]

def linear_quantize(x, scaling_factor, zero_point):
    if len(x.shape) == 4:
        scaling_factor = scaling_factor.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    elif len(x.shape) == 2:
        scaling_factor = scaling_factor.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        assert False

    x_quan = torch.round(scaling_factor * x - zero_point)

    return x_quan


def linear_dequantize(x_quan, scaling_factor, zero_point):
    if len(x_quan.shape) == 4:
        scaling_factor = scaling_factor.view(-1, 1, 1, 1)
        zero_point = zero_point.view(-1, 1, 1, 1)
    elif len(x_quan.shape) == 2:
        scaling_factor = scaling_factor.view(-1, 1)
        zero_point = zero_point.view(-1, 1)
    else:
        assert False

    x = (x_quan + zero_point) / scaling_factor

    return x 


def asymmetric_linear_no_clipping(wordlength, x_min, x_max):

    scaling_factor = (2**wordlength - 1) / torch.clamp((x_max - x_min), min=1e-8)
    zero_point = scaling_factor * x_min

    if isinstance(zero_point, torch.Tensor):
        zero_point = zero_point.round()
    else:
        zero_point = float(round(zero_point))

    zero_point += 2**(wordlength - 1)

    return scaling_factor, zero_point

def saturate(w_quan, wordlength):
    n = 2**(wordlength - 1)
    w_quan = torch.clamp(w_quan, -n, n - 1)

    return w_quan

class WeightQuantizer():
    def __init__(self, model):
        bFirst = True

        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                if bFirst:
                    bFirst = False
                    self.w_min = torch.min(module.weight)
                    self.w_max = torch.max(module.weight)
                else:
                    self.w_min = torch.minimum(self.w_min, torch.min(module.weight))
                    self.w_max = torch.maximum(self.w_max, torch.max(module.weight))
    
        print("weight min:", self.w_min)
        print("weight max:", self.w_max)

    def AsymmetricQuantHandler(self, w, wordlength, quantization_method):
        if quantization_method == QuanMode.CHANNEL_BFP:
            filter_num = w.size()[0]
            w_block = w.data.contiguous().view(filter_num, -1)
            w_min = w_block.min(dim=1)[0]
            w_max = w_block.max(dim=1)[0]
        
        elif quantization_method == QuanMode.LAYER_BFP:
            w_min = w.data.min()
            w_max = w.data.max()
        
        elif quantization_method == QuanMode.NETWORK_FP:
            w_min = torch.tensor([self.w_min])
            w_max = torch.tensor([self.w_max])
            w_min = w_min.to(w.device)
            w_max = w_max.to(w.device)

        scaling_factor, zero_point = asymmetric_linear_no_clipping(wordlength, w_min, w_max)
        w_quan = linear_quantize(w, scaling_factor, zero_point)
        w_quan = saturate(w_quan, wordlength)         
        w_approx = linear_dequantize(w_quan, scaling_factor, zero_point)

        return w_approx


class QuanAct(nn.Module):
    def __init__(self,
                 activation_wordlength,
                 quantization_method):

        super(QuanAct, self).__init__()
        self.activation_wordlength = activation_wordlength
        self.quantization_method = quantization_method
        self.gather_data = True
        self.register_buffer('x_min', torch.zeros(1))
        self.register_buffer('x_max', torch.zeros(1))
        self.register_buffer('scaling_factor', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))

    def get_scale_shift(self):
        self.scaling_factor, self.zero_point = asymmetric_linear_no_clipping(self.activation_wordlength, self.x_min, self.x_max)

    def forward(self, x):

        if self.gather_data:
            if self.quantization_method == QuanMode.CHANNEL_BFP:
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
            if self.quantization_method == QuanMode.CHANNEL_BFP:
                x = x.transpose(0, 1)
            
            x_quan = linear_quantize(x, self.scaling_factor, self.zero_point)
            x_quan = saturate(x_quan, self.activation_wordlength)
            x_quan = linear_dequantize(x_quan, self.scaling_factor, self.zero_point)
            
            if self.quantization_method == QuanMode.CHANNEL_BFP:
                x_quan = x_quan.transpose(0, 1)

            return x_quan


def activation_quantization(model, wordlength, quantization_method, calibrate_loader):
    # add activation quantisation module
    replace_dict ={}
    for name, module in model.named_modules(): 
        if type(module) in QUAN_TARGET_MODULES:
            module_quan = nn.Sequential(*[QuanAct(wordlength, quantization_method), copy.deepcopy(module), QuanAct(wordlength, quantization_method)]) 
            replace_dict[module] = module_quan

    replace_modules(model, replace_dict)

    model.eval() 
    if torch.cuda.is_available():
        model = model.cuda()

    # gather activation data
    with torch.no_grad():
        for i, (images, target) in enumerate(calibrate_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            model(images)

    for name, module in model.named_modules():
        if isinstance(module, QuanAct):
            module.gather_data = False

    # find scaling factor and zero point
    bFirst = True
    for module in model.modules():
        if isinstance(module, QuanAct):
            if bFirst:
                bFirst = False
                network_min = module.x_min.min()
                network_max = module.x_max.max()
            else:
                network_min = torch.minimum(network_min, module.x_min.min())
                network_max = torch.maximum(network_max, module.x_max.max())

    print("activation min:", network_min)
    print("activation max:", network_max)

    for module in model.modules():
        if isinstance(module, QuanAct):
            if quantization_method == QuanMode.NETWORK_FP:
                module.x_min = network_min
                module.x_max = network_max
            module.get_scale_shift()

    return model

def model_quantisation(model, calibrate_loader, quantization_method=QuanMode.NETWORK_FP, weight_width=16, data_width=16):
    weight_quantizer = WeightQuantizer(model)

    for name, module in model.named_modules(): 
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            quantized_weight = weight_quantizer.AsymmetricQuantHandler(module.weight, weight_width, quantization_method)
            module.weight.data.copy_(quantized_weight)

    activation_quantization(model, data_width, quantization_method, calibrate_loader)