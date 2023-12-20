import torch

import numpy as np
import torch.nn as nn

from models.classification.utils import AverageMeter
from quantization.utils import QuantMode, \
    ACTIVA_QUANT_MODULES, WEIGHT_QUANT_MODULES, \
    QuantAct, linear_quantize, saturate

ACTIVA_ENCODE_MODULES = ACTIVA_QUANT_MODULES
WEIGHT_ENCODE_MODULES = WEIGHT_QUANT_MODULES

def get_compression_ratio(x, encoded_x, x_bits, l_bits):
    return (len(encoded_x) * (x_bits + l_bits)) / (len(x.flatten()) * x_bits)

def encode(x, word_length, scaling_factor, zero_point, l_bits, transpose=False):
    if torch.cuda.is_available():
        x = x.cuda()
        scaling_factor = scaling_factor.cuda()
        zero_point = zero_point.cuda()
    # convert to quantized int representation
    if transpose:
        x = x.transpose(0, 1)
    x_quant = linear_quantize(x, scaling_factor, zero_point)
    x_quant = saturate(x_quant, word_length)
    if transpose:
        x_quant = x_quant.transpose(0, 1)

    # todo: fix the flatten dim order to match hw streaming
    x_flatten = x_quant.flatten()

    diff = torch.cat((torch.tensor([1], device=x.device), torch.diff(x_flatten)))
    indices = torch.nonzero(diff != 0).flatten()
    lengths = torch.diff(torch.cat((indices, torch.tensor([len(x_flatten)], device=x.device))))

    # combine consecutive elements and counts
    combined = torch.stack((x_flatten[indices], lengths)).T

    # split elements if the length exceeds 2^l_bits
    mask = combined[:, 1] > 2 ** l_bits
    split_indices = torch.nonzero(mask).flatten()
    for index in reversed(split_indices):  # reverse order to avoid index shifts
        value, length = combined[index]
        repeat = int(length / (2 ** l_bits))
        additional_elements = [[value, 2 ** l_bits]] * repeat
        remain = length - repeat * (2 ** l_bits)
        if remain > 0:
            additional_elements.append([value, remain])
        additional_elements = torch.tensor(additional_elements, device=x.device)
        combined = torch.cat((combined[:index], additional_elements, combined[index + 1:]), dim=0)
        
    return combined

def log_encoding(module, input, output):
    batch_size = input[0].shape[0]
    encoded_data = encode(input[0], module.word_length, 
        module.scaling_factor, module.zero_point, module.l_bits, 
        (module.mode == QuantMode.CHANNEL_BFP))
    ratio = get_compression_ratio(input[0], encoded_data, module.word_length, module.l_bits)
    module.encoding_ratio.update(ratio, batch_size)

def encode_model(model_wrapper, l_bits):
    assert "quantization" in model_wrapper.sideband_info.keys(), "Only quantized models can be encoded"
    
    weight_width = model_wrapper.sideband_info["quantization"]["weight_width"]
    data_width = model_wrapper.sideband_info["quantization"]["data_width"]

    encode_info = {}
    for name, module in model_wrapper.named_modules():
        if isinstance(module, WEIGHT_QUANT_MODULES):
            assert name.endswith(".1")
            name = name[:-2] # remove the quantization suffix
            scale = model_wrapper.sideband_info["quantization"][name]["weight_scale"]
            zero_point = model_wrapper.sideband_info["quantization"][name]["weight_zero_point"]
            encoded_weight = encode(module.weight.data, weight_width, scale, zero_point, l_bits)
            ratio = get_compression_ratio(module.weight.data, encoded_weight, weight_width, l_bits)
            encode_info[name] = {"weight_compression_ratio": ratio}
        elif isinstance(module, QuantAct):
            assert name.endswith(".0") or name.endswith(".2")        
            module.l_bits = l_bits
            module.encoding_ratio = AverageMeter('compression ratio', ':6.3f')
            module.register_forward_hook(log_encoding)

    model_wrapper.inference("calibrate")
    for name, module in model_wrapper.named_modules():
        if isinstance(module, QuantAct):
            if name[:-2] not in encode_info.keys():
                encode_info[name[:-2]] = {}
            if name.endswith(".0"):
                encode_info[name[:-2]]["input_compression_ratio"] = module.encoding_ratio.avg
            elif name.endswith(".2"):
                encode_info[name[:-2]]["output_compression_ratio"] = module.encoding_ratio.avg
            else:
                assert False, "unexpected module name"

    model_wrapper.sideband_info["encoding"] = encode_info
    
    compression_ratio = []
    for v in encode_info.values():
        compression_ratio += list(v.values())
    compression_ratio = np.mean(compression_ratio)
    return compression_ratio