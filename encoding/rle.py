import torch
import torch.nn as nn

from quantization.utils import ACTIVA_QUANT_MODULES, WEIGHT_QUANT_MODULES, \
    linear_quantize, saturate

ACTIVA_ENCODE_MODULES = ACTIVA_QUANT_MODULES
WEIGHT_ENCODE_MODULES = WEIGHT_QUANT_MODULES

def get_compression_ratio(x, encoded_x, x_bits, l_bits):
    return (len(encoded_x) * (x_bits + l_bits)) / (len(x.flatten()) * x_bits)

def encode(x, word_length, scaling_factor, zero_point, l_bits):
    # convert to quantized int representation
    x_quant = linear_quantize(x, scaling_factor, zero_point)
    x_quant = saturate(x_quant, word_length)

    # todo: fix the dim order as hw streaming
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


def encode_model(model_wrapper, l_bits):
    assert "quantization" in model_wrapper.sideband_info.keys(), "Only quantized models can be encoded"
    
    weight_width = model_wrapper.sideband_info["quantization"]["weight_width"]
    data_width = model_wrapper.sideband_info["quantization"]["data_width"]

    for name, module in model_wrapper.named_modules():
        if isinstance(module, WEIGHT_QUANT_MODULES):
            assert name.endswith(".1")
            name = name[:-2] # remove the quantization suffix
            scale = model_wrapper.sideband_info["quantization"][name]["weight_scale"]
            zero_point = model_wrapper.sideband_info["quantization"][name]["weight_zero_point"]
            encoded_weight = encode(module.weight.data, weight_width, scale, zero_point, l_bits)
            print(name, "weight compression ratio:", get_compression_ratio(module.weight.data, encoded_weight, weight_width, l_bits))
