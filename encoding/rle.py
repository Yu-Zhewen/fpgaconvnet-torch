import torch
import torch.nn as nn

from encoding.utils import (avg_compress_ratio, avg_compress_ratio_detailed,
                            convert_to_int)
from models.classification.utils import AverageMeter
from quantization.utils import WEIGHT_QUANT_MODULES, QuantAct, QuantMode


def rle_compression_ratio(x, encoded_x, x_bits, l_bits):
    assert len(x.shape) == 1, "x must be a 1D tensor"
    assert len(encoded_x.shape) == 2 and encoded_x.shape[1] == 2, "encoded_x must be a 2D tensor with 2 columns"
    return (encoded_x.shape[0] * (x_bits + l_bits)) / (x.shape[0] * x_bits)

def rle_encode(x_flatten, l_bits):
    diff = torch.cat((torch.tensor([1], device=x_flatten.device), torch.diff(x_flatten)))
    indices = torch.nonzero(diff != 0).flatten()
    lengths = torch.diff(torch.cat((indices, torch.tensor([len(x_flatten)], device=x_flatten.device))))

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
        additional_elements = torch.tensor(additional_elements, device=x_flatten.device)
        combined = torch.cat((combined[:index], additional_elements, combined[index + 1:]), dim=0)

    return combined

def log_encoding(module, input, output):
    batch_size = input[0].shape[0]
    quant_data = convert_to_int(input[0], module.word_length,
        module.scaling_factor, module.zero_point, (module.mode == QuantMode.CHANNEL_BFP))
    encoded_data = rle_encode(quant_data, module.l_bits)
    ratio = rle_compression_ratio(quant_data, encoded_data, module.word_length, module.l_bits)
    module.encoding_ratio.update(ratio, batch_size)

def rle_model(model_wrapper, l_bits):
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
            quant_weight = convert_to_int(module.weight.data, weight_width, scale, zero_point, False)
            encoded_weight = rle_encode(quant_weight, l_bits)
            ratio = rle_compression_ratio(quant_weight, encoded_weight, weight_width, l_bits)
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

    return avg_compress_ratio(encode_info), avg_compress_ratio_detailed(encode_info)