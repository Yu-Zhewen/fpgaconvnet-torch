import torch
from dahuffman import HuffmanCodec

from encoding.utils import (avg_compress_ratio, avg_compress_ratio_detailed,
                            convert_to_int)
from quantization.utils import WEIGHT_QUANT_MODULES, QuantAct, QuantMode


def get_huffman_encoding_ratio(count, x_width):
    keys = [int(i) for i in range(-2**(x_width - 1), 2**(x_width - 1))]
    hist = { keys[i]: int(count[i]) for i in range(len(keys)) }

    codec = HuffmanCodec.from_frequencies(hist)
    table = codec.get_code_table()

    bits = [table[i][0] for i in keys]
    avg_bits = count @ torch.tensor(bits, dtype=torch.float32, device=count.device) / torch.sum(count)
    ratio = avg_bits.item() / x_width
    return ratio

def log_hist_count(module, input, output):
    quant_data = convert_to_int(input[0], module.word_length,
        module.scaling_factor, module.zero_point, (module.mode == QuantMode.CHANNEL_BFP))
    count = torch.histc(quant_data, bins=2**module.word_length, min=-2**(module.word_length - 1), max=2**(module.word_length - 1)-1)
    module.hist_count += count

def huffman_model(model_wrapper):
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
            count = torch.histc(quant_weight, bins=2**weight_width, min=-2**(weight_width - 1), max=2**(weight_width - 1)-1)
            ratio = get_huffman_encoding_ratio(count, weight_width)
            encode_info[name] = {"weight_compression_ratio": ratio}
        elif isinstance(module, QuantAct):
            assert name.endswith(".0") or name.endswith(".2")
            module.hist_count = torch.zeros(2**data_width, dtype=torch.float32)
            if torch.cuda.is_available():
                module.hist_count = module.hist_count.cuda()
            module.register_forward_hook(log_hist_count)

    model_wrapper.inference("calibrate")
    for name, module in model_wrapper.named_modules():
        if isinstance(module, QuantAct):
            if name[:-2] not in encode_info.keys():
                encode_info[name[:-2]] = {}
            if name.endswith(".0"):
                encode_info[name[:-2]]["input_compression_ratio"] = get_huffman_encoding_ratio(module.hist_count, data_width)
            elif name.endswith(".2"):
                encode_info[name[:-2]]["output_compression_ratio"] = get_huffman_encoding_ratio(module.hist_count, data_width)
            else:
                assert False, "unexpected module name"

    model_wrapper.sideband_info["encoding"] = encode_info
    return avg_compress_ratio(encode_info), avg_compress_ratio_detailed(encode_info)