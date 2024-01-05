from collections import defaultdict

import numpy as np
import torch

from quantization.utils import linear_quantize


def convert_to_int(x, word_length, scaling_factor, zero_point, transpose):
    if torch.cuda.is_available():
        x = x.cuda()
        scaling_factor = scaling_factor.cuda()
        zero_point = zero_point.cuda()

    # convert to quantized int representation
    if transpose:
        x = x.transpose(0, 1)
    x_quant = linear_quantize(x, word_length, scaling_factor, zero_point)
    if transpose:
        x_quant = x_quant.transpose(0, 1)

    # todo: fix the flatten dim order to match hw streaming
    x_flatten = x_quant.flatten()
    return x_flatten

def avg_compress_ratio(encode_info):
    compression_ratio = []
    for v in encode_info.values():
        compression_ratio += list(v.values())
    compression_ratio = np.mean(compression_ratio)
    return compression_ratio

def avg_compress_ratio_detailed(encode_info):
    sum_dict = defaultdict(float)
    count_dict = defaultdict(int)

    for v in encode_info.values():
        for key, value in v.items():
            sum_dict[key] += value
            count_dict[key] += 1

    return {key: sum_dict[key] / count_dict[key] for key in sum_dict}