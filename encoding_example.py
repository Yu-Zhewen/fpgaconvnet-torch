import argparse
import os
import pathlib
import random
from statistics import mean

import torch

from encoding.huffman import huffman_model
from encoding.rle import rle_model
from models import initialize_wrapper
from quantization.utils import QuantMode, quantize_model


def main():
    parser = argparse.ArgumentParser(description='Quantization Example')
    parser.add_argument('--dataset_name', default="camvid", type=str,
                        help='dataset name')
    parser.add_argument('--dataset_path', metavar='DIR', default="~/dataset/CamVid",
                        help='path to dataset')
    parser.add_argument('--model_name', metavar='ARCH', default='unet',
                        help='model architecture')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    parser.add_argument('--output_path', default=None, type=str,
                        help='output path')

    args = parser.parse_args()
    if args.output_path == None:
        args.output_path = os.path.join(os.getcwd(),
         f"output/{args.dataset_name}/{args.model_name}")
    pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)
    print(args)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    random.seed(0)
    torch.manual_seed(0)

    model_wrapper = initialize_wrapper(args.dataset_name, args.model_name,
                                       os.path.expanduser(args.dataset_path), args.batch_size, args.workers)

    # quantization
    model_wrapper.load_model()
    quantize_model(model_wrapper,  {
                   'weight_width': 8, 'data_width': 8, 'mode': QuantMode.CHANNEL_BFP})

    # encoding
    # print("Encoding model in RLE...")
    # ratio, ratio_detailed = rle_model(model_wrapper, 8)
    # print("compression ratio: ", ratio)
    # for k, v in ratio_detailed.items():
    #     print(f"{k}: {v}")
    # print("compression ratio (detailed): ", mean(ratio_detailed.values()))
    # model_wrapper.generate_onnx_files(os.path.join(args.output_path, "rle"))

    print("Encoding model in Huffman...")
    ratio, ratio_detailed = huffman_model(model_wrapper)
    print("compression ratio: ", ratio)
    for k, v in ratio_detailed.items():
        print(f"{k}: {v}")
    print("compression ratio (detailed): ", mean(ratio_detailed.values()))
    model_wrapper.generate_onnx_files(os.path.join(args.output_path, "huffman"))

if __name__ == '__main__':
    main()
