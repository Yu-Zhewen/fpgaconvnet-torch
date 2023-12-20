import argparse
import os
import pathlib
import random

import torch

from models import initialize_wrapper
from quantization.utils import QuantMode, quantize_model


def main():
    parser = argparse.ArgumentParser(description='Quantization Example')
    parser.add_argument('--dataset_name', default="imagenet", type=str,
                        help='dataset name')
    parser.add_argument('--dataset_path', metavar='DIR', default="~/dataset/ILSVRC2012_img",
                        help='path to dataset')
    parser.add_argument('--model_name', metavar='ARCH', default='resnet18',
                        help='model architecture')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
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

    # TEST 1
    print("FLOAT32 Inference")
    model_wrapper.inference("test")
    model_wrapper.generate_onnx_files(
        os.path.join(args.output_path, "float32"))

    # TEST 2
    print("NETWORK FP16 Inference")
    # reload the model everytime a new quantization mode is tested
    model_wrapper.load_model()
    quantize_model(model_wrapper, {
                   'weight_width': 16, 'data_width': 16, 'mode': QuantMode.NETWORK_FP})
    model_wrapper.inference("test")
    model_wrapper.generate_onnx_files(
        os.path.join(args.output_path, "network_fp16"))

    # TEST 3
    print("NETWORK FP8 Inference")
    model_wrapper.load_model()
    quantize_model(model_wrapper, {
                   'weight_width': 8, 'data_width': 8, 'mode': QuantMode.NETWORK_FP})
    model_wrapper.inference("test")
    model_wrapper.generate_onnx_files(
        os.path.join(args.output_path, "network_fp8"))

    # TEST 4
    print("LAYER BFP8 Inference")
    model_wrapper.load_model()
    quantize_model(model_wrapper, {
                   'weight_width': 8, 'data_width': 8, 'mode': QuantMode.LAYER_BFP})
    model_wrapper.inference("test")
    model_wrapper.generate_onnx_files(
        os.path.join(args.output_path, "layer_bfp8"))

    # TEST 5
    print("CHANNEL BFP8 Inference")
    # note: CHANNEL_BFP can be worse than LAYER_BFP, if calibration size is small!
    model_wrapper.load_model()
    quantize_model(model_wrapper,  {
                   'weight_width': 8, 'data_width': 8, 'mode': QuantMode.CHANNEL_BFP})
    model_wrapper.inference("test")
    model_wrapper.generate_onnx_files(
        os.path.join(args.output_path, "channel_bfp8"))


if __name__ == '__main__':
    main()
