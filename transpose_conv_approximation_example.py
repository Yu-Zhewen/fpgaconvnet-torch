import argparse
import copy
import os
import pathlib
import random
import torch

from models import initialize_wrapper
from quantization.utils import QuantMode, quantize_model
from conv_transp_approx.utils import apply_conv_transp_approx

def main():
    parser = argparse.ArgumentParser(description='Transpose Convolution Approximation Example')
    parser.add_argument('--dataset_name', default="camvid", type=str,
                        help='dataset name')
    parser.add_argument('--dataset_path', metavar='DIR', default="~/dataset/ILSVRC2012_img",
                        help='path to dataset')
    parser.add_argument('--model_name', metavar='ARCH', default='unet',
                        help='model architecture')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    parser.add_argument('--output_path', default=None, type=str,
                        help='output path')


    args = parser.parse_args()
    if args.output_path == None:
        args.output_path = os.getcwd() + "/output"
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
    model_wrapper.generate_onnx_files(os.path.join(args.output_path, "fp32"))


    # TEST 12
    apply_conv_transp_approx(model_wrapper=model_wrapper, upsampling_mode="bilinear", kernel_approx_strategy="average")
    print("FLOAT32 Inference Conv Transpose Approximation")
    model_wrapper.inference("test")
    # FIXME: if we use generate_onnx_files here the onnx saved model does not contain the changes made by apply_conv_transp_approx
    model_wrapper.onnx_exporter(os.path.join(args.output_path, "fp32_approx", f"{args.model_name}_f32.onnx"))

if __name__ == '__main__':
    main()

