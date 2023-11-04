import argparse
import copy
import os
import pathlib
import random
import torch

from models.segmentation.lggmri import BrainModelWrapper
from quantization.utils import QuantMode, quantize_model

def main():
    parser = argparse.ArgumentParser(description='PyTorch LLR MRI Brain Tumor Segmentation')
    parser.add_argument('--data', metavar='DIR', default="~/dataset/lgg-mri-segmentation/kaggle_3m",
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='unet',
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
        args.output_path = os.getcwd() + "/output"
    pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)
    print(args)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    random.seed(0)
    torch.manual_seed(0)

    os.environ['LGGMRI_PATH'] = os.path.expanduser(args.data)
    model_wrapper = BrainModelWrapper(args.arch)
    model_wrapper.load_data(args.batch_size, args.workers)

    # TEST 1
    print("FLOAT32 Inference")
    model_wrapper.load_model()
    model_wrapper.inference("validate")
 
    # TEST 2
    print("NETWORK FP16 Inference")
    model_wrapper.load_model()
    quantize_model(model_wrapper, {'weight_width': 16, 'data_width': 16, 'mode': QuantMode.NETWORK_FP})
    model_wrapper.inference("validate")
    model_wrapper.generate_onnx_files(os.path.join(args.output_path, "fp16"))

    # TEST 3
    print("NETWORK FP8 Inference")
    model_wrapper.load_model()
    quantize_model(model_wrapper, {'weight_width': 8, 'data_width': 8, 'mode': QuantMode.NETWORK_FP})
    model_wrapper.inference("validate")

    '''
    # TEST 4
    print("LAYER BFP8 Inference")
    model_wrapper.load_model()
    quantize_model(model_wrapper, {'weight_width': 8, 'data_width': 8, 'mode': QuantMode.LAYER_BFP})
    model_wrapper.inference("validate")

    # TEST 5
    print("CHANNEL BFP8 Inference") 
    # note: CHANNEL_BFP can be worse than LAYER_BFP, if calibration size is small!
    model_wrapper.load_model()
    quantize_model(model_wrapper,  {'weight_width': 8, 'data_width': 8, 'mode': QuantMode.CHANNEL_BFP})
    model_wrapper.inference("validate") 
    model_wrapper.generate_onnx_files(os.path.join(args.output_path, "channel_bfp8"))
    '''
    # todo: fix bfp for convtranspose2d, weight shape (in_channels, out_channels, kernel_size, kernel_size)

if __name__ == '__main__':
    main()

