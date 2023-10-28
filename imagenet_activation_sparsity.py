import argparse
import copy
import os
import pathlib
import random
import torch

from models.classification.imagenet import TorchvisionModelWrapper
from quantization.utils import QuantMode, quantize_model
from sparsity.utils import measure_model_sparsity

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet')
    parser.add_argument('--data', metavar='DIR', default="~/dataset/ILSVRC2012_img",
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
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

    os.environ['IMAGENET_PATH'] = os.path.expanduser(args.data)
    model_wrapper = TorchvisionModelWrapper(args.arch)
    model_wrapper.load_data(args.batch_size, args.workers)

    print("NETWORK FP16 Inference")
    model_wrapper.load_model()
    quantize_model(model_wrapper, QuantMode.NETWORK_FP, 16, 16)
    top1, top5 = model_wrapper.inference("validate")

    # post-activation sparsity has zero impact on accuracy
    # measure sparsity-related stats on calibration set
    measure_model_sparsity(model_wrapper)
    model_wrapper.generate_onnx_files(os.path.join(args.output_path, "sparse"))

if __name__ == '__main__':
    main()

