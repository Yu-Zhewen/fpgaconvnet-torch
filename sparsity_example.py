import argparse
import copy
import os
import pathlib
import random
import torch

from models import initialize_wrapper
from quantization.utils import QuantMode, quantize_model
from sparsity.prune_utils import apply_weight_pruning
from sparsity.utils import measure_model_sparsity

def main():
    parser = argparse.ArgumentParser(description='Sparsity Example')
    parser.add_argument('--dataset_name', default="imagenet", type=str,
                        help='dataset name') 
    parser.add_argument('--dataset_path', metavar='DIR', default="~/dataset/ILSVRC2012_img",
                        help='path to dataset')
    parser.add_argument('--model_name', metavar='ARCH', default='resnet18',
                        help='model architecture')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers')
    parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    parser.add_argument('--weight_threshold', default=0.005, type=float,
                        help='threshold for weight pruning')

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

    print("NETWORK FP16 Inference")
    quantize_model(model_wrapper, {'weight_width': 16, 'data_width': 16, 'mode': QuantMode.NETWORK_FP})
    top1, top5 = model_wrapper.inference("test")

    if args.weight_threshold is None:
        # post-activation sparsity has zero impact on accuracy
        print("POST-ACTIVATION SPARSITY")
    else:
        # apply weight pruning
        print("WEIGHT PRUNING")
        apply_weight_pruning(model_wrapper, args.weight_threshold)
        top1, top5 = model_wrapper.inference("test")

    # measure sparsity-related stats on calibration set
    avg_sparsity = measure_model_sparsity(model_wrapper)
    model_wrapper.generate_onnx_files(os.path.join(args.output_path, "sparse"))
    print(f"Average sparsity: {avg_sparsity}")

if __name__ == '__main__':
    main()

