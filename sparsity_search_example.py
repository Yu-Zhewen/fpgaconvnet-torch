import argparse
import copy
import os
import pathlib
import random
import torch

from models import initialize_wrapper
from optimiser_interface.utils import opt_cli_launcher, load_hardware_checkpoint
from quantization.utils import QuantMode, quantize_model
from sparsity.prune_utils import apply_weight_pruning
from sparsity.relu_utils import apply_threshold_relu
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

    parser.add_argument('--weight_step_size', default=0.001, type=float,
                        help='threshold for weight pruning')
    parser.add_argument('--activation_step_size', default=0.01, type=float,
                        help='threshold for weight pruning')
    parser.add_argument('--accuracy_tolerance', default=2.0, type=float,
                        help='accuracy tolerance')

    parser.add_argument('--output_path', default=None, type=str,
                        help='output path')                     

    parser.add_argument("--platform", default="u250", type=str)
    parser.add_argument("--optimiser_config", default="single_partition_throughput", type=str)   

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
    base_top1, base_top5 = model_wrapper.inference("test")
    model_copy = copy.deepcopy(model_wrapper.model)
    avg_sparsity = measure_model_sparsity(model_wrapper)
    print(f"Avg Sparsity: {avg_sparsity}")
    opt_dir = os.path.join(args.output_path, f"base")
    model_wrapper.generate_onnx_files(args.output_path)
    onnx_path = os.path.join(args.output_path, f"{args.model_name}.onnx")
    opt_cli_launcher(args.model_name, onnx_path, opt_dir, device=args.platform, opt_cfg=args.optimiser_config)
    _, (base_throughput, _, _) = load_hardware_checkpoint(onnx_path, opt_dir, args.platform, os.path.join(opt_dir, "config.json"))
    
    top1 = base_top1
    throughput = base_throughput 
    weight_threshold = 0.0
    activation_threshold = 0.0
    step = 0
    model_wrapper.model = copy.deepcopy(model_copy)
    while base_top1 - top1 < args.accuracy_tolerance:   
        print(f"**Step: {step}**")
        print("Increment Weight Sparsity....")
        apply_weight_pruning(model_wrapper, weight_threshold+args.weight_step_size)
        apply_threshold_relu(model_wrapper, activation_threshold)
        w_top1, w_top5 = model_wrapper.inference("test")
        avg_sparsity = measure_model_sparsity(model_wrapper)
        print(f"Avg Sparsity: {avg_sparsity}")
        opt_dir = os.path.join(args.output_path, f"sparsity_w{step}")
        model_wrapper.generate_onnx_files(args.output_path)
        onnx_path = os.path.join(args.output_path, f"{args.model_name}.onnx")
        opt_cli_launcher(args.model_name, onnx_path, opt_dir, device=args.platform, opt_cfg=args.optimiser_config)
        _, (w_throughput, _, _) = load_hardware_checkpoint(onnx_path, opt_dir, args.platform, os.path.join(opt_dir, "config.json"))
        model_wrapper.model = copy.deepcopy(model_copy)

        print("Increment Activation Sparsity....")
        apply_weight_pruning(model_wrapper, weight_threshold)
        apply_threshold_relu(model_wrapper, activation_threshold+args.activation_step_size)
        a_top1, a_top5 = model_wrapper.inference("test")
        avg_sparsity = measure_model_sparsity(model_wrapper)
        print(f"Avg Sparsity: {avg_sparsity}")
        opt_dir = os.path.join(args.output_path, f"sparsity_a{step}")
        model_wrapper.generate_onnx_files(args.output_path)
        onnx_path = os.path.join(args.output_path, f"{args.model_name}.onnx")
        opt_cli_launcher(args.model_name, onnx_path, opt_dir, device=args.platform, opt_cfg=args.optimiser_config)
        _, (a_throughput, _, _) = load_hardware_checkpoint(onnx_path, opt_dir, args.platform, os.path.join(opt_dir, "config.json"))
        model_wrapper.model = copy.deepcopy(model_copy)

        if (top1 - w_top1 + 1e-6) / (w_throughput - throughput + 1e-6) < (top1 - a_top1 + 1e-6) / (a_throughput - throughput + 1e-6):
            weight_threshold += args.weight_step_size
            top1 = w_top1
            throughput = w_throughput
            print("Weight Sparsity is better")
        else:
            activation_threshold += args.activation_step_size
            top1 = a_top1
            throughput = a_throughput
            print("Activation Sparsity is better")
        step += 1
        print(f"**Top1: {top1}, throughput: {throughput}, Weight Threshold: {weight_threshold}, Activation Threshold: {activation_threshold}**")


if __name__ == '__main__':
    main()

