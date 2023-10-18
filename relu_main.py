#Imports
import argparse
import datetime
import json
import toml
import numpy
import wandb
from torch import nn
import os
import sys
import pathlib

from onnx_sparsity_attribute import onnx_layer_name_cast
from utils import *

THRESHOLD_INC = 0.005

def imagenet_main_api(model_name, gpu_id, output_dir, threshold_path):
    from imagenet_main import imagenet_main
    
    saved_argv = sys.argv
    sys.argv  = ['imagenet_main.py']
    sys.argv += ['--arch', model_name]
    if gpu_id is not None:
        sys.argv += ['--gpu', gpu_id]
    sys.argv += ['--relu_threshold', threshold_path]
    sys.argv += ['--output_path', output_dir]
    acc_top1, acc_top5, sparsity_avg = imagenet_main()
    sys.argv = saved_argv
    return acc_top1, acc_top5, sparsity_avg

def onnx_export_api(model_name, data_path, onnx_dir, threshold_path):
    from onnx_sparsity_attribute import export_main
    
    saved_argv = sys.argv
    sys.argv  = ['onnx_sparsity_attribute.py']
    sys.argv += ['--arch', model_name]
    sys.argv += ['--data_path', data_path]
    sys.argv += ['--export_path', onnx_dir]
    sys.argv += ['--relu_thresholds_path', threshold_path]
    export_main()
    sys.argv = saved_argv

def fpgaconvnet_optimiser_api(model_name, onnx_path, output_dir, device, optimiser_config):
    from fpgaconvnet.optimiser.cli import main

    platform_path = os.path.join(os.environ['FPGACONVNET_OPTIMISER'], f'examples/platforms/{device}.toml')
    optimiser_config_path = os.path.join(os.environ['FPGACONVNET_OPTIMISER'], f'examples/{optimiser_config}.toml')
    saved_argv = sys.argv
    sys.argv  = ['cli.py']
    sys.argv += ['--name', model_name]
    sys.argv += ['--model_path', onnx_path]
    sys.argv += ['--platform_path', platform_path]
    sys.argv += ['--output_path', output_dir]
    sys.argv += ['-b', '256']
    sys.argv += ['--objective', 'throughput']
    sys.argv += ['--optimiser', 'greedy_partition']
    sys.argv += ['--optimiser_config_path', optimiser_config_path]
    main()
    sys.argv = saved_argv

    with open(os.path.join(output_dir, 'report.json'), 'r') as f:
        report = json.load(f)
        throughput = report["network"]["performance"]["throughput (FPS)"]
        latency = report["network"]["performance"]["latency (s)"]
        resources = report["network"]["max_resource_usage"]
    return throughput, latency, resources

def fpgaconvnet_fixed_hardware_api(onnx_file, device, fixed_hardware_checkpoint, output_dir):
    from fpgaconvnet.optimiser.solvers import Solver
    from fpgaconvnet.parser.Parser import Parser
    from fpgaconvnet.platform.Platform import Platform
    
    config_parser = Parser(backend="chisel", quant_mode="auto", custom_onnx = True)
    net = config_parser.onnx_to_fpgaconvnet(onnx_file) # parse the onnx model
    net = config_parser.prototxt_to_fpgaconvnet(net, fixed_hardware_checkpoint)
    platform_path = os.path.join(os.environ['FPGACONVNET_OPTIMISER'], f'examples/platforms/{device}.toml')
    platform = Platform()
    platform.update(platform_path)
    solver = Solver(net, platform)
    solver.update_partitions()
    solver.create_report(os.path.join(output_dir,"report.json"))
    solver.net.save_all_partitions(os.path.join(output_dir, "config.json"))

    with open(os.path.join(output_dir, 'report.json'), 'r') as f:
        report = json.load(f)
        throughput = report["network"]["performance"]["throughput (FPS)"]
        latency = report["network"]["performance"]["latency (s)"]
        resources = report["network"]["max_resource_usage"]
    return throughput, latency, resources

def fpgaconvnet_get_slowest_node_api(onnx_path, config_path, device):
    from fpgaconvnet.optimiser.solvers import Solver
    from fpgaconvnet.parser.Parser import Parser
    import fpgaconvnet.tools.graphs as graphs
    from fpgaconvnet.tools.layer_enum import LAYER_TYPE
    from fpgaconvnet.platform.Platform import Platform
    
    config_parser = Parser(backend="chisel", quant_mode="auto", custom_onnx = True)
    net = config_parser.onnx_to_fpgaconvnet(onnx_path) # parse the onnx model
    net = config_parser.prototxt_to_fpgaconvnet(net, config_path)
    platform_path = os.path.join(os.environ['FPGACONVNET_OPTIMISER'], f'examples/platforms/{device}.toml')
    platform = Platform()
    platform.update(platform_path)
    solver = Solver(net, platform)
    solver.update_partitions()
    slowest_layers = [] # list, as there are multiple partitions
    for partition in net.partitions:
        partition.remove_squeeze()
        layers = []
        for layer in graphs.ordered_node_list(partition.graph):
            if partition.graph.nodes[layer]['type'] == LAYER_TYPE.Convolution:
               layers.append(layer)
        node_latencys = np.array([ partition.graph.nodes[layer]['hw'].latency() for layer in layers ])
        index = list(reversed(np.argsort(node_latencys, kind='mergesort')))[0]
        conv_layer = layers[index]
        for prev_layer in graphs.get_prev_nodes(partition.graph, conv_layer):
            if partition.graph.nodes[prev_layer]['type'] == LAYER_TYPE.ThresholdedReLU:
                slowest_layers.append(prev_layer)
            elif partition.graph.nodes[prev_layer]['type'] in [LAYER_TYPE.Split, LAYER_TYPE.Concat, LAYER_TYPE.EltWise]:
                for prev_prev_layer in graphs.get_prev_nodes(partition.graph, prev_layer):
                    if partition.graph.nodes[prev_prev_layer]['type'] == LAYER_TYPE.ThresholdedReLU:
                        slowest_layers.append(prev_prev_layer)
    relu_index_dict = {}
    for n in graphs.ordered_node_list(net.graph):
        if net.graph.nodes[n]['type'] == LAYER_TYPE.ThresholdedReLU:
            relu_index_dict[n] = len(relu_index_dict)
    slowest_layers_indices = [ relu_index_dict[layer] for layer in slowest_layers ]
    return slowest_layers_indices

#Main
if __name__ == "__main__":

    #Command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        help='model architecture: ' +
                            ' | '.join(model_names))
    parser.add_argument('--relu-policy', choices=['slowest_node', 'uniform'], default="slowest_node", type=str)
    parser.add_argument('--fixed-hardware-checkpoint', default=None, type=str,
                        help='path of config.json file generated by optimiser')
    parser.add_argument('--runs', default=100, type=int,
                        help='how many runs')
    parser.add_argument("--platform", default="u250", type=str)
    parser.add_argument("--optimiser_config", default="greedy_partition_throughput_resnet", type=str)
    parser.add_argument('--gpu', default=None, type=str,
                        help='GPU id to use.')
    parser.add_argument('--enable-wandb', action="store_true", help='enable wandb')
    parser.add_argument('--output_path', default=None, type=str)

    args = parser.parse_args()
    args.platform_path = os.path.join(os.environ["FPGACONVNET_OPTIMISER"], f"examples/platforms/{args.platform}.toml")
    if args.output_path == None:
        fs = "fixed_" if args.fixed_hardware_checkpoint else "" 
        args.output_path = os.getcwd() + f"/{args.arch}_{args.relu_policy}_{fs}relu_output"
    pathlib.Path(args.output_path).mkdir(parents=True, exist_ok=True)

    #Initialise wandb
    if args.enable_wandb:
        wandb.login()
        start_time = datetime.datetime.now()
        timestamp_str = str(start_time).replace(" ", "_").replace(".", "_").replace(":", "_").replace("-", "_")
        hardware_suffix = "_fixed_hardware" if args.fixed_hardware_checkpoint else ""
        name = f"{args.relu_policy}{hardware_suffix}_{timestamp_str}"      
        wandb.init(
            # Set the project where this run will be logged
            project= "-".join([args.arch, "relu"]),
            name = name,
            # Track hyperparameters and run metadata
            config={
                "platform": args.platform
            })

    #Initialise relu_thresholds
    print("=> using pre-trained model '{}'".format(args.arch))
    model = load_model(args.arch)
    relu_thresholds = {}
    threshold = 0.0
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU) or isinstance(module, nn.ReLU6):
            relu_thresholds[name + ".1"] = threshold # .1 to match the naming quantized model

    #For run in runs
    for run in range(args.runs):
        #Create log_dir
        log_dir = args.output_path + "/run_" + str(run)
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

        #Store relu_thresholds
        threshold_path = log_dir + "/relu_thresholds.json"
        with open(threshold_path, 'w') as fp:
            json.dump(relu_thresholds, fp, indent=4)

        #Run imagenet_main.py
        data_dir = log_dir + "/data"
        pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
        top1, top5, sparsity_avg = imagenet_main_api(args.arch, args.gpu, data_dir, threshold_path)

        #Annotate sparsity
        onnx_dir = log_dir + "/onnx_models"
        pathlib.Path(onnx_dir).mkdir(parents=True, exist_ok=True)
        onnx_export_api(args.arch, data_dir, onnx_dir, threshold_path)
        sparse_onnx_path = os.path.join(onnx_dir, args.arch + "_sparse.onnx")

        # Load from fixed hardware or Run optimiser
        opt_dir = log_dir + "/optimiser"
        pathlib.Path(opt_dir).mkdir(parents=True, exist_ok=True)
        if args.fixed_hardware_checkpoint:
            throughput, latency, resources = fpgaconvnet_fixed_hardware_api(sparse_onnx_path, args.platform, args.fixed_hardware_checkpoint, opt_dir)
        else:
            throughput, latency, resources = fpgaconvnet_optimiser_api(args.arch, sparse_onnx_path, opt_dir, args.platform, args.optimiser_config)

        #Log into wandb
        log_info = relu_thresholds | resources | {"top1_accuracy": top1, "top5_accuracy": top5, "throughput": throughput, "latency": latency, "network_sparsity": sparsity_avg}
        print("Logging:", log_info)
        if (args.enable_wandb):
            wandb.log(log_info)

        #Update based on relu-policy
        if args.relu_policy == "uniform":
            threshold = round(threshold + THRESHOLD_INC, 4)
            for name, module in model.named_modules():
                if isinstance(module, nn.ReLU) or isinstance(module, nn.ReLU6):
                    relu_thresholds[name + ".1"] = threshold # .1 to match the naming quantized model
        elif args.relu_policy == "slowest_node":    
            slowest_layers_indices = fpgaconvnet_get_slowest_node_api(sparse_onnx_path, os.path.join(opt_dir, "config.json"), args.platform)
            for i, (k, v) in enumerate(relu_thresholds.items()):
                if i in slowest_layers_indices:
                    relu_thresholds[k] = round(threshold + THRESHOLD_INC, 4)