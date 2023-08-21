#Imports
import argparse
import datetime
import json
import toml
import numpy
import wandb
from torch import nn
import os

import fpgaconvnet.tools.graphs as graphs
from fpgaconvnet.tools.layer_enum import LAYER_TYPE
from fpgaconvnet.parser.Parser import Parser

from utils import *

#Get new throughput function
def get_new_throughput(model_name, net, sparsity_path):

        for partition_index in range(len(net.partitions)):
            # print("Patition:", partition_index)
            partition = net.partitions[partition_index]
            for layer in graphs.ordered_node_list(partition.graph):

                #Check if layer is a Convolution layer tha can benefit from sparsit
                if (partition.graph.nodes[layer]['type'] == LAYER_TYPE.Convolution):

                    if len(partition.graph.nodes[layer]['hw'].sparsity):
                        layer_name = layer_name_translation(model_name, layer)
                        np_path = os.path.join(sparsity_path, model_name + "_" + layer_name + "_histograms.npy")
                        histograms_data = np.load(np_path)
                        histograms = histograms_data/histograms_data.sum(axis = 1)[:, np.newaxis]
                        partition.graph.nodes[layer]['hw'].sparsity = histograms

        net.update_partitions()

        return net.get_throughput(), net.get_latency()

#Layer name translation function
def layer_name_translation(model_name, onnx_name):
    onnx_name = onnx_name.split("_")
    if model_name in ["resnet18", "resnet50"]:
        if len(onnx_name) == 3: # first conv
            torch_name = onnx_name[1]+ ".1"
        else:
            assert len(onnx_name) in [5,6] 
            torch_name = onnx_name[2] + "." +onnx_name[-2]+ ".1"
    elif model_name == "mobilenet_v2":
        if len(onnx_name) == 5: # first and last conv
            torch_name = onnx_name[-2] + ".1"
        else:
            assert len(onnx_name) in [6,7]
            torch_name = onnx_name[2] + "." + onnx_name[-2] + ".1"
    elif model_name in ["alexnet", "vgg11", "vgg16"]:
        torch_name = onnx_name[-2] + ".1"
    elif model_name == "repvgg-a0":
        torch_name = ".".join(onnx_name[1:-1]) + ".1"
    return torch_name

THRESHOLD_INC = 0.005
#Main
if __name__ == "__main__":

    #Command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        help='model architecture: ' +
                            ' | '.join(model_names))
    parser.add_argument('--relu-policy', choices=['slowest_node', 'uniform'], default="uniform", type=str,
                        help='')
    parser.add_argument('--fixed-hardware', action="store_true",
                        help='')
    parser.add_argument('--normalise-hardware', action="store_true",
                        help='')
    parser.add_argument('--use-old-sparsity', action="store_true",
                        help='')
    parser.add_argument('--runs', default=100, type=int,
                        help='how many runs')


    parser.add_argument("--sparsity_path",  default="runlog/resnet18/", type=str,
                        help='Path to sparsity log dir for old sparsity')

    parser.add_argument("--accuracy_path",  default="runlog/resnet18/uniform_accuracy.csv", type=str,
                        help='Path to accuracy .csv file for old accuracy')

    parser.add_argument("--model_path",  default="onnx_models/resnet18/resnet18_uniform_relu_0.0.onnx", type=str,
                        help='Path to sparse .onnx model')

    parser.add_argument("--platform_path", default="../fpgaconvnet-optimiser/examples/platforms/u250.toml" , type=str,
                        help='Path to platform specs (.toml)')

    parser.add_argument("--optimised_config_path",  default="../fpgaconvnet-optimiser/fpgaconvnet/optimiser/resnet18/resnet18_uniform_relu_0.0/config.json", type=str,
                        help='Path to optimised configuration (.json)')

    parser.add_argument('--gpu', default=None, type=str,
                        help='GPU id to use.')

    parser.add_argument('--enable-wandb', action="store_true", help='enable wandb')

    args = parser.parse_args()

    #Initialise wandb
    if args.enable_wandb:
        wandb.login()
        start_time = datetime.datetime.now()
        name = args.relu_policy + "_" + str(start_time).replace(" ","_").replace(".","_").replace(":","_").replace("-", "_")
        if (args.fixed_hardware):
            name = args.relu_policy + "_fixed_hardware_" + str(start_time).replace(" ","_").replace(".","_").replace(":","_").replace("-", "_")

        wandb.init(
            # Set the project where this run will be logged
            project= "-".join([args.arch, "relu"]),
            name = name,
            # Track hyperparameters and run metadata
            config={
                "platform": "u250"
            })

    #Initialise relu_thresholds
    print("=> using pre-trained model '{}'".format(args.arch))
    model = load_model(args.arch)
    relu_thresholds = {}
    for name, module in model.named_modules():
            if isinstance(module, nn.ReLU) or isinstance(module, nn.ReLU6):
                relu_thresholds[name + ".1"] = 0.0


    #For run in runs
    threshold = 0.0
    acc_file = "runlog/" + args.arch + "/" + args.relu_policy + "_accuracy.csv"
    if args.relu_policy == "slowest_node" and not args.fixed_hardware:
        acc_file = "runlog/" + args.arch + "/" + args.relu_policy + "_changing_accuracy.csv"
    for run in range(args.runs):

        #If old sparsity, note metrics
        if args.use_old_sparsity:
            assert args.relu_policy == "uniform"
            # Note accuracy
            with open(args.accuracy_path, 'r') as f:
                lines = f.read().splitlines()
                line = lines[run + 1]
                line_vals = line.split(",")
                top1 = float(line_vals[-3])
                top5 = float(line_vals[-2])
                sparsity = float(line_vals[-1])

            sparsity_dir = args.sparsity_path + "/uniform_relu_" + str(threshold)

        #Else collect sparsity
        else:
            if args.relu_policy == "uniform":
                log_dir = args.arch + "/uniform_relu_" + str(threshold)
                threshold_path = "relu_thresholds/" + args.arch + "/" + args.arch + "_uniform_relu_" + str(threshold) + ".json"
            elif args.relu_policy == "slowest_node":
                if (args.fixed_hardware):
                    log_dir = args.arch + "/slowest_node_" + str(run)
                    threshold_path = "relu_thresholds/" + args.arch + "/" + args.arch + "_slowest_node_" + str(run) + ".json"
                else:
                    log_dir = args.arch + "/slowest_node_changing_" + str(run)
                    threshold_path = "relu_thresholds/" + args.arch + "/" + args.arch + "_slowest_node_changing_" + str(run) + ".json"

            #Create log_dir
            if not os.path.isdir("runlog/" + log_dir):
                os.makedirs("runlog/" + log_dir)
            log_file="runlog/" + log_dir + "/log.txt"

            #Store relu_thresholds
            with open(threshold_path, 'w') as fp:
                json.dump(relu_thresholds, fp)

            os.system("python imagenet_main.py -a " + args.arch + " --gpu " + args.gpu + " --output_path runlog/" + log_dir + " --accuracy_output " + acc_file + " --relu_threshold " + threshold_path)

            sparsity_dir = "runlog/" + log_dir

            # Note accuracy
            with open(acc_file, 'r') as f:
                lines = f.read().splitlines()
                last_line = lines[-1]
                top1 = float(last_line.split(",")[-3])
                top5 = float(last_line.split(",")[-2])
                sparsity = float(last_line.split(",")[-1])



        #If fixed hardware, parse network and get throughput and latency from fixed hardwareusing collected sparsity
        if (args.fixed_hardware):
            config_parser = Parser(backend="chisel", quant_mode="auto") # use the HLS backend with 16-bit fixed-point quantisation
            net = config_parser.onnx_to_fpgaconvnet(args.model_path, args.platform_path) # parse the onnx model

            net = config_parser.prototxt_to_fpgaconvnet(net, args.optimised_config_path)

            net.update_partitions()

            throughput, latency = get_new_throughput(args.arch, net, sparsity_dir)

            log_info =  relu_thresholds | {"top1_accuracy": top1, "top5_accuracy": top5, "throughput": throughput, "latency": latency, "network_sparsity": sparsity}
            print("Logging:", log_info)

        #Else annotate sparsity, run optimiser, note resources, throughput, and latency
        else:
            #Annotate sparsity
            if args.relu_policy == "uniform":
                dense_onnx_path = "onnx_models/" + args.arch + "/" + args.arch + ".onnx"
                sparse_onnx_path = "onnx_models/" + args.arch + "/" + args.arch +  "_uniform_relu_" + str(threshold) + ".onnx"
            elif args.relu_policy == "slowest_node":
                if (args.fixed_hardware):
                    dense_onnx_path = "onnx_models/" + args.arch + "/" + args.arch + ".onnx"
                    sparse_onnx_path = "onnx_models/" + args.arch + "/" + args.arch +  "_slowest_node_" + str(run) + ".onnx"
                else:
                    dense_onnx_path = "onnx_models/" + args.arch + "/" + args.arch + ".onnx"
                    sparse_onnx_path = "onnx_models/" + args.arch + "/" + args.arch +  "_slowest_node_changing_" + str(run) + ".onnx"

            os.system("python onnx_sparsity_attribute_full.py -a " + args.arch + " --data " + sparsity_dir + " --dense_onnx_path " + dense_onnx_path + " --sparse_onnx_path " + sparse_onnx_path)


            # Run optimiser
            if args.relu_policy == "uniform":
                output_path = "../fpgaconvnet-optimiser/fpgaconvnet/optimiser/" + args.arch + "/" + args.arch + "_uniform_relu_" + str(threshold)
            elif args.relu_policy == "slowest_node":
                output_path = "../fpgaconvnet-optimiser/fpgaconvnet/optimiser/" + args.arch + "/" + args.arch + "_slowest_node_" + str(run)

            os.system("python -u ../fpgaconvnet-optimiser/fpgaconvnet/optimiser/cli.py --rerun-optim -n "+ args.arch + " -m " + sparse_onnx_path + " -o " + output_path + " -p  " + args.platform_path + " -b 256 --objective throughput --optimiser greedy_partition --optimiser_config_path ../fpgaconvnet-optimiser/examples/greedy_partition_throughput_residual.toml")

            # Note throughput
            f = open(output_path + "/report.json")
            report = json.load(f)
            throughput = report["network"]["performance"]["throughput"]
            latency = report["network"]["performance"]["latency"]
            resources = report["network"]["max_resource_usage"]
            f.close()


            # # Create resource toml file
            # f = open(args.platform_path, 'r')
            # new_toml = toml.load(f)
            # for key, value in resources.items():
            #     if key == "DSP":
            #         new_toml["resources"][key] = round(value/0.9)
            # f.close()

            # # Write resource toml file
            # if not os.path.isdir("../fpgaconvnet-optimiser/examples/platforms/" + args.arch + "_cifar10_uniform_relu_norm/"):
            #     os.mkdir("../fpgaconvnet-optimiser/examples/platforms/" + args.arch + "_cifar10_uniform_relu_norm/")
            # platform_path = "../fpgaconvnet-optimiser/examples/platforms/" + args.arch + "_cifar10_uniform_relu_norm/u250_" + str(relu_threshold) + ".toml"
            # f = open(platform_path, 'w')
            # toml.dump(new_toml, f)
            # f.close()


            #If normalise, run dense and sparse normalised
            if (args.normalise_hardware):
                pass

            else:
                log_info = relu_thresholds | resources | {"top1_accuracy": top1, "top5_accuracy": top5, "throughput": throughput, "latency": latency, "network_sparsity": sparsity}


        #Log into wandb
        if (args.enable_wandb):
            wandb.log(log_info)


        #Update based on relu-policy
        threshold = round(threshold + THRESHOLD_INC, 4)

        if args.relu_policy == "uniform":
            for name, module in model.named_modules():
                if isinstance(module, nn.ReLU):
                    relu_thresholds[name + ".1"] = round(threshold, 4)
        elif args.relu_policy == "slowest_node":
            if not (args.fixed_hardware):
                config_parser = Parser(backend="chisel", quant_mode="auto") # use the HLS backend with 16-bit fixed-point quantisation
                net = config_parser.onnx_to_fpgaconvnet(sparse_onnx_path, args.platform_path) # parse the onnx model

                net = config_parser.prototxt_to_fpgaconvnet(net, output_path + "/config.json")

                net.update_partitions()
                
            # Update ReLU thresholds for slowest node
            replaced_layers = set()
            previous_relu = None
            #Change slowest node in each partition
            for partition_index in range(len(net.partitions)):
                replace_layer = None
                max_latency = 0
                partition = net.partitions[partition_index]
                for layer in graphs.ordered_node_list(partition.graph):
                    #Keep track of preceding relu layer
                    if isinstance(partition.graph.nodes[layer]['type'], list):
                        if LAYER_TYPE.ReLU in partition.graph.nodes[layer]['type']:
                            previous_relu = layer
                    elif (partition.graph.nodes[layer]['type'] == LAYER_TYPE.ReLU):
                        previous_relu = layer

                    #Check if layer is a Convolution layer tha can benefit from sparsit
                    if (partition.graph.nodes[layer]['type'] == LAYER_TYPE.Convolution):
                        layer_latency = partition.graph.nodes[layer]['hw'].latency()
                        if previous_relu != None:
                            previous_layer = layer_name_translation(args.arch, previous_relu)
                            if layer_latency > max_latency and len(partition.graph.nodes[layer]['hw'].sparsity):
                                max_latency = layer_latency
                                replace_layer = previous_layer

                if replace_layer != None and replace_layer not in replaced_layers:
                    relu_thresholds[replace_layer] += THRESHOLD_INC
                    replaced_layers.add(replace_layer)