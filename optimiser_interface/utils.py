import json
import os 
import sys

from fpgaconvnet.optimiser.cli import main
from fpgaconvnet.optimiser.solvers import Solver
from fpgaconvnet.parser.Parser import Parser
from fpgaconvnet.platform.Platform import Platform

def process_opt_report(output_dir):
    with open(os.path.join(output_dir, 'report.json'), 'r') as f:
        report = json.load(f)
    throughput = report["network"]["performance"]["throughput (FPS)"]
    latency = report["network"]["performance"]["latency (s)"]
    resources = report["network"]["avg_resource_usage"]
    return throughput, latency, resources

def opt_cli_launcher(model_name, onnx_path, output_dir,
                    batch_size=256, device="u250", 
                    opt_obj='throughput', opt_solver='greedy_partition', opt_cfg="single_partition_throughput", override={}):

    platform_path = os.path.join(os.environ['FPGACONVNET_OPTIMISER'], f'examples/platforms/{device}.toml')
    opt_cfg_path = os.path.join(os.environ['FPGACONVNET_OPTIMISER'], f'examples/optimisers/{opt_cfg}.toml')
    saved_argv = sys.argv
    sys.argv  = ['cli.py']
    sys.argv += ['--name', model_name]
    sys.argv += ['--model_path', onnx_path]
    sys.argv += ['--platform_path', platform_path]
    sys.argv += ['--output_path', output_dir]
    sys.argv += ['-b', str(batch_size)]
    sys.argv += ['--objective', opt_obj]
    sys.argv += ['--optimiser', opt_solver]
    sys.argv += ['--optimiser_config_path', opt_cfg_path]
    sys.argv += ['--custom_onnx']
    for k, v in override.items():
        sys.argv += [f'--{k}', str(v)]
    main()
    sys.argv = saved_argv

    # return process_opt_report(output_dir)

def load_hardware_checkpoint(onnx_path, output_dir, device, checkpoint):
    config_parser = Parser(backend="chisel", quant_mode="auto", custom_onnx = True)
    net = config_parser.onnx_to_fpgaconvnet(onnx_path) # parse the onnx model
    net = config_parser.prototxt_to_fpgaconvnet(net, checkpoint)
    platform_path = os.path.join(os.environ['FPGACONVNET_OPTIMISER'], f'examples/platforms/{device}.toml')
    platform = Platform()
    platform.update(platform_path)
    solver = Solver(net, platform)
    solver.update_partitions()
    solver.create_report(os.path.join(output_dir,"report.json"))
    solver.net.save_all_partitions(os.path.join(output_dir, "config.json"))

    return net, process_opt_report(output_dir)