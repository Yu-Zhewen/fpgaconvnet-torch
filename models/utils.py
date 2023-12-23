import copy
import onnx
import os
import pathlib

import torch.nn as nn

def replace_modules(model, replace_dict):
    for name, module in model.named_modules(): 
        for subname, submodule in module.named_children():
            if submodule in replace_dict.keys():
                new_submodule = replace_dict[submodule]
                assert(hasattr(module, subname))
                setattr(module,subname,new_submodule)

def set_nodeattr(node, attr_name, attr_value):
    #print("annotate ", node.name, attr_name, attr_value)
    new_attr = onnx.helper.make_attribute(attr_name, attr_value)
    node.attribute.append(new_attr)

def onnx_to_torch_name_cast(onnx_name, onnx_type, sep="/"):
    onnx_name = onnx_name.split(sep)
    onnx_name = [name for name in onnx_name if name != ""]
    buffer = [onnx_name[0]]
    for i in range(1, len(onnx_name)):
        if onnx_name[i].startswith(buffer[-1]):
            buffer[-1] = onnx_name[i]
        else:
            buffer.append(onnx_name[i])
    if buffer[-1] == onnx_type:
        buffer.pop()
    torch_name = ".".join(buffer)
    return torch_name

def _insert_threshold_relu(onnx_model, sideband_info):
    if "threshold_relu" not in sideband_info:
        return
    info = sideband_info["threshold_relu"]
    for index, node in enumerate(onnx_model.graph.node):
        if node.op_type != "Relu":
            continue
        onnx_model.graph.node.remove(node)
        layer_name = onnx_to_torch_name_cast(node.name, node.op_type)
        if "quantization" in sideband_info.keys():
            layer_name += ".1" # nn.Sequential(QuantAct, module, QuantAct)
        new_node_name = "/".join(node.name.split("/")[:-1] + ["ThresholdedReLU"])
        new_node = onnx.helper.make_node(
            "ThresholdedRelu",
            name= new_node_name,
            inputs=[*node.input],
            outputs=node.output,
            alpha = info[layer_name]
        )
        onnx_model.graph.node.insert(index, new_node)
        next_node = next(filter(lambda x: node.output[0] in x.input, onnx_model.graph.node))
        next_node.input[0] = new_node.output[0]

def _annotate_quantization(onnx_model, sideband_info):
    from quantization.utils import QuantMode

    if "quantization" not in sideband_info:
        return
    info = sideband_info["quantization"]
    for node in onnx_model.graph.node:
        if node.op_type in ["Conv", "Gemm"]:
            set_nodeattr(node, "weight_width", info["weight_width"])
            set_nodeattr(node, "data_width", info["data_width"])
            set_nodeattr(node, "acc_width", info["weight_width"] + info["data_width"])
            set_nodeattr(node, "block_floating_point", info["mode"] in [QuantMode.LAYER_BFP, QuantMode.CHANNEL_BFP])
        else:
            set_nodeattr(node, "data_width", info["data_width"])

def _annotate_sparsity(onnx_model, sideband_info):
    if "sparsity" not in sideband_info:
        return
    info = sideband_info["sparsity"]
    for node in onnx_model.graph.node:
        if node.op_type == 'Conv':
            layer_name = onnx_to_torch_name_cast(node.name, node.op_type)
            if "quantization" in sideband_info.keys():
                layer_name += ".1" # nn.Sequential(QuantAct, module, QuantAct)
            histograms_data = info[layer_name]["hist"].cpu().numpy()
            histograms_data = histograms_data/histograms_data.sum(axis = 1, keepdims=True) # normalize
            set_nodeattr(node, "channel_sparsity_hist", histograms_data.flatten())

def find_producer(onnx_graph, tensor_name):
    """Finds and returns the node that produces the tensor with given name, in onnx graph."""
    for x in onnx_graph.node:
        if tensor_name in x.output:
            return x
    return None

def find_consumers(onnx_graph, tensor_name):
    """Finds and returns a list of the nodes that consume tensor with given name, in onnx graph"""
    consumers = []
    for n in onnx_graph.node:
        for inp_tensor in n.input:
            if inp_tensor == tensor_name:
                consumers.append(n)
    return consumers

def _annotate_encoding(onnx_model, sideband_info):
    if "encoding" not in sideband_info:
        return
    info = sideband_info["encoding"]
    for node in onnx_model.graph.node:
        layer_name = onnx_to_torch_name_cast(node.name, node.op_type)
        if layer_name in info.keys():
            set_nodeattr(node, "input_compression_ratio", [info[layer_name]["input_compression_ratio"]])
            set_nodeattr(node, "output_compression_ratio", [info[layer_name]["output_compression_ratio"]])
            if node.op_type in ["Conv", "Gemm"]:
                set_nodeattr(node, "weight_compression_ratio", [info[layer_name]["weight_compression_ratio"]])
        else:
            if node.op_type == 'Constant':
                continue
            inputs = node.input
            outputs = node.output
            if node.op_type in ['Resize', 'Split']:
                inputs = [node.input[0]]
            input_compression_ratio = []
            for input_name in inputs:
                p_node = find_producer(onnx_model.graph, input_name)
                if p_node == None:
                    input_compression_ratio.append(1.0) # todo: fix missing info
                    continue
                p_name = onnx_to_torch_name_cast(p_node.name, p_node.op_type)
                if p_name in info.keys():
                    input_compression_ratio.append(info[p_name]["output_compression_ratio"])
                else:
                    input_compression_ratio.append(1.0) # todo: fix missing info
            set_nodeattr(node, "input_compression_ratio", input_compression_ratio)
            output_compression_ratio = []
            for output_name in outputs:
                c_node = find_consumers(onnx_model.graph, output_name)
                if len(c_node) == 0:
                    output_compression_ratio.append(1.0) # todo: fix missing info
                    continue
                c_node = c_node[0]
                c_name = onnx_to_torch_name_cast(c_node.name, c_node.op_type)
                if c_name in info.keys():
                    output_compression_ratio.append(info[c_name]["input_compression_ratio"])
                else:
                    output_compression_ratio.append(1.0) # todo: fix missing info
            set_nodeattr(node, "output_compression_ratio", output_compression_ratio)
                

def generate_onnx_files(self, output_path):
    model_copy = copy.deepcopy(self.model)
    self.load_model() # load f32 model

    # todo: add relu6 support
    replace_dict = {}
    for module in self.modules():
        if isinstance(module, nn.ReLU6):
            replace_dict[module] = nn.ReLU()
    self.replace_modules(replace_dict)

    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    # export f32 model
    f32_onnx_path = os.path.join(output_path, f"{self.model_name}_f32.onnx")
    self.onnx_exporter(f32_onnx_path)

    # if model is compressed
    if len(self.sideband_info) > 0: 
        onnx_model = onnx.load(f32_onnx_path)
        # note: the order of the following passes matters
        _insert_threshold_relu(onnx_model, self.sideband_info)
        _annotate_quantization(onnx_model, self.sideband_info)
        _annotate_sparsity(onnx_model, self.sideband_info)
        _annotate_encoding(onnx_model, self.sideband_info)
        fpgaconvnet_onnx_path = os.path.join(output_path, f"{self.model_name}.onnx")
        onnx.save(onnx_model, fpgaconvnet_onnx_path)
    else:
        fpgaconvnet_onnx_path = f32_onnx_path

    self.model = model_copy # restore model
    return fpgaconvnet_onnx_path