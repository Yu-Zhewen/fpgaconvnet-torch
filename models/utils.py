import copy
import onnx
import os
import pathlib

import torch.nn as nn

def replace_modules(self, replace_dict):
    for name, module in self.model.named_modules(): 
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
        if buffer[-1] in onnx_name[i]:
            buffer[-1] = onnx_name[i]
        else:
            buffer.append(onnx_name[i])
    if buffer[-1] == onnx_type:
        buffer.pop()
    torch_name = ".".join(buffer)
    return torch_name

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
        _annotate_quantization(onnx_model, self.sideband_info)
        _annotate_sparsity(onnx_model, self.sideband_info)
        fpgaconvnet_onnx_path = os.path.join(output_path, f"{self.model_name}.onnx")
        onnx.save(onnx_model, fpgaconvnet_onnx_path)

    self.model = model_copy # restore model