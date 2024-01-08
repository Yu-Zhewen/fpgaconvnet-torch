from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, parameter_count


class TorchModelWrapper(nn.Module, ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.data_loaders = {}
        self.sideband_info = {}
        super().__init__()

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def inference(self):
        pass

    def onnx_exporter(self, onnx_path):
        random_input = torch.randn(self.input_size)
        if torch.cuda.is_available():
            random_input = random_input.cuda()
        torch.onnx.export(self, random_input, onnx_path, verbose=False, keep_initializers_as_inputs=True)

    def forward(self, x):
        return self.model(x)

    def replace_modules(self, replace_dict):
        from models.utils import replace_modules
        replace_modules(self.model, replace_dict)

    def profile(self):
        random_input = torch.randn(self.input_size)
        flop_counter = FlopCountAnalysis(self.model, random_input)
        # ignore batch norm
        flop_counter._ignored_ops.add("aten::batch_norm")
        del flop_counter._op_handles["aten::batch_norm"]

        param_counter = parameter_count(self.model)
        macs = flop_counter.total()
        params = param_counter['']

        print(f"MACs: {macs}, Params: {params}")

    from models.utils import generate_onnx_files