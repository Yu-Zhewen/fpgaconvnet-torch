import torch.nn as nn
from abc import ABC, abstractmethod

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

    from models.utils import replace_modules
    from models.utils import export_onnx