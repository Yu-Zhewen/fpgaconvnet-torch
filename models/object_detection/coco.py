import os
import torch
import yaml

from models.base import TorchModelWrapper
# note: do NOT move ultralytic import to the top, otherwise the edit in settings will not take effect

class UltralyticsModelWrapper(TorchModelWrapper):

    def load_model(self, eval=True):
        from ultralytics import YOLO 
        self.yolo = YOLO(self.model_name)
        self.model = self.yolo.model
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # utlralytics conv bn fusion is currently not working for compressed model
        # disbale it for now
        def _fuse(verbose=True):
            return self.model
        self.model.fuse = _fuse

    def load_data(self, batch_size, workers):
        from ultralytics import settings

        DATASET_PATH = os.environ.get("COCO_PATH", os.path.expanduser("~/dataset/ultralytics/datasets"))
        assert DATASET_PATH.endswith("/datasets"), "dataset path should end with 'datasets'"
        # set dataset path
        settings.update({'datasets_dir': DATASET_PATH})

        # note: ultralytics automatically handle the dataloaders, only need to set the path
        self.data_loaders['calibrate'] = "coco128.yaml"
        self.data_loaders['validate'] = "coco128.yaml"
        
        self.batch_size = batch_size
        self.workers = workers
        
    def inference(self, mode="validate"):
        self.yolo.model = self.model
        return self.yolo.val(batch=self.batch_size, workers=self.workers,
            data=self.data_loaders[mode], plots=False)

    def onnx_exporter(self, onnx_path):
        path = self.yolo.export(format="onnx", simplify=True)
        os.rename(path, onnx_path)