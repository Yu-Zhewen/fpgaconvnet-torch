import onnx
import os
import torch

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from models.base import TorchModelWrapper
from onnxsim import simplify


class MmsegmentationModelWrapper(TorchModelWrapper):
    def load_model(self, eval=True):
        assert self.model_name == 'unet'
        # todo: add mmseg as submodule?
        MMSEG_PATH = os.environ.get(
            "MMSEG_PATH", os.path.expanduser("../mmsegmentation"))
        config_path = os.path.join(
            MMSEG_PATH, "configs/unet/unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py")
        cfg = Config.fromfile(config_path)
        checkpoint_path = "https://download.openmmlab.com/mmsegmentation/v0.5/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes_20211210_145204-6860854e.pth"
        # runner only load checkpoint when running inference, too late for compression, as model is already substituted
        # cfg.load_from = checkpoint_path
        cfg.work_dir = os.path.join('./mmseg_work_dirs', self.model_name)
        cfg.data_root = os.environ.get(
            "CITYSCAPES_PATH", os.path.expanduser("~/dataset/cityscapes"))
        cfg.train_dataloader.dataset.data_root = cfg.data_root
        cfg.test_dataloader.dataset.data_root = cfg.data_root
        cfg.val_dataloader.dataset.data_root = cfg.data_root

        self.runner = Runner.from_cfg(cfg)
        self.model = self.runner.model
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_path, file_name=f"{self.model_name}.pth")[
            'state_dict']
        self.model.load_state_dict(state_dict)

    def load_data(self, batch_size, workers):  # todo: fix this
        # let the runner handle the data loading
        # todo: download cityscapes dataset
        # https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets
        pass

    def inference(self, mode="validate"):
        mode = "validate" if mode == "test" else mode
        print("Inference mode: {}".format(mode))
        if mode in ["validate", "calibrate"]:
            # todo: we should probably use the runner.test() method instead
            results = self.runner.val()
            print(results)

    def onnx_exporter(self, onnx_path):
        # todo: support other input sizes
        random_input = torch.randn(1, 3, 512, 1024)
        if torch.cuda.is_available():
            random_input = random_input.cuda()
        torch.onnx.export(self, random_input, onnx_path,
                          verbose=False, keep_initializers_as_inputs=True)

        model = onnx.load(onnx_path)
        model_simp, check = simplify(model)
        onnx.save(model_simp, onnx_path)
