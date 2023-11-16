import onnx
import os
import torch

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmengine.runner import load_checkpoint
from models.base import TorchModelWrapper
from onnxsim import simplify


class MmactionModelWrapper(TorchModelWrapper):

    def load_model(self, val=True):
        assert self.model_name in ["x3d_s", "x3d_m"]
        # todo: add mmaction2 as submodule?
        MMACTION_PATH = os.environ.get(
            "MMACTION_PATH", os.path.expanduser("../mmaction2"))

        match self.model_name:
            case "x3d_s":
                config_path = os.path.join(
                    MMACTION_PATH, "configs/recognition/x3d/x3d_s_13x6x1_facebook-kinetics400-rgb.py")
                checkpoint_path = "https://drive.google.com/uc?export=download&id=1vD6XdN9biOnWKFp-BvDC5QCer2on_jLi"
            case "x3d_m":
                config_path = os.path.join(
                    MMACTION_PATH, "configs/recognition/x3d/x3d_m_16x5x1_facebook-kinetics400-rgb.py")
                checkpoint_path = "https://drive.google.com/uc?export=download&id=1l6x6LOmSfpugMOSuEZYb4foRIC8jXMQU"

        cfg = Config.fromfile(config_path)
        # runner only load checkpoint when running inference, too late for compression, as model is already substituted
        # cfg.load_from = checkpoint_path

        cfg.work_dir = os.path.join('./mmaction2_work_dirs', self.model_name)
        cfg.data_root = os.path.join(os.environ.get(
            "UCF101_PATH", os.path.expanduser("~/dataset/ucf101")), "videos")
        cfg.data_root_val = cfg.data_root
        cfg.ann_file_test = os.path.join(os.environ.get("UCF101_PATH", os.path.expanduser(
            "~/dataset/ucf101")), "testlist01_mmaction_videos.txt")
        cfg.model.cls_head.num_classes = 101

        cfg.test_dataloader.dataset.data_prefix = dict(video=cfg.data_root)
        cfg.test_dataloader.dataset.ann_file = cfg.ann_file_test

        cfg.test_dataloader.batch_size = 8
        cfg.test_dataloader.num_workers = 8

        # cfg.log_level = "WARNING"
        self.runner = Runner.from_cfg(cfg)
        self.model = self.runner.model
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_path)[
            'state_dict']
        self.model.load_state_dict(state_dict)
        # load_checkpoint(self.model, checkpoint_path, map_location="cpu")

    def load_data(self, batch_size, workers):  # todo: fix this
        # let the runner handle the data loading
        # todo: download ucf101 dataset (https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)
        # todo: dowload Train/Test Splits for Action Recognition on UCF101 (https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip)
        # todo: prepare the dataset following the guidelines from mmaction (https://github.com/open-mmlab/mmaction2/blob/main/docs/en/user_guides/2_dataset_prepare.md#prepare-datasets)
        pass

    def inference(self, mode="validate"):
        mode = "validate" if mode == "test" else mode
        print("Inference mode: {}".format(mode))
        if mode in ["validate", "calibrate"]:
            results = self.runner.test()
            print(results)

    def onnx_exporter(self, onnx_path):
        # todo: support other input sizes
        random_input = torch.randn(1, 1, 3, 16, 256, 256)
        if torch.cuda.is_available():
            random_input = random_input.cuda()
        torch.onnx.export(self.model, random_input, onnx_path,
                          verbose=False, keep_initializers_as_inputs=True)

        model = onnx.load(onnx_path)
        model_simp, check = simplify(model)
        onnx.save(model_simp, onnx_path)
