import copy
import os
import timm
import torch
import torchvision

import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models.base import TorchModelWrapper
from models.classification.utils import _inference, BasicBlockReluFixed, BottleneckReluFixed, get_train_subset_indices
from torchvision.models.resnet import BasicBlock, Bottleneck

class ImagenetModelWrapper(TorchModelWrapper):
    def __init__(self, model_name, input_size=(1, 3, 224, 224), num_classes=1000):
        self.input_size = input_size # todo: support other input sizes
        self.num_classes = num_classes
        super().__init__(model_name)

    def load_data(self, batch_size, workers, calib_size=1000):
        # todo: download dataset
        # https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
        # https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh

        assert self.input_size[2] == 224, "todo: support other input sizes / transforms"
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        IMAGENET_PATH = os.environ.get("IMAGENET_PATH", os.path.expanduser("~/dataset/ILSVRC2012_img"))
        valdir = os.path.join(IMAGENET_PATH, 'val')
        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        val_loader = data.DataLoader(
            datasets.ImageFolder(valdir, val_transforms),
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)
        self.data_loaders['validate'] = val_loader

        traindir = os.path.join(IMAGENET_PATH, 'train')
        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_dataset = datasets.ImageFolder(traindir, train_transforms)
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)
        self.data_loaders['train'] = train_loader
        
        train_labels = [sample[1] for sample in train_dataset.samples]
        calib_indices = get_train_subset_indices(train_labels, calib_size, self.num_classes)
        calib_sampler = data.sampler.SubsetRandomSampler(calib_indices)
        calib_dataset = datasets.ImageFolder(traindir, val_transforms)
        calib_loader = data.DataLoader(
            calib_dataset,
            batch_size=batch_size, sampler=calib_sampler,
            num_workers=workers, pin_memory=True)
        self.data_loaders['calibrate'] = calib_loader

    def inference(self, mode="validate"):
        mode = "validate" if mode == "test" else mode
        print("Inference mode: {}".format(mode))
        return _inference(self.data_loaders[mode], self.model, nn.CrossEntropyLoss())


class TorchvisionModelWrapper(ImagenetModelWrapper):
    # https://github.com/pytorch/vision

    def load_model(self, eval=True):
        self.model = torchvision.models.__dict__[self.model_name](pretrained=True)
        self.model_fixer()
        # todo: fuse bn
        if torch.cuda.is_available():
            self.model.cuda()

    def model_fixer(self):
        replace_dict = {}
        for name, module in self.named_modules(): 
            if isinstance(module, BasicBlock):
                replace_dict[module] = BasicBlockReluFixed(copy.deepcopy(module))
            elif isinstance(module, Bottleneck):
                replace_dict[module] = BottleneckReluFixed(copy.deepcopy(module))
        self.replace_modules(replace_dict)


class TimmModelWrapper(ImagenetModelWrapper):
    # https://github.com/huggingface/pytorch-image-models

    def load_model(self, eval=True):
        self.model = timm.create_model(self.model_name, pretrained=True)
        if eval:
            self.model = timm.utils.model.reparameterize_model(self.model)
            # todo: fuse bn
        if torch.cuda.is_available():
            self.model.cuda()