import os
import torch

import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models.base import TorchModelWrapper
from torchvision.datasets import CIFAR10, CIFAR100
from models.classification.utils import _inference, get_train_subset_indices

class ChenyaofoModelWrapper(TorchModelWrapper):
    # https://github.com/chenyaofo/pytorch-cifar-models
    
    def __init__(self, model_name, input_size=(1, 3, 32, 32), num_classes=10):
        self.input_size = input_size
        self.num_classes = num_classes
        super().__init__(model_name)

    def load_model(self):
        self.model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar{self.num_classes}_{self.model_name}", pretrained=True)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def load_data(self, batch_size, workers, calib_size=5000):
        if self.num_classes == 10:
            dataset_builder = CIFAR10
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.2023, 0.1994, 0.2010]
            root = os.environ.get("CIFAR_10_PATH", os.path.expanduser("~/dataset/cifar10"))
        elif self.num_classes == 100:
            dataset_builder = CIFAR100
            mean = [0.5070, 0.4865, 0.4409]
            std = [0.2673, 0.2564, 0.2761]
            root = os.environ.get("CIFAR_100_PATH", os.path.expanduser("~/dataset/cifar100"))

        root = os.environ['CIFAR_PATH']
        #train_transforms = transforms.Compose([
        #    transforms.RandomCrop(32, padding=4),
        #    transforms.RandomHorizontalFlip(),
        #    transforms.ToTensor(),
        #    transforms.Normalize(mean=mean, std=std)])

        val_transforms = transforms.Compose([   
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        #train_set = dataset_builder(root, train=True, transform=train_transforms, download=True)
        calib_set = dataset_builder(root, train=True, transform=val_transforms, download=True)
        test_set = dataset_builder(root, train=False, transform=val_transforms, download=True)

        calib_indices = get_train_subset_indices(calib_set.targets, calib_size, self.num_classes)
        calib_sampler = data.sampler.SubsetRandomSampler(calib_indices)
        calib_loader = data.DataLoader(calib_set,
            batch_size=batch_size, sampler=calib_sampler,
            num_workers=workers, pin_memory=True)

        test_loader = data.DataLoader(test_set, 
            batch_size=batch_size, shuffle=False, 
            num_workers=workers, pin_memory=True)

        self.data_loaders['test'] = test_loader
        self.data_loaders['calibrate'] = calib_loader

    def inference(self, mode="test"):
        print("Inference mode: {}".format(mode))
        return _inference(self.data_loaders[mode], self.model, nn.CrossEntropyLoss())