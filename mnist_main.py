import argparse
import os
import random
from urllib.request import urlretrieve
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sparsity_utils import *
from quan_utils import *

LENET5 = nn.Sequential(
    nn.Conv2d(1, 6, (5, 5), stride=1, padding=0),  # (1,28,28) -> (6,24,24)
    nn.MaxPool2d(2),  # (6,24,24) -> (6,12,12)
    nn.ReLU(),
    nn.Conv2d(6, 16, (5, 5), stride=1, padding=0),  # (6,12,12) -> (16,8,8)
    nn.MaxPool2d(2),  # (16,8,8) -> (16,4,4)
    nn.ReLU(),
    nn.Flatten(),  # (16,4,4) -> (256,)
    nn.Linear(256, 120),  # (256,) -> (120,)
    nn.ReLU(),
    nn.Linear(120, 84),  # (120,) -> (84,)
    nn.ReLU(),
    nn.Linear(84, 10),  # (84,) -> (10,)
    nn.LogSoftmax(dim=1),  # (10,) log probabilities
)

parser = argparse.ArgumentParser(description='PyTorch MNIST')
parser.add_argument('--data', metavar='DIR', default="~/dataset",
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='lenet5',
                    choices=['lenet5'])
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('-p', '--print-freq', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

parser.add_argument('--coarse_in', default=-1, type=int,
                    help='')

def mnist_main():
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    model = LENET5

    MODEL_DIR = Path(".")
    MODEL_DIR.mkdir(exist_ok=True)
    filename = "mnist_classifier.pth"
    model_path = MODEL_DIR / filename

    if not model_path.is_file():
        WEB_DIR = "https://raw.githubusercontent.com/icaros-usc/pyribs/master/examples/tutorials/_static/"
        urlretrieve(WEB_DIR + filename, str(model_path))

    state_dict = torch.load(str(MODEL_DIR / "mnist_classifier.pth"), map_location='cpu')
    LENET5.load_state_dict(state_dict)
    random_input = torch.randn(1, 1, 28, 28)

    if args.gpu is not None:
        print("Use GPU: {}".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        random_input = random_input.cuda()
    else:
        print('using CPU, this will be slow')

    #torch.onnx.export(model, random_input, args.arch+".onnx", verbose=False, keep_initializers_as_inputs=True)  

    # define loss function (criterion)
    criterion = nn.NLLLoss().cuda(args.gpu)

    # Data loading code
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(args.data, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model_quantisation(model, test_loader)

    replace_with_vanilla_convolution(model)
    handle_list = regsiter_hooks(model, args.coarse_in)

    validate(test_loader, model, criterion, args.print_freq)

    output_sparsity_to_csv(args.arch, model, accum_input=True)
    delete_hooks(model, handle_list)

if __name__ == '__main__':
    mnist_main()