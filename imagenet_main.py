import argparse
import os
import random

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sparsity_utils import *
from quan_utils import *

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet')
parser.add_argument('--data', metavar='DIR', default="~/dataset/ILSVRC2012_img",
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names))
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

def imagenet_main():
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True)
    random_input = torch.randn(1, 3, 224, 224)

    if args.gpu is not None:
        print("Use GPU: {}".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        random_input = random_input.cuda()
    else:
        print('using CPU, this will be slow')
   
    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.arch == "resnet18":
        fix_resnet(model, export_to_fpgaconvnet=False)

    #torch.onnx.export(model, random_input, args.arch+".onnx", verbose=False, keep_initializers_as_inputs=True) 

    model_quantisation(model, val_loader)

    replace_with_vanilla_convolution(model)
    handle_list = regsiter_hooks(model, args.coarse_in)

    validate(val_loader, model, criterion, args.print_freq)

    output_sparsity_to_csv(args.arch, model, accum_input=True)
    delete_hooks(model, handle_list)

if __name__ == '__main__':
    imagenet_main()
