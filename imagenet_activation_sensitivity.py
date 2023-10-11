import argparse
import os
import random

import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from utils import *
from sparsity_utils import *
from quan_utils import *
from relu_utils import *

from fpgaconvnet.parser.Parser import Parser


parser = argparse.ArgumentParser(description='PyTorch ImageNet')
parser.add_argument('--data', metavar='DIR', default="~/dataset/ILSVRC2012_img",
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names))

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')


parser.add_argument('--ma_window_size', default=None, type=int,
                    help='')
parser.add_argument('--calibration-size', default=4, type=int,
                    help='')

parser.add_argument("--accuracy_output",  default=None, type=str,
                    help='Path to csv file to write accuracy to')



def imagenet_main():
    args = parser.parse_args()

    # if args.output_path == None:
    #     output_dir = str(args.arch) + "_output_relu_" + str(args.relu_threshold)
    #     if not os.path.isdir(output_dir):
    #         os.makedirs(output_dir)
    #     args.output_path = os.path.join(os.getcwd(), output_dir)

    print(args)

    random.seed(0)
    torch.manual_seed(0)

    # create model
    print("=> using pre-trained model '{}'".format(args.arch))
    model = load_model(args.arch)
    random_input = torch.randn(1, 3, 224, 224)

    if args.gpu is not None:
        print("Use GPU: {}".format(args.gpu))
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        random_input = random_input.cuda()
        valdir = os.path.join(args.data, 'val')
        traindir = os.path.join(args.data, 'train')
    else:
        print('using CPU, this will be slow')
        valdir = os.path.join(args.data, 'val')
        traindir = os.path.join(args.data, 'val')

    print("Calculating MACs and Params")
    calculate_macs_params(model, random_input, False, inference_mode=True)
    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # Data loading code
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


    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))
    # calibrate_size = 50000
    calibrate_size = args.calibration_size
    # per class few sampling, different from random_split
    # https://github.com/mit-han-lab/proxylessnas/blob/6e7a96b7190963e404d1cf9b37a320501e62b0a0/search/data_providers/imagenet.py#L21
    # assert calibrate_size % 1000 == 0
    """
    rand_indexes = torch.randperm(len(train_dataset)).tolist()
    train_labels = [sample[1] for sample in train_dataset.samples]
    per_class_remain = [calibrate_size // 1000] * 1000
    train_indexes, calibrate_indexes = [], []
    for idx in rand_indexes:
        label = train_labels[idx]
        if per_class_remain[label] > 0:
            calibrate_indexes.append(idx)
            per_class_remain[label] -= 1
        else:
            train_indexes.append(idx)
    """
    #Randomness handled by seeds
    rand_indexes = torch.randperm(len(train_dataset)).tolist()
    calibrate_indexes = random.choices(rand_indexes, k=calibrate_size)

    #train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
    calibrate_sampler = torch.utils.data.sampler.SubsetRandomSampler(calibrate_indexes)

    #train_loader = torch.utils.data.DataLoader(
    #    train_dataset,
    #    batch_size=args.batch_size,
    #    num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    calibrate_dataset = datasets.ImageFolder(traindir, transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]))

    calibrate_loader = torch.utils.data.DataLoader(
        calibrate_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, sampler=calibrate_sampler)


    #-----------------Model Quantisation----------------
    # todo: measure post-quantisation results???
    print("Quantising model")
    model_quantisation(model, calibrate_loader, quantization_method=QuanMode.NETWORK_FP, weight_width=16, data_width=16)
    print("Model quantised")
    original_top1, original_top5 = validate(val_loader, model, criterion)
    print("Accuracy above is for quantised model")
    original_top1 = float(str(original_top1).split("( ")[1][:-1])
    original_top5 = float(str(original_top5).split("( ")[1][:-1])
    # use vanilla convolution to measure
    # post-activation (post-sliding-window, to be more precise) sparsity

    #-----------------Variable ReLU Sensitivity---------------------
    relu_list = []
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):#or isinstance(module, nn.Linear):
            relu_list.append(name)

    model_copy = copy.deepcopy(model)

    for relu_layer in relu_list:
        min_thresh = 0
        max_thresh = 20
        while (max_thresh - min_thresh) > 0.01:
            recorded = False
            for threshold in np.linspace(min_thresh, max_thresh, 21):
                model = copy.deepcopy(model_copy)
                replace_layer_with_variable_relu(model, relu_layer, threshold=threshold)
                print("Variable ReLU added")
                top1, top5 = validate(val_loader, model, criterion)
                print("Accuracy above is for " + str(relu_layer) + " with ReLU threshold:" + str(threshold))
                top1 = str(top1).split("( ")[1][:-1]
                top5 = str(top5).split("( ")[1][:-1]
                output_dir = args.accuracy_output + "/" + str(args.arch)
                output_accuracy_to_csv(args.arch, threshold, relu_layer, top1, top5, output_dir)
                if float(top5) < 0.99*original_top5 and not recorded:
                    min_thresh, max_thresh = threshold - (max_thresh - min_thresh)/20, threshold
                    recorded = True


if __name__ == '__main__':
    imagenet_main()

#