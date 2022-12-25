import time
import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock
from torchvision.models.mobilenetv2 import InvertedResidual
import copy
import types

def validate(val_loader, model, criterion, print_freq=0):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if print_freq != 0 and i % print_freq == 0:
                progress.display(i)
            break
        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        
    return top1, top5

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,), no_reduce=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        if no_reduce:
            return correct

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def replace_modules(model, replace_dict):
    for name, module in model.named_modules(): 
        for subname, submodule in module.named_children():
            if submodule in replace_dict.keys():
                new_submodule = replace_dict[submodule]
                assert(hasattr(module, subname))
                setattr(module,subname,new_submodule)

# duplicate relu instances for quantisation 
class BasicBlockReluFixed(nn.Module):
    def __init__(self, origin_block):
        super(BasicBlockReluFixed, self).__init__()

        self.conv1 = origin_block.conv1
        self.bn1 = origin_block.bn1
        self.relu1 = nn.ReLU()

        self.conv2 = origin_block.conv2
        self.bn2 = origin_block.bn2
        self.relu2 = nn.ReLU()

        self.downsample = origin_block.downsample
        self.stride = origin_block.stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out

# remove residual connection to pass through fpgaConvNet parser
class BasicBlockNonResidual(nn.Module):
    def __init__(self, origin_block):
        super(BasicBlockNonResidual, self).__init__()

        self.conv1 = origin_block.conv1
        self.bn1 = origin_block.bn1
        self.relu1 = nn.ReLU()
        
        self.conv2 = origin_block.conv2
        self.bn2 = origin_block.bn2
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out

def fix_resnet(model, export_to_fpgaconvnet=False):
    replace_dict = {}

    for name, module in model.named_modules(): 
        if export_to_fpgaconvnet:
            if isinstance(module, BasicBlock):
                replace_dict[module] = BasicBlockNonResidual(copy.deepcopy(module))
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                replace_dict[module] = nn.AvgPool2d((7,7))
        else:
            if isinstance(module, BasicBlock):
                replace_dict[module] = BasicBlockReluFixed(copy.deepcopy(module))

    replace_modules(model, replace_dict)

def fix_mobilenet(model, export_to_fpgaconvnet=False):

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = self.avg_pool2d(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    model.avg_pool2d = nn.AvgPool2d((7,7))
    model._forward_impl = types.MethodType(_forward_impl, model)

    replace_dict = {}
    if export_to_fpgaconvnet:
        for name, module in model.named_modules():
            if isinstance(module, InvertedResidual):
                module.use_res_connect = False
            if isinstance(module, nn.ReLU6):
                replace_dict[module] = nn.ReLU()
    replace_modules(model, replace_dict)