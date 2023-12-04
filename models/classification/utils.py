import random
import time
import torch

import torch.nn as nn

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

def _inference(data_loader, model, criterion, print_freq=0, silence=False):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(data_loader):
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

        if not silence:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                .format(top1=top1, top5=top5))
        
    return top1.avg.item(), top5.avg.item()

# todo: turn this into a general transformation
# separate relu instances 
class BasicBlockReluFixed(nn.Module):
    def __init__(self, origin_block):
        super(BasicBlockReluFixed, self).__init__()

        self.conv1 = origin_block.conv1
        self.bn1 = origin_block.bn1
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = origin_block.conv2
        self.bn2 = origin_block.bn2
        self.relu2 = nn.ReLU(inplace=True)

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

# todo: turn this into a general transformation
# separate relu instances 
class BottleneckReluFixed(nn.Module):
    def __init__(self, origin_block):
        super(BottleneckReluFixed, self).__init__()

        self.conv1 = origin_block.conv1
        self.bn1 = origin_block.bn1
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = origin_block.conv2
        self.bn2 = origin_block.bn2
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = origin_block.conv3
        self.bn3 = origin_block.bn3
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = origin_block.downsample
        self.stride = origin_block.stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out

def get_train_subset_indices(train_labels, subset_size, num_classes):
    if subset_size % num_classes == 0:
        # M-class-N-shot few sample
        rand_indices = torch.randperm(len(train_labels)).tolist()
        per_class_remain = [subset_size // num_classes] * num_classes
        calib_indices = []
        for idx in rand_indices:
            label = train_labels[idx]
            if per_class_remain[label] > 0:
                calib_indices.append(idx)
                per_class_remain[label] -= 1
    else:
        # random split
        rand_indices = torch.randperm(len(train_dataset)).tolist()
        calib_indices = random.choices(rand_indices, k=subset_size)
    return calib_indices