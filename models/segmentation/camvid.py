import os
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from collections import OrderedDict
from models.base import TorchModelWrapper
from models.segmentation.utils import apply_conv_transp_approx
from PIL import Image
from torch.utils import data


class NncfModelWrapper(TorchModelWrapper):
    # https://github.com/openvinotoolkit/nncf

    def __init__(self, model_name, input_size=(1, 3, 368, 480), num_classes=12):
        self.input_size = input_size
        self.num_classes = num_classes
        super().__init__(model_name)

    def load_model(self, eval=True, approx_transpose_conv=True):
        assert self.model_name == 'unet'

        self.model = UNet(input_size_hw=self.input_size[2:], in_channels=self.input_size[1], n_classes=self.num_classes)
        checkpoint = torch.hub.load_state_dict_from_url('https://storage.openvinotoolkit.org/repositories/nncf/models/v2.6.0/torch/unet_camvid.pth', file_name="unet_camvid.pth")
        state_dict = checkpoint['state_dict']
        # remove 'module.' prefix
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

        if approx_transpose_conv:
            apply_conv_transp_approx(self.model)

        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def load_data(self, batch_size, workers):
        # todo: download dataset
        # https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid
        CAMVID_PATH = os.environ.get("CAMVID_PATH", os.path.expanduser("~/dataset/CamVid"))

        val_transforms = Compose([
                    Resize(size=self.input_size[2:]),
                    ToTensor(),
                    Normalize(mean=[0.39068785, 0.40521392, 0.41434407], std=[0.29652068, 0.30514979, 0.30080369])
                ])
        val_data = CamVid(CAMVID_PATH, "val", transforms=val_transforms)
        test_data = CamVid(CAMVID_PATH, "test", transforms=val_transforms)

        val_loader = data.DataLoader(
            val_data,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            collate_fn=collate_fn)
        test_loader = data.DataLoader(
            test_data,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True,
            collate_fn=collate_fn)

        self.data_loaders['calibrate'] = val_loader
        self.data_loaders['validate'] = val_loader
        self.data_loaders['test'] = test_loader

    def inference(self, mode="validate"):
        mode = "validate" if mode == "test" else mode
        print("Inference mode: {}".format(mode))
        data_loader = self.data_loaders[mode]
        self.model.eval()

        def _center_crop(layer, target_size):
            if layer.dim() == 4:
                # Cropping feature maps
                _, _, layer_height, layer_width = layer.size()
                diff_y = (layer_height - target_size[0]) // 2
                diff_x = (layer_width - target_size[1]) // 2
                return layer[:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]

            # If dimension is not 4, assume that we are cropping ground truth labels
            assert layer.dim() == 3
            _, layer_height, layer_width = layer.size()
            diff_y = (layer_height - target_size[0]) // 2
            diff_x = (layer_width - target_size[1]) // 2
            return layer[:, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]

        ignore_index = list(data_loader.dataset.color_encoding).index("unlabeled")
        metric = IoU(len(data_loader.dataset.color_encoding), ignore_index=ignore_index)
        metric.reset()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(data_loader):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = self.model(inputs)
                metric.add(outputs.detach(), labels.detach())

        mIOU = metric.value()[1]
        print("mIOU: ", mIOU)
        return mIOU
        

def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., : img.shape[-2], : img.shape[-1]].copy_(img)
    return batched_imgs

def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = transforms.functional.resize(image, self.size)
        target = transforms.functional.resize(target, self.size, interpolation=transforms.InterpolationMode.NEAREST)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        image = transforms.functional.to_tensor(image)
        target = torch.as_tensor(np.asarray(target).copy(), dtype=torch.int64)
        return image, target

class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = transforms.functional.normalize(image, mean=self.mean, std=self.std)
        return image, target


# https://github.com/openvinotoolkit/nncf/blob/master/examples/torch/common/models/segmentation/unet.py
class UNet(nn.Module):
    def __init__(
        self,
        input_size_hw,
        in_channels=3,
        n_classes=2,
        depth=5,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode="upconv",
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Args:
            in_channels (int): number of input channels
            input_size_hw: a tuple of (height, width) of the input images
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm prior to layers with an
                               activation function
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super().__init__()
        assert up_mode in ("upconv", "upsample")
        if (input_size_hw[0] % 2 ** (depth - 1)) or (input_size_hw[1] % 2 ** (depth - 1)):
            raise ValueError("UNet may only operate on input resolutions aligned to 2**(depth - 1)")
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, 2 ** (wf + i), up_mode, padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        x = self.last(x)
        #if is_tracing_state() and version.parse(torch.__version__) >= version.parse("1.1.0"):
            # While exporting, add extra post-processing layers into the graph
            # so that the model outputs class probabilities instead of class scores
        #    softmaxed = F.softmax(x, dim=1)
        #    return softmaxed
        return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super().__init__()
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ReLU())

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        block.append(nn.ReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


def center_crop(layer, target_size):
    if layer.dim() == 4:
        # Cropping feature maps
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]

    # If dimension is not 4, assume that we are cropping ground truth labels
    assert layer.dim() == 3
    _, layer_height, layer_width = layer.size()
    diff_y = (layer_height - target_size[0]) // 2
    diff_x = (layer_width - target_size[1]) // 2
    return layer[:, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])]


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super().__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )
        self.conv_block = UNetConvBlock(in_size, out_size, padding, batch_norm)
        self.padding = padding

    def forward(self, x, bridge):
        up = self.up(x)
        if self.padding:
            out = torch.cat([up, bridge], 1)
        else:
            crop1 = center_crop(bridge, up.shape[2:])
            out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


def unet(num_classes, pretrained=False, **kwargs):
    model = UNet(n_classes=num_classes, **kwargs)

    return model


def pil_loader(data_path, label_path):
    """Loads a sample and label image given their path as PIL images.

    Keyword arguments:
    - data_path (``string``): The filepath to the image.
    - label_path (``string``): The filepath to the ground-truth image.

    Returns the image and the label as PIL images.

    """
    data = Image.open(data_path)
    label = Image.open(label_path)

    return data, label

def get_files(folder, name_filter=None, extension_filter=None):
    """Helper function that returns the list of files in a specified folder
    with a specified extension.

    Keyword arguments:
    - folder (``string``): The path to a folder.
    - name_filter (```string``, optional): The returned files must contain
    this substring in their filename. Default: None; files are not filtered.
    - extension_filter (``string``, optional): The desired file extension.
    Default: None; files are not filtered

    """
    if not os.path.isdir(folder):
        raise RuntimeError('"{0}" is not a folder.'.format(folder))

    # Filename filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files that do not
    # contain "name_filter"
    if name_filter is None:
        # This looks hackish...there is probably a better way
        name_cond = lambda filename: True
    else:
        name_cond = lambda filename: name_filter in filename

    # Extension filter: if not specified don't filter (condition always true);
    # otherwise, use a lambda expression to filter out files whose extension
    # is not "extension_filter"
    if extension_filter is None:
        # This looks hackish...there is probably a better way
        ext_cond = lambda filename: True
    else:
        ext_cond = lambda filename: filename.endswith(extension_filter)

    filtered_files = []

    # Explore the directory tree to get files that contain "name_filter" and
    # with extension "extension_filter"
    for path, _, files in os.walk(folder):
        files.sort()
        for file in files:
            if name_cond(file) and ext_cond(file):
                full_path = os.path.join(path, file)
                filtered_files.append(full_path)

    return filtered_files


class CamVid(data.Dataset):
    """CamVid dataset loader where the dataset is arranged as in
    https://github.com/alexgkendall/SegNet-Tutorial/tree/master/CamVid.


    Keyword arguments:
    - root_dir (``string``): Root directory path.
    - mode (``string``): The type of dataset: 'train' for training set, 'val'
    for validation set, and 'test' for test set.
    - transform (``callable``, optional): A function/transform that  takes in
    an PIL image and returns a transformed version. Default: None.
    - label_transform (``callable``, optional): A function/transform that takes
    in the target and transforms it. Default: None.
    - loader (``callable``, optional): A function to load an image given its
    path. By default ``default_loader`` is used.

    """

    # Training dataset root folders
    train_folder = "train"
    train_lbl_folder = "trainannot"

    # Validation dataset root folders
    val_folder = "val"
    val_lbl_folder = "valannot"

    # Test dataset root folders
    test_folder = "test"
    test_lbl_folder = "testannot"

    # Images extension
    img_extension = ".png"

    # Default encoding for pixel value, class name, and class color
    color_encoding = OrderedDict(
        [
            ("sky", (128, 128, 128)),
            ("building", (128, 0, 0)),
            ("pole", (192, 192, 128)),
            #("road_marking", (255, 69, 0)),
            ("road", (128, 64, 128)),
            ("pavement", (60, 40, 222)),
            ("tree", (128, 128, 0)),
            ("sign_symbol", (192, 128, 128)),
            ("fence", (64, 64, 128)),
            ("car", (64, 0, 128)),
            ("pedestrian", (64, 64, 0)),
            ("bicyclist", (0, 128, 192)),
            ("unlabeled", (0, 0, 0)),
        ]
    )


    def __init__(self, root, image_set="train", transforms=None, loader=pil_loader):
        super().__init__()
        self.root_dir = root
        self.mode = image_set
        self.transforms = transforms
        self.loader = loader

        if self.mode.lower() == "train":
            # Get the training data and labels filepaths
            self.train_data = get_files(
                os.path.join(self.root_dir, self.train_folder), extension_filter=self.img_extension
            )

            self.train_labels = get_files(
                os.path.join(self.root_dir, self.train_lbl_folder), extension_filter=self.img_extension
            )
        elif self.mode.lower() == "val":
            # Get the validation data and labels filepaths
            self.val_data = get_files(
                os.path.join(self.root_dir, self.val_folder), extension_filter=self.img_extension
            )

            self.val_labels = get_files(
                os.path.join(self.root_dir, self.val_lbl_folder), extension_filter=self.img_extension
            )
        elif self.mode.lower() == "test":
            # Get the test data and labels filepaths
            self.test_data = get_files(
                os.path.join(self.root_dir, self.test_folder), extension_filter=self.img_extension
            )

            self.test_labels = get_files(
                os.path.join(self.root_dir, self.test_lbl_folder), extension_filter=self.img_extension
            )
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test")

    def __getitem__(self, index):
        """
        Args:
        - index (``int``): index of the item in the dataset

        Returns:
        A tuple of ``PIL.Image`` (image, label) where label is the ground-truth
        of the image.

        """
        if self.mode.lower() == "train":
            data_path, label_path = self.train_data[index], self.train_labels[index]
        elif self.mode.lower() == "val":
            data_path, label_path = self.val_data[index], self.val_labels[index]
        elif self.mode.lower() == "test":
            data_path, label_path = self.test_data[index], self.test_labels[index]
        else:
            raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test")

        img, label = self.loader(data_path, label_path)

        if self.transforms is not None:
            img, label = self.transforms(img, label)

        return img, label

    def __len__(self):
        """Returns the length of the dataset."""
        if self.mode.lower() == "train":
            return len(self.train_data)
        if self.mode.lower() == "val":
            return len(self.val_data)
        if self.mode.lower() == "test":
            return len(self.test_data)

        raise RuntimeError("Unexpected dataset mode. Supported modes are: train, val and test")

class Metric:
    """Base class for all metrics.

    From: https://github.com/pytorch/tnt/blob/master/torchnet/meter/meter.py
    """

    def reset(self):
        pass

    def add(self):
        pass

    def value(self):
        pass


class ConfusionMatrix(Metric):
    """Constructs a confusion matrix for a multi-class classification problems.

    Does not support multi-label, multi-class problems.

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.

    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, normalized=False):
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int32)
        self.normalized = normalized
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix

        The shape of the confusion matrix is K x K, where K is the number
        of classes.

        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.

        """
        # If target and/or predicted are tensors, convert them to numpy arrays
        if torch.is_tensor(predicted):
            predicted = predicted.cpu().numpy()
        if torch.is_tensor(target):
            target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], "number of targets and predicted outputs do not match"

        if np.ndim(predicted) != 1:
            assert (
                predicted.shape[1] == self.num_classes
            ), "number of predictions does not match size of confusion matrix"
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.num_classes) and (
                predicted.min() >= 0
            ), "predicted values are not between 0 and k-1"

        if np.ndim(target) != 1:
            assert target.shape[1] == self.num_classes, "Onehot target does not match size of confusion matrix"
            assert (target >= 0).all() and (target <= 1).all(), "in one-hot encoding, target values should be 0 or 1"
            assert (target.sum(1) == 1).all(), "multi-label setting is not supported"
            target = np.argmax(target, 1)

        # Ignore out-of-bounds target labels
        valid_indices = np.where((target >= 0) & (target < self.num_classes))
        target = target[valid_indices]
        predicted = predicted[valid_indices]

        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.num_classes**2)

        # See Pylint issue #2721
        # pylint: disable=no-member
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]

        return self.conf


class IoU(Metric):
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).

    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix(num_classes, normalized)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError as e:
                raise ValueError("'ignore_index' must be an int or iterable") from e

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.

        Keyword arguments:
        - predicted (Tensor): Can be a (N, K, H, W) tensor of
        predicted scores obtained from the model for N examples and K classes,
        or (N, H, W) tensor of integer values between 0 and K-1.
        - target (Tensor): Can be a (N, K, H, W) tensor of
        target scores for N examples and K classes, or (N, H, W) tensor of
        integer values between 0 and K-1.

        """
        # Dimensions check
        assert predicted.size(0) == target.size(0), "number of targets and predicted outputs do not match"
        assert (
            predicted.dim() == 3 or predicted.dim() == 4
        ), "predictions must be of dimension (N, H, W) or (N, K, H, W)"
        assert target.dim() == 3 or target.dim() == 4, "targets must be of dimension (N, H, W) or (N, K, H, W)"

        # If the tensor is in categorical format convert it to integer format
        if predicted.dim() == 4:
            _, predicted = predicted.max(1)
        if target.dim() == 4:
            _, target = target.max(1)

        self.conf_metric.add(predicted.view(-1), target.view(-1))

    def value(self):
        """Computes the IoU and mean IoU.

        The mean computation ignores NaN elements of the IoU array.

        Returns:
            Tuple: (IoU, mIoU). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.
        """
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            for _ in self.ignore_index:
                conf_matrix[:, self.ignore_index] = 0
                conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive
        false_negative = np.sum(conf_matrix, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide="ignore", invalid="ignore"):
            iou = true_positive / (true_positive + false_positive + false_negative)

        if self.ignore_index is not None:
            iou_valid_cls = np.delete(iou, self.ignore_index)
            miou = np.nanmean(iou_valid_cls)
        else:
            miou = np.nanmean(iou)
        return iou, miou