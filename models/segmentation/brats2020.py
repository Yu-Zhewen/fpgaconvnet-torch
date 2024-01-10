import os

import nibabel as nib
import numpy as np
import onnx
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations import Compose
from onnxsim import simplify
from skimage.transform import resize
from sklearn.model_selection import StratifiedKFold
from torch.utils import data
from tqdm import tqdm

from models.base import TorchModelWrapper
from models.segmentation.utils import apply_conv_transp_approx


class Unet3DKaggleModelWrapper(TorchModelWrapper):
    # https://www.kaggle.com/code/polomarco/brats20-3dunet-3dautoencoder/notebook
    def __init__(self, model_name, input_size=(1, 4, 155, 240, 240), num_classes=3):
        self.input_size = input_size
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(model_name)

    def load_model(self, eval=True, approx_transpose_conv=True):
        assert self.model_name == 'unet3d'

        self.model = UNet3d(
            in_channels=self.input_size[1], n_classes=self.num_classes, n_channels=24).to(self.device)

        checkpoint_path = "https://drive.google.com/uc?export=download&id=1NiyVXIr5zcnd3F-zNi3FCj6PmYZnabH-"
        state_dict = torch.hub.load_state_dict_from_url(
            checkpoint_path, file_name=f"{self.model_name}.pth", map_location=self.device)

        self.model.load_state_dict(state_dict, strict=True)

        if approx_transpose_conv:
            apply_conv_transp_approx(self.model)

        self.model = self.model.to(self.device)

    def load_data(self, batch_size, workers):
        BraTS2020_PATH = os.environ.get("BraTS2020_PATH", os.path.expanduser("~/dataset/BraTS2020"))

        dataset = BraTS2020(BraTS2020_PATH, "validation")

        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=True)

        self.data_loaders['calibrate'] = dataloader
        self.data_loaders['validate'] = dataloader

    def inference(self, mode="validate"):
        def _compute_outputs(images: torch.Tensor,
                             targets: torch.Tensor):
            images = images.to(self.device)
            targets = targets.to(self.device)
            logits = self.model(images)
            return logits

        mode = "validate" if mode == "test" else mode
        print("Inference mode: {}".format(mode))
        data_loader = self.data_loaders[mode]
        self.model.eval()
        meter = Meter()

        with torch.no_grad():
            for data_batch in tqdm(data_loader):
                images, targets = data_batch['image'], data_batch['mask']

                logits = _compute_outputs(images, targets)

                meter.update(logits.detach().cpu(), targets.detach().cpu())

        dice_score, iou_score = meter.get_metrics()
        print("Dice: {:.4f}, IoU: {:.4f}".format(dice_score, iou_score))

    def onnx_exporter(self, onnx_path):
        random_input = torch.randn(self.input_size)
        if torch.cuda.is_available():
            random_input = random_input.cuda()
        replace_dict = {}
        for module in self.model.modules():
            if isinstance(module, nn.GroupNorm):
                replace_dict[module] = nn.BatchNorm3d(module.num_channels).to(self.device)
        self.replace_modules(replace_dict)
        torch.onnx.export(self, random_input, onnx_path, verbose=False, keep_initializers_as_inputs=True)
        model = onnx.load(onnx_path)
        model_simp, check = simplify(model)
        onnx.checker.check_model(model_simp)
        onnx.save(model_simp, onnx_path)

class BraTS2020(data.Dataset):
    def __init__(self, dataset_path: str, phase: str = "validation", is_resize: bool = False, seed: int = 55):
        self.dataset_path = dataset_path
        self.phase = phase
        self.transforms = Compose([], is_check_shapes=False)
        self.data_types = ['_flair.nii', '_t1.nii', '_t1ce.nii', '_t2.nii']
        self.is_resize = is_resize
        self.seed = seed
        self.seed_everything(self.seed)
        self.df = self.load_df(fold=0)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, 'Brats20ID']
        root_path = self.df.loc[self.df['Brats20ID'] == id_]['path'].values[0]
        # load all modalities
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(root_path, id_ + data_type)
            img = self.load_img(img_path)

            if self.is_resize:
                img = self.resize(img)

            img = self.normalize(img)
            images.append(img)
        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))

        if self.phase != "test":
            mask_path = os.path.join(root_path, id_ + "_seg.nii")
            mask = self.load_img(mask_path)

            if self.is_resize:
                mask = self.resize(mask)
                mask = np.clip(mask.astype(np.uint8), 0, 1).astype(np.float32)
                mask = np.clip(mask, 0, 1)
            mask = self.preprocess_mask_labels(mask)

            augmented = self.transforms(image=img.astype(np.float32),
                                        mask=mask.astype(np.float32))

            img = augmented['image']
            mask = augmented['mask']

            return {
                "Id": id_,
                "image": img,
                "mask": mask,
            }

        return {
            "Id": id_,
            "image": img,
        }

    def seed_everything(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def load_df(self, fold):
        survival_info_df = pd.read_csv(os.path.join(
            self.dataset_path, "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData", "survival_info.csv"))
        name_mapping_df = pd.read_csv(os.path.join(
            self.dataset_path, "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData", "name_mapping.csv"))

        name_mapping_df.rename(
            {'BraTS_2020_subject_ID': 'Brats20ID'}, axis=1, inplace=True)

        df = survival_info_df.merge(
            name_mapping_df,  on="Brats20ID", how="right")

        paths = []
        for _, row in df.iterrows():

            id_ = row['Brats20ID']
            phase = id_.split("_")[-2]

            if phase == 'Training':
                path = os.path.join(os.path.join(
                    self.dataset_path, "BraTS2020_TrainingData", "MICCAI_BraTS2020_TrainingData"), id_)
            else:
                path = os.path.join(os.path.join(
                    self.dataset_path, "BraTS2020_ValidationData", "MICCAI_BraTS2020_ValidationData"), id_)
            paths.append(path)

        df['path'] = paths

        train_data = df.loc[df['Age'].notnull()].reset_index(drop=True)
        train_data["Age_rank"] = train_data["Age"] // 10 * 10
        train_data = train_data.loc[train_data['Brats20ID']
                                    != 'BraTS20_Training_355'].reset_index(drop=True, )

        skf = StratifiedKFold(
            n_splits=7, random_state=self.seed, shuffle=True
        )
        for i, (train_index, val_index) in enumerate(
                skf.split(train_data, train_data["Age_rank"])
        ):
            train_data.loc[val_index, "fold"] = i

        match self.phase:
            case "validation":
                phase_df = train_data.loc[train_data['fold'] == fold].reset_index(
                    drop=True)
            case "train":
                phase_df = train_data.loc[train_data['fold'] != fold].reset_index(
                    drop=True)
            case "test":
                phase_df = df.loc[~df['Age'].notnull()].reset_index(drop=True)
            case _:
                raise NotImplementedError(
                    f"Phase {self.phase} not implemented")

        return phase_df

    def load_img(self, file_path):
        data = nib.load(file_path)
        data = np.asarray(data.dataobj)
        return data

    def normalize(self, data: np.ndarray):
        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)

    def resize(self, data: np.ndarray):
        data = resize(data, (78, 120, 120), preserve_range=True)
        return data

    def preprocess_mask_labels(self, mask: np.ndarray):

        mask_WT = mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1

        mask = np.stack([mask_WT, mask_TC, mask_ET])
        mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1))

        return mask


def dice_coef_metric(probabilities: torch.Tensor,
                     truth: torch.Tensor,
                     treshold: float = 0.5,
                     eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Dice score for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: dice score aka f1.
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert (predictions.shape == truth.shape)
    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = 2.0 * (truth_ * prediction).sum()
        union = truth_.sum() + prediction.sum()
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


def jaccard_coef_metric(probabilities: torch.Tensor,
                        truth: torch.Tensor,
                        treshold: float = 0.5,
                        eps: float = 1e-9) -> np.ndarray:
    """
    Calculate Jaccard index for data batch.
    Params:
        probobilities: model outputs after activation function.
        truth: truth values.
        threshold: threshold for probabilities.
        eps: additive to refine the estimate.
        Returns: jaccard score aka iou."
    """
    scores = []
    num = probabilities.shape[0]
    predictions = (probabilities >= treshold).float()
    assert (predictions.shape == truth.shape)

    for i in range(num):
        prediction = predictions[i]
        truth_ = truth[i]
        intersection = (prediction * truth_).sum()
        union = (prediction.sum() + truth_.sum()) - intersection + eps
        if truth_.sum() == 0 and prediction.sum() == 0:
            scores.append(1.0)
        else:
            scores.append((intersection + eps) / union)
    return np.mean(scores)


class Meter:
    '''factory for storing and updating iou and dice scores.'''

    def __init__(self, treshold: float = 0.5):
        self.threshold: float = treshold
        self.dice_scores: list = []
        self.iou_scores: list = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Takes: logits from output model and targets,
        calculates dice and iou scores, and stores them in lists.
        """
        probs = torch.sigmoid(logits)
        dice = dice_coef_metric(probs, targets, self.threshold)
        iou = jaccard_coef_metric(probs, targets, self.threshold)

        self.dice_scores.append(dice)
        self.iou_scores.append(iou)

    def get_metrics(self) -> np.ndarray:
        """
        Returns: the average of the accumulated dice and iou scores.
        """
        dice = np.mean(self.dice_scores)
        iou = np.mean(self.iou_scores)
        return dice, iou


class DoubleConv(nn.Module):
    """(Conv3D -> BN -> ReLU) * 2"""

    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv3d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm3d(out_channels),
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY //
                   2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3d(nn.Module):
    def __init__(self, in_channels, n_classes, n_channels):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.n_channels = n_channels

        self.conv = DoubleConv(in_channels, n_channels)
        self.enc1 = Down(n_channels, 2 * n_channels)
        self.enc2 = Down(2 * n_channels, 4 * n_channels)
        self.enc3 = Down(4 * n_channels, 8 * n_channels)
        self.enc4 = Down(8 * n_channels, 8 * n_channels)

        self.dec1 = Up(16 * n_channels, 4 * n_channels)
        self.dec2 = Up(8 * n_channels, 2 * n_channels)
        self.dec3 = Up(4 * n_channels, n_channels)
        self.dec4 = Up(2 * n_channels, n_channels)
        self.out = Out(n_channels, n_classes)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)

        mask = self.dec1(x5, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask
