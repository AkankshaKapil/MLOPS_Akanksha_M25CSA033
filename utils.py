import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

# -------------------------------
# Dataset with optional augmentation
# -------------------------------
class CityScapesDataset(Dataset):
    def __init__(self, image_dir, mask_dir, target_size=(128, 128), is_train=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.target_size = target_size   # (height, width)
        self.is_train = is_train

        # Simple augmentation for training (only horizontal flip – safe)
        if is_train:
            self.flip = T.RandomHorizontalFlip(p=0.5)
        else:
            self.flip = None

        # Normalization for RGB images (ImageNet stats)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        # Read image (RGB) and mask (grayscale)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize both to target_size
        image = cv2.resize(image, self.target_size[::-1], interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, self.target_size[::-1], interpolation=cv2.INTER_NEAREST)

        # Convert to PIL for torchvision transforms
        image = T.ToPILImage()(image)
        mask = T.ToPILImage()(mask)

        # Apply random horizontal flip (same to both image and mask)
        if self.flip:
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.flip(image)
            torch.manual_seed(seed)
            mask = self.flip(mask)

        # Convert to tensor and normalize image
        image = T.ToTensor()(image)
        image = self.normalize(image)
        mask = torch.tensor(np.array(mask), dtype=torch.long)

        return image, mask


def get_dataloaders(data_dir=".", batch_size=8, target_size=(128, 128)):
    """
    data_dir: directory containing CameraRGB/ and CameraMask/
    """
    image_dir = os.path.join(data_dir, "CameraRGB")
    mask_dir = os.path.join(data_dir, "CameraMask")

    # Create full dataset (train split with augmentation, test without)
    full_train = CityScapesDataset(image_dir, mask_dir, target_size=target_size, is_train=True)
    full_test  = CityScapesDataset(image_dir, mask_dir, target_size=target_size, is_train=False)

    # 80-20 split with seed 42
    train_size = int(0.8 * len(full_train))
    test_size = len(full_test) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_train, [train_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader


# -------------------------------
# Metrics
# -------------------------------
def iou_score(pred, target, num_classes=23):
    pred = pred.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append((intersection / union).item())
    return np.nanmean(ious)

def dice_score(pred, target, num_classes=23):
    pred = pred.argmax(dim=1)
    dices = []
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        target_mask = (target == cls)
        intersection = (pred_mask & target_mask).sum().float()
        dice = (2. * intersection) / (pred_mask.sum() + target_mask.sum() + 1e-6)
        dices.append(dice.item())
    return np.mean(dices)


# -------------------------------
# Losses: Focal Loss + Dice Loss
# -------------------------------
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_soft = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1]).permute(0,3,1,2).float()
        intersection = (pred_soft * target_one_hot).sum(dim=(2,3))
        union = pred_soft.sum(dim=(2,3)) + target_one_hot.sum(dim=(2,3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

class CombinedLoss(nn.Module):
    def __init__(self, weight_focal=0.5, weight_dice=0.5):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.w_f = weight_focal
        self.w_d = weight_dice

    def forward(self, pred, target):
        return self.w_f * self.focal(pred, target) + self.w_d * self.dice(pred, target)
