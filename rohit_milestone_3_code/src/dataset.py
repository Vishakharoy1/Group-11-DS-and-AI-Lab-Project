"""
src/dataset.py
─────────────────────────────────────────────
Dataset loader for binary real/fake deepfake detection.

Expected folder structure (PyTorch ImageFolder compatible):
    dataset/
        train/
            real/   ← real face images
            fake/   ← AI-generated / manipulated faces
        val/
            real/
            fake/
        test/
            real/
            fake/

Classes: real=0, fake=1  (alphabetical ImageFolder order)
"""

import os
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF


# ── JPEG compression transform ──────────────────────────────────────────────
class RandomJPEGCompression:
    """Simulate JPEG compression as a forensic-aware augmentation."""

    def __init__(self, quality_min: int = 75, quality_max: int = 100):
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = np.random.randint(self.quality_min, self.quality_max + 1)
        # Encode and decode via numpy/cv2 to simulate compression
        arr = np.array(img)
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", arr_bgr, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return Image.fromarray(decoded_rgb)


# ── Transform builders ───────────────────────────────────────────────────────
def build_transforms(
    image_size: int,
    aug_cfg: dict,
    is_train: bool,
) -> transforms.Compose:
    """
    Build torchvision transform pipeline.

    For training: light forensic-safe augmentations.
    For val/test: only resize + normalize (no augmentation).
    """
    mean = aug_cfg["normalize_mean"]
    std = aug_cfg["normalize_std"]

    if is_train:
        transform_list = [
            transforms.Resize((image_size, image_size)),
        ]
        if aug_cfg.get("horizontal_flip", True):
            transform_list.append(transforms.RandomHorizontalFlip())
        if aug_cfg.get("brightness", 0) > 0 or aug_cfg.get("contrast", 0) > 0:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=aug_cfg.get("brightness", 0.2),
                    contrast=aug_cfg.get("contrast", 0.2),
                )
            )
        # JPEG compression (simulate real-world distribution)
        transform_list.append(
            RandomJPEGCompression(
                quality_min=aug_cfg.get("jpeg_quality_min", 75),
                quality_max=aug_cfg.get("jpeg_quality_max", 100),
            )
        )
        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    else:
        transform_list = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

    return transforms.Compose(transform_list)


# ── DataLoader builder ───────────────────────────────────────────────────────
def get_dataloaders(
    data_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    aug_cfg: dict,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Build train / val / test DataLoaders using ImageFolder.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    train_transform = build_transforms(image_size, aug_cfg, is_train=True)
    eval_transform  = build_transforms(image_size, aug_cfg, is_train=False)

    splits = {
        "train": (os.path.join(data_dir, "train"), train_transform),
        "val":   (os.path.join(data_dir, "val"),   eval_transform),
        "test":  (os.path.join(data_dir, "test"),  eval_transform),
    }

    loaders = {}
    class_names = None

    for split, (path, tfm) in splits.items():
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"[Dataset] '{path}' not found.\n"
                f"Make sure your data follows the structure:\n"
                f"  {data_dir}/train/real/, {data_dir}/train/fake/, etc."
            )
        dataset = datasets.ImageFolder(root=path, transform=tfm)
        if class_names is None:
            class_names = dataset.classes   # ['fake', 'real'] or ['real', 'fake']

        shuffle = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=split == "train",  # avoid incomplete last batch during training
        )

    return loaders["train"], loaders["val"], loaders["test"], class_names


def get_class_weights(train_loader: DataLoader, num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights to handle class imbalance.
    Returns a tensor of shape [num_classes].
    """
    counts = torch.zeros(num_classes)
    for _, labels in train_loader:
        for c in range(num_classes):
            counts[c] += (labels == c).sum()
    weights = counts.sum() / (num_classes * counts)
    return weights
