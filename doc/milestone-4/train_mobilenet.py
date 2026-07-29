"""
rohit/mobilenetv3large/train_mobilenet.py
─────────────────────────────────────────────────────────────────
Standalone MobileNetV3-Large Training & Evaluation Pipeline.
Supports dynamic hyperparameter overrides for optimization sweeps.
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
)

# ── Import dataset utilities if available ─────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import build_transforms, get_dataloaders
from src.utils import set_seed, make_dirs


def get_device() -> torch.device:
    """Returns PyTorch device preferring Apple Silicon MPS -> CUDA -> CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def build_mobilenet(
    model_name: str = "mobilenetv3_large_100",
    pretrained: bool = True,
    num_classes: int = 2,
    dropout: float = 0.2,
) -> nn.Module:
    """Instantiate MobileNetV3-Large with custom dropout and classification head."""
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
    return model


def build_optimizer(
    model: nn.Module,
    opt_type: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build Adam or AdamW optimizer."""
    opt_type = opt_type.lower()
    if opt_type == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif opt_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_type}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_type: str,
    epochs: int,
):
    """Build CosineAnnealingLR or ReduceLROnPlateau scheduler."""
    sched_type = sched_type.lower()
    if sched_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-7
        )
    elif sched_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )
    else:
        raise ValueError(f"Unsupported scheduler: {sched_type}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_train: bool,
    epoch: int = 1,
    epochs: int = 1,
) -> Tuple[float, Dict[str, float]]:
    """Run single epoch of training or validation/test."""
    model.train(is_train)
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    desc = f"Epoch {epoch:2d}/{epochs:2d} [{'Train' if is_train else 'Val  '}]"
    pbar = tqdm(loader, desc=desc, leave=False, file=sys.stdout)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        probs = torch.softmax(logits.detach(), dim=1)[:, 1]
        preds = logits.detach().argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        total_loss += loss.item() * images.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")
        sys.stdout.flush()

    n = len(loader.dataset)
    avg_loss = total_loss / n

    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    probs_arr = np.array(all_probs)

    auc = roc_auc_score(labels_arr, probs_arr) if len(np.unique(labels_arr)) > 1 else 0.0
    metrics = {
        "acc": accuracy_score(labels_arr, preds_arr),
        "f1": f1_score(labels_arr, preds_arr, zero_division=0),
        "precision": precision_score(labels_arr, preds_arr, zero_division=0),
        "recall": recall_score(labels_arr, preds_arr, zero_division=0),
        "auc": auc,
    }
    return avg_loss, metrics


def train_and_evaluate(
    hparams: Dict[str, Any],
    exp_name: str = "baseline",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train MobileNetV3-Large with given hyperparameters and evaluate on Test set.
    """
    set_seed(hparams.get("seed", 42))
    device = get_device()

    data_dir = os.path.join(ROOT, hparams.get("data_dir", "dataset"))
    image_size = hparams.get("image_size", 224)
    batch_size = hparams.get("batch_size", 64)
    num_workers = hparams.get("num_workers", 8)
    epochs = hparams.get("epochs", 5)

    lr = hparams.get("learning_rate", 1e-3)
    weight_decay = hparams.get("weight_decay", 0.01)
    dropout = hparams.get("dropout", 0.2)
    opt_type = hparams.get("optimizer", "adamw")
    sched_type = hparams.get("scheduler", "cosine")
    label_smoothing = hparams.get("label_smoothing", 0.0)

    ckpt_dir = os.path.join(ROOT, hparams.get("checkpoint_dir", "mobilenetv3large/checkpoints"))
    make_dirs(ckpt_dir)
    save_ckpt_path = os.path.join(ckpt_dir, f"{exp_name}_best.pth")

    if verbose:
        print(f"\n========================================================", flush=True)
        print(f" Experiment: {exp_name}", flush=True)
        print(f" Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}", flush=True)
        print(f" LR: {lr} | WD: {weight_decay} | Dropout: {dropout}", flush=True)
        print(f" Opt: {opt_type} | Sched: {sched_type} | Label Smooth: {label_smoothing}", flush=True)
        print(f"========================================================", flush=True)

    # Dataloaders
    aug_cfg = hparams.get("augmentation", {
        "horizontal_flip": True,
        "brightness": 0.2,
        "contrast": 0.2,
        "jpeg_quality_min": 75,
        "jpeg_quality_max": 100,
        "normalize_mean": [0.485, 0.456, 0.406],
        "normalize_std": [0.229, 0.224, 0.225],
    })

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        aug_cfg=aug_cfg,
        pin_memory=False,
    )

    # Model
    model = build_mobilenet(
        model_name=hparams.get("model_name", "mobilenetv3_large_100"),
        pretrained=hparams.get("pretrained", True),
        num_classes=hparams.get("num_classes", 2),
        dropout=dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = build_optimizer(model, opt_type, lr, weight_decay)
    scheduler = build_scheduler(optimizer, sched_type, epochs)

    best_val_auc = 0.0
    best_val_metrics = {}
    total_train_time = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True, epoch=epoch, epochs=epochs
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, device, is_train=False, epoch=epoch, epochs=epochs
        )

        if sched_type == "plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        elapsed = time.time() - t0
        total_train_time += elapsed

        if verbose:
            print(
                f"\nEpoch {epoch:2d}/{epochs:2d} ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} Acc: {train_metrics['acc']:.4f} AUC: {train_metrics['auc']:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_metrics['acc']:.4f} AUC: {val_metrics['auc']:.4f}",
                flush=True
            )

        if val_metrics["auc"] >= best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_val_metrics = {
                "val_loss": val_loss,
                "val_acc": val_metrics["acc"],
                "val_f1": val_metrics["f1"],
                "val_auc": val_metrics["auc"],
            }
            torch.save(model.state_dict(), save_ckpt_path)

    # Final evaluation on Test Set using best saved checkpoint
    if os.path.exists(save_ckpt_path):
        model.load_state_dict(torch.load(save_ckpt_path, map_location=device))

    test_loss, test_metrics = run_epoch(
        model, test_loader, criterion, optimizer, device, is_train=False
    )

    if verbose:
        print(
            f"\n--> TEST RESULTS ({exp_name}) | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_metrics['acc'] * 100:.2f}% | "
            f"Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | "
            f"F1: {test_metrics['f1']:.4f} | AUC: {test_metrics['auc']:.4f}"
        )

    results = {
        "exp_name": exp_name,
        "hparams": hparams,
        "val_loss": best_val_metrics.get("val_loss", 0.0),
        "val_acc": best_val_metrics.get("val_acc", 0.0),
        "val_f1": best_val_metrics.get("val_f1", 0.0),
        "val_auc": best_val_metrics.get("val_auc", 0.0),
        "test_loss": test_loss,
        "test_acc": test_metrics["acc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_auc": test_metrics["auc"],
        "total_time_sec": total_train_time,
    }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Large")
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "mobilenetv3large", "config.yaml"),
        help="Path to config file",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_and_evaluate(cfg, exp_name="baseline_test", verbose=True)
