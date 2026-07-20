"""
src/evaluate.py
─────────────────────────────────────────────
Evaluation script for Phase 1 baseline.

Loads the best checkpoint and evaluates on the test set.
Outputs:
  - Console: Accuracy, Precision, Recall, F1, ROC-AUC
  - outputs/roc_curve.png
  - outputs/confusion_matrix.png
  - outputs/classification_report.txt
"""

import os
import sys
import argparse

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix, classification_report,
    roc_curve, auc,
)
from tqdm import tqdm
from torch.cuda.amp import autocast

# ── Local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model   import build_model
from src.dataset import get_dataloaders
from src.utils   import get_device, get_logger, make_dirs


# ── Inference ────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device, use_amp):
    """Run model over a DataLoader. Returns arrays of labels, probs, preds."""
    model.eval()
    all_labels, all_probs, all_preds = [], [], []

    for images, labels in tqdm(loader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)

        probs = torch.softmax(logits, dim=1)[:, 1]   # prob of class=1 (fake)
        preds = logits.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_probs),
        np.array(all_preds),
    )


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_roc_curve(labels, probs, output_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#6C63FF", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve — ConvNeXt V2 Baseline", fontsize=14)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return roc_auc


def plot_confusion_matrix(labels, preds, class_names, output_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Actual", fontsize=13)
    ax.set_title("Confusion Matrix — ConvNeXt V2 Baseline", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def evaluate(cfg_path: str, checkpoint_path: str, split: str = "test"):
    # Load config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir  = os.path.join(base_dir, cfg["output_dir"])
    make_dirs(output_dir)

    logger = get_logger("evaluate",
                        os.path.join(output_dir, "evaluate.log"))
    device, use_amp = get_device(cfg["mixed_precision"])
    logger.info(f"Device: {device} | AMP: {use_amp}")

    # ── Data ─────────────────────────────────────────────────────────────────
    data_dir = os.path.join(base_dir, cfg["data_dir"])
    _, _, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        aug_cfg=cfg["augmentation"],
        pin_memory=cfg["pin_memory"],
    )
    logger.info(f"Classes: {class_names}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt['epoch']})")

    # ── Inference ─────────────────────────────────────────────────────────────
    labels, probs, preds = run_inference(model, test_loader, device, use_amp)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    rauc = roc_auc_score(labels, probs)

    logger.info("=" * 55)
    logger.info("           TEST SET RESULTS — Phase 1 Baseline")
    logger.info("=" * 55)
    logger.info(f"  Accuracy : {acc:.4f}")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall   : {rec:.4f}")
    logger.info(f"  F1-score : {f1:.4f}")
    logger.info(f"  ROC-AUC  : {rauc:.4f}")
    logger.info("=" * 55)

    # Classification report
    report = classification_report(labels, preds, target_names=class_names)
    logger.info("\n" + report)
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # ── Plots ─────────────────────────────────────────────────────────────────
    roc_path = os.path.join(output_dir, "roc_curve.png")
    cm_path  = os.path.join(output_dir, "confusion_matrix.png")

    plot_roc_curve(labels, probs, roc_path)
    plot_confusion_matrix(labels, preds, class_names, cm_path)

    logger.info(f"ROC curve saved    → {roc_path}")
    logger.info(f"Confusion matrix   → {cm_path}")

    return {"accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "auc": rauc}


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/best_model.pth",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--split", default="test", choices=["val", "test"],
        help="Which split to evaluate on"
    )
    args = parser.parse_args()

    # Resolve paths relative to rohit/
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    evaluate(
        cfg_path=os.path.join(root, args.config),
        checkpoint_path=os.path.join(root, args.checkpoint),
        split=args.split,
    )
