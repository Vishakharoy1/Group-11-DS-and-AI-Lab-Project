"""
run_train.py
─────────────────────────────────────────────
Single entry point for Phase 1 training.

Usage:
    python run_train.py
    python run_train.py --config configs/config.yaml

This runs all 3 fine-tuning stages sequentially:
    Stage 1: Head only      (backbone frozen)        — fast
    Stage 2: Head + last stage                       — moderate
    Stage 3: Full model     (everything unfrozen)    — full fine-tune

Best checkpoint is saved to: checkpoints/best_model.pth
"""

import os
import sys
import argparse
import yaml

import torch
import torch.nn as nn

# ── Ensure src/ is importable ─────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.model   import build_model
from src.dataset import get_dataloaders
from src.train   import Trainer
from src.utils   import set_seed, get_device, get_logger, make_dirs, count_parameters


def build_optimizer(model, stage_cfg: dict) -> torch.optim.Optimizer:
    """Build AdamW with separate LRs for backbone and head."""
    param_groups = model.get_optimizer_param_groups(
        lr_head=stage_cfg["lr_head"],
        lr_backbone=stage_cfg["lr_backbone"],
    )
    return torch.optim.AdamW(param_groups, weight_decay=0.05)


def build_scheduler(optimizer, epochs: int, cfg: dict):
    """CosineAnnealingLR over the stage's epochs."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7
    )


def main(cfg_path: str):
    # ── Load config ───────────────────────────────────────────────────────────
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # ── Setup ─────────────────────────────────────────────────────────────────
    set_seed(cfg["seed"])
    device, use_amp = get_device(cfg["mixed_precision"])

    checkpoint_dir = os.path.join(ROOT, cfg["checkpoint_dir"])
    log_dir        = os.path.join(ROOT, cfg["log_dir"])
    output_dir     = os.path.join(ROOT, cfg["output_dir"])
    data_dir       = os.path.join(ROOT, cfg["data_dir"])

    make_dirs(checkpoint_dir, log_dir, output_dir)

    logger = get_logger(
        "run_train",
        log_file=os.path.join(log_dir, "run_train.log"),
    )
    logger.info(f"Device: {device} | AMP: {use_amp}")
    logger.info(f"Config: {cfg_path}")

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=data_dir,
        image_size=cfg["image_size"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        aug_cfg=cfg["augmentation"],
        pin_memory=cfg["pin_memory"],
    )
    logger.info(f"Classes: {class_names}")
    logger.info(
        f"Train: {len(train_loader.dataset)} | "
        f"Val: {len(val_loader.dataset)} | "
        f"Test: {len(test_loader.dataset)}"
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_id = cfg.get('model_name', cfg.get('name', 'dual_stream'))
    logger.info(f"Building model: {model_id}")
    model = build_model(cfg).to(device)

    param_info = count_parameters(model)
    logger.info(
        f"Parameters: total={param_info['total']:,} | "
        f"trainable={param_info['trainable']:,}"
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg.get("label_smoothing", 0.0)
    )

    # ── Best AUC tracker across all stages ────────────────────────────────────
    global_best_auc = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # Staged fine-tuning
    # ══════════════════════════════════════════════════════════════════════════
    for stage_cfg in cfg["stages"]:
        stage_name   = stage_cfg["name"]
        stage_epochs = stage_cfg["epochs"]
        freeze_mode  = stage_cfg["freeze"]

        logger.info("=" * 60)
        logger.info(f"  Starting: {stage_name}  (freeze={freeze_mode}, epochs={stage_epochs})")
        logger.info("=" * 60)

        # ── Apply freeze strategy ─────────────────────────────────────────────
        if freeze_mode == "all":
            model.freeze_backbone()
        elif freeze_mode == "partial":
            model.unfreeze_last_stage()
        elif freeze_mode == "none":
            model.unfreeze_all()
        else:
            raise ValueError(f"Unknown freeze mode: {freeze_mode}")

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Trainable params this stage: {trainable:,}")

        # ── Optimizer + Scheduler ────────────────────────────────────────────
        optimizer = build_optimizer(model, stage_cfg)
        scheduler = build_scheduler(optimizer, stage_epochs, cfg)

        # ── Trainer ──────────────────────────────────────────────────────────
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            use_amp=use_amp,
            grad_clip=1.0,
            stage_name=stage_name,
        )

        # Override best_auc so the global best checkpoint is tracked
        trainer.best_auc = global_best_auc
        best_metrics = trainer.train(epochs=stage_epochs, logger=logger)
        global_best_auc = max(global_best_auc, trainer.best_auc)

    # ══════════════════════════════════════════════════════════════════════════
    # Final evaluation on test set
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 60)
    logger.info("  Running final evaluation on TEST set...")
    logger.info("=" * 60)

    from src.evaluate import evaluate
    metrics = evaluate(
        cfg_path=cfg_path,
        checkpoint_path=os.path.join(checkpoint_dir, "best_model.pth"),
    )

    logger.info("─" * 60)
    logger.info("  FINAL TEST RESULTS")
    logger.info("─" * 60)
    for k, v in metrics.items():
        logger.info(f"  {k:12s}: {v:.4f}")
    logger.info("─" * 60)
    logger.info(f"  Outputs saved to: {output_dir}")
    logger.info("  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 1 — ConvNeXt V2 Deepfake Detection Baseline"
    )
    parser.add_argument(
        "--config",
        default=os.path.join(ROOT, "configs", "config.yaml"),
        help="Path to config YAML (default: configs/config.yaml)",
    )
    args = parser.parse_args()
    main(cfg_path=args.config)
