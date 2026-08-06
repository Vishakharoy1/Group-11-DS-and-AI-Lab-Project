"""
src/train.py
─────────────────────────────────────────────
Training engine for Phase 1 ConvNeXt V2 baseline.

Features:
  - Mixed-precision training (torch.cuda.amp)
  - Per-epoch metrics: loss, accuracy, AUC, F1
  - Best checkpoint saving (by val AUC)
  - TensorBoard logging
  - Gradient clipping for stability
"""

import os
import time
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
)


class Trainer:
    """
    Handles one complete training stage (fixed freeze config + LR).
    Instantiate a new Trainer for each stage, or reuse and call .train().
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        device: torch.device,
        checkpoint_dir: str,
        log_dir: str,
        use_amp: bool = True,
        grad_clip: float = 1.0,
        stage_name: str = "stage",
    ):
        self.model         = model
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.optimizer     = optimizer
        self.scheduler     = scheduler
        self.criterion     = criterion
        self.device        = device
        self.checkpoint_dir = checkpoint_dir
        self.use_amp       = use_amp
        self.grad_clip     = grad_clip
        self.stage_name    = stage_name

        self.scaler = GradScaler(enabled=use_amp)
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, stage_name))

        self.best_auc   = 0.0
        self.best_epoch = 0
        self.global_step = 0

    # ── One epoch ─────────────────────────────────────────────────────────────
    def _run_epoch(
        self, loader: DataLoader, training: bool
    ) -> Tuple[float, dict]:
        """Run one pass over the loader. Returns (loss, metrics_dict)."""
        self.model.train(training)
        total_loss = 0.0
        all_labels, all_probs, all_preds = [], [], []

        desc = "Train" if training else "Val  "
        pbar = tqdm(loader, desc=desc, leave=False)

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.set_grad_enabled(training):
                with autocast(enabled=self.use_amp):
                    logits = self.model(images)
                    loss   = self.criterion(logits, labels)

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.global_step += 1

            probs = torch.softmax(logits.detach(), dim=1)[:, 1]  # prob of "fake"
            preds = logits.detach().argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            total_loss += loss.item() * images.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        n = len(loader.dataset)
        avg_loss = total_loss / n

        metrics = self._compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
        )
        return avg_loss, metrics

    # ── Metrics ───────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_metrics(
        labels: np.ndarray, preds: np.ndarray, probs: np.ndarray
    ) -> dict:
        auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
        return {
            "accuracy":  accuracy_score(labels, preds),
            "f1":        f1_score(labels, preds, zero_division=0),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall":    recall_score(labels, preds, zero_division=0),
            "auc":       auc,
        }

    # ── Logging ───────────────────────────────────────────────────────────────
    def _log(self, prefix: str, loss: float, metrics: dict, epoch: int) -> None:
        self.writer.add_scalar(f"{prefix}/loss",      loss,               epoch)
        self.writer.add_scalar(f"{prefix}/accuracy",  metrics["accuracy"], epoch)
        self.writer.add_scalar(f"{prefix}/f1",        metrics["f1"],       epoch)
        self.writer.add_scalar(f"{prefix}/auc",       metrics["auc"],      epoch)
        self.writer.add_scalar(
            "lr/head",
            self.optimizer.param_groups[-1]["lr"], epoch
        )

    # ── Checkpoint ────────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, val_auc: float, is_best: bool) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        ckpt = {
            "epoch":      epoch,
            "stage":      self.stage_name,
            "val_auc":    val_auc,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        # Always save latest
        path_latest = os.path.join(self.checkpoint_dir, f"{self.stage_name}_latest.pth")
        torch.save(ckpt, path_latest)

        if is_best:
            path_best = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(ckpt, path_best)

    # ── Main train loop ───────────────────────────────────────────────────────
    def train(self, epochs: int, logger=None) -> dict:
        """
        Run training for `epochs` epochs.
        Returns dict with best val metrics.
        """
        best_metrics = {}

        for epoch in range(1, epochs + 1):
            t0 = time.time()

            # ── Train ──────────────────────────────────────────────────────
            train_loss, train_metrics = self._run_epoch(self.train_loader, training=True)

            # ── Validate ───────────────────────────────────────────────────
            val_loss, val_metrics = self._run_epoch(self.val_loader, training=False)

            # ── Scheduler step ─────────────────────────────────────────────
            self.scheduler.step()

            # ── Log ────────────────────────────────────────────────────────
            self._log("train", train_loss, train_metrics, epoch)
            self._log("val",   val_loss,   val_metrics,   epoch)

            elapsed = time.time() - t0
            msg = (
                f"[{self.stage_name}] Epoch {epoch:3d}/{epochs} "
                f"| T-loss {train_loss:.4f} T-AUC {train_metrics['auc']:.4f} "
                f"| V-loss {val_loss:.4f} V-AUC {val_metrics['auc']:.4f} "
                f"V-F1 {val_metrics['f1']:.4f} "
                f"| {elapsed:.1f}s"
            )
            if logger:
                logger.info(msg)
            else:
                print(msg)

            # ── Save checkpoint ────────────────────────────────────────────
            is_best = val_metrics["auc"] > self.best_auc
            if is_best:
                self.best_auc   = val_metrics["auc"]
                self.best_epoch = epoch
                best_metrics    = val_metrics
            self._save_checkpoint(epoch, val_metrics["auc"], is_best)

        self.writer.close()

        summary = (
            f"[{self.stage_name}] Best val AUC: {self.best_auc:.4f} "
            f"at epoch {self.best_epoch}"
        )
        if logger:
            logger.info(summary)
        else:
            print(summary)

        return best_metrics
