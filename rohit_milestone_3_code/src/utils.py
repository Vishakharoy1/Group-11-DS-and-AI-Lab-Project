"""
src/utils.py
─────────────────────────────────────────────
Utility helpers: seed, logging, directory setup.
"""

import os
import random
import logging
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops (slight speed trade-off)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, log_file: str = None) -> logging.Logger:
    """Create a logger that writes to stdout and optionally a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


def make_dirs(*dirs: str) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_device(mixed_precision: bool = True):
    """
    Return the best available device and whether AMP is usable.

    Priority: CUDA (NVIDIA) → MPS (Apple Silicon) → CPU

    AMP notes:
      - CUDA: full AMP support (GradScaler + autocast)
      - MPS : AMP disabled — GradScaler not supported on Apple Silicon.
              Speed is recovered via larger batch sizes on unified memory.
      - CPU : AMP disabled
    """
    if torch.cuda.is_available():
        device  = torch.device("cuda")
        use_amp = mixed_precision
        print(f"[Device] CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device  = torch.device("mps")   # Apple Silicon GPU (M-series)
        use_amp = False                  # GradScaler unsupported on MPS
        print("[Device] Apple MPS GPU detected (M-series) — AMP disabled, using large batch size instead")
    else:
        device  = torch.device("cpu")
        use_amp = False
        print("[Device] No GPU found — running on CPU")
    return device, use_amp


def count_parameters(model: torch.nn.Module) -> dict:
    """Return total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
