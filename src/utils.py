"""
Shared utilities for the Face Liveness Detection system.

Provides: seed fixing, device selection, run directory setup,
dual-output logging, FPS measurement, and parameter counting.
"""

from __future__ import annotations

import logging
import random
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(prefer_mps: bool = True) -> torch.device:
    """Return the best available device: CUDA → MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Run directory
# ---------------------------------------------------------------------------

def setup_run_dir(experiments_root: str | Path, run_name: str) -> Path:
    """Create and return the run directory tree.

    Creates:
        <experiments_root>/results/<run_name>/
        <experiments_root>/results/<run_name>/weights/
    """
    run_dir = Path(experiments_root) / "results" / run_name
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(name: str, log_file: Optional[str | Path] = None) -> logging.Logger:
    """Create a logger that writes to both console and (optionally) a file."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s — %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# FPS / throughput measurement
# ---------------------------------------------------------------------------

def compute_fps(
    model: nn.Module,
    device: torch.device,
    image_size: int = 224,
    batch_size: int = 1,
    n_iters: int = 200,
    warmup: int = 20,
) -> dict:
    """Measure model throughput (frames per second) and latency.

    Args:
        model      : PyTorch model in eval mode
        device     : target device
        image_size : spatial size of input frames
        batch_size : inference batch size (use 1 for real-time simulation)
        n_iters    : number of timed iterations
        warmup     : number of warmup iterations (not counted)

    Returns:
        dict with keys: fps, mean_ms, min_ms, max_ms
    """
    model.eval()
    dummy = torch.randn(batch_size, 3, image_size, image_size, device=device)

    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        latencies: list[float] = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

    mean_ms = float(np.mean(latencies))
    return {
        "fps": round(1000.0 / mean_ms * batch_size, 2),
        "mean_ms": round(mean_ms, 3),
        "min_ms": round(float(np.min(latencies)), 3),
        "max_ms": round(float(np.max(latencies)), 3),
    }


def save_fps_report(fps_data: dict, run_dir: Path | str, model_name: str) -> None:
    """Write FPS report to text file in the run directory."""
    out = Path(run_dir) / "fps_report.txt"
    lines = [
        f"Model        : {model_name}",
        f"FPS          : {fps_data['fps']}",
        f"Mean latency : {fps_data['mean_ms']} ms",
        f"Min latency  : {fps_data['min_ms']} ms",
        f"Max latency  : {fps_data['max_ms']} ms",
    ]
    out.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Model stats
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Rolling FPS counter for webcam
# ---------------------------------------------------------------------------

class RollingFPS:
    """Compute a rolling average FPS over the last N frames."""

    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Record the current timestamp and return current FPS estimate."""
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0
