"""
Centralized configuration for the Face Liveness Detection system.

Handles automatic environment detection (local MacBook vs Google Colab),
device selection (CUDA → MPS → CPU), and path resolution.

Usage:
    cfg = load_config("experiments/configs/resnet18_baseline.yaml")
    # or
    cfg = Config()  # uses auto-detected defaults
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Constants shared across the codebase
# ---------------------------------------------------------------------------

CLASS_NAMES = {0: "spoof", 1: "live"}
CLASS_IDX = {"spoof": 0, "live": 1}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Environment / device detection
# ---------------------------------------------------------------------------

def auto_detect_env() -> tuple[str, str]:
    """Return (env_name, device_string).

    env_name : "colab" | "local"
    device   : "cuda"  | "mps" | "cpu"
    """
    import torch

    is_colab = (
        os.path.exists("/content")
        and (
            "COLAB_GPU" in os.environ
            or "COLAB_RELEASE_TAG" in os.environ
            or "COLAB_BACKEND_VERSION" in os.environ
        )
    )
    env = "colab" if is_colab else "local"

    if torch.cuda.is_available():
        device = "cuda"
    elif (
        not is_colab
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        device = "mps"
    else:
        device = "cpu"

    return env, device


def get_repo_root() -> Path:
    """Return the Assignment_3 directory regardless of where the script is run."""
    return Path(__file__).resolve().parent.parent


def get_data_root(env: str) -> Path:
    if env == "colab":
        return Path("/content/drive/MyDrive/liveness/data")
    return get_repo_root() / "data"


def get_experiment_root(env: str) -> Path:
    if env == "colab":
        return Path("/content/drive/MyDrive/liveness/experiments")
    return get_repo_root() / "experiments"


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class Config:
    # --- Environment (auto-detected; can be overridden) ---
    env: str = "local"
    device: str = "cpu"

    # --- Paths (set via post_init after env is known) ---
    data_root: str = ""
    experiment_root: str = ""
    run_name: str = "resnet18_run1"

    # --- Model ---
    model_name: str = "resnet18"    # resnet18 | resnet34 | resnet50 | yolo11n-cls | yolo11s-cls
    num_classes: int = 2
    pretrained: bool = True

    # --- Training ---
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 5               # early-stopping epochs without improvement

    # --- Data ---
    image_size: int = 224
    subset_fraction: float = 1.0    # overridden to 0.1 locally by post_init
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    num_workers: int = 4            # overridden to 0 on MPS by post_init
    seed: int = 42

    # --- Augmentation flags ---
    use_horizontal_flip: bool = True
    use_brightness_jitter: bool = True
    use_blur: bool = True

    # --- Face detector ---
    face_detector: str = "mediapipe"  # haar | mediapipe | mtcnn
    face_detector_confidence: float = 0.7
    face_padding_frac: float = 0.1

    # --- ONNX ---
    onnx_opset: int = 17

    def __post_init__(self):
        # Fill paths from env if not already set
        if not self.data_root:
            self.data_root = str(get_data_root(self.env))
        if not self.experiment_root:
            self.experiment_root = str(get_experiment_root(self.env))

        # Sensible local defaults to keep iteration fast on CPU/MPS
        if self.env == "local":
            if self.subset_fraction == 1.0:
                self.subset_fraction = 0.1
            if self.epochs == 30:
                self.epochs = 5
            if self.batch_size == 32:
                self.batch_size = 8

        # MPS does not support multi-process DataLoader well in all torch versions
        if self.device == "mps":
            self.num_workers = 0

    # --- Convenience helpers ---

    @property
    def processed_dir(self) -> Path:
        return Path(self.data_root) / "processed"

    @property
    def splits_dir(self) -> Path:
        return Path(self.data_root) / "splits"

    @property
    def run_dir(self) -> Path:
        return Path(self.experiment_root) / "results" / self.run_name

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Factory / loader
# ---------------------------------------------------------------------------

def _auto_config() -> Config:
    """Create a Config with auto-detected env and device."""
    env, device = auto_detect_env()
    return Config(env=env, device=device)


def load_config(yaml_path: str | Path, overrides: Optional[dict] = None) -> Config:
    """Load a YAML experiment config, merge with auto-detected environment defaults.

    The YAML file may contain any subset of Config fields.
    CLI overrides (from argparse) are applied last.
    """
    cfg = _auto_config()

    path = Path(yaml_path)
    if path.exists():
        with open(path) as f:
            yaml_data = yaml.safe_load(f) or {}
        for key, value in yaml_data.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    else:
        raise FileNotFoundError(f"Config file not found: {path}")

    if overrides:
        for key, value in overrides.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)

    # Re-run post_init logic after applying overrides
    # (paths may have changed if data_root was specified in YAML)
    if not Path(cfg.data_root).is_absolute():
        cfg.data_root = str(get_repo_root() / cfg.data_root)
    if not Path(cfg.experiment_root).is_absolute():
        cfg.experiment_root = str(get_repo_root() / cfg.experiment_root)

    return cfg


def save_config_snapshot(cfg: Config, output_dir: Path | str) -> None:
    """Dump the final merged config to YAML for reproducibility."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "config_snapshot.yaml", "w") as f:
        yaml.dump(cfg.to_dict(), f, default_flow_style=False, sort_keys=True)


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    env, device = auto_detect_env()
    print(f"Environment : {env}")
    print(f"Device      : {device}")
    print(f"Repo root   : {get_repo_root()}")
    cfg = _auto_config()
    print(f"\nDefault config:")
    for k, v in cfg.to_dict().items():
        print(f"  {k}: {v}")
