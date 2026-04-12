"""
Main training entry point for the Face Liveness Detection system.

Dispatches to either the ResNet training pipeline (src/trainer.py) or the
YOLO training pipeline (src/models/yolo_classifier.py) based on the model_name
in the experiment config.

Usage:
    # Full run (Colab):
    python train.py --config experiments/configs/resnet18_baseline.yaml

    # Quick local smoke test (10% data, 2 epochs):
    python train.py --config experiments/configs/resnet18_baseline.yaml \
        --subset 0.05 --epochs 2

    # Resume from checkpoint:
    python train.py --config experiments/configs/resnet18_baseline.yaml \
        --resume experiments/results/resnet18_run1/weights/last.pt

    # YOLO model:
    python train.py --config experiments/configs/yolo11n_cls.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from src.config import load_config, Config
from src.utils import set_seed, get_logger

log = get_logger("train")

YOLO_MODELS = {"yolo11n-cls", "yolo11s-cls"}


def parse_args():
    p = argparse.ArgumentParser(description="Train liveness detector")
    p.add_argument(
        "--config", type=Path, required=True,
        help="Path to YAML experiment config"
    )
    p.add_argument("--run-name", type=str, default=None, help="Override run_name")
    p.add_argument("--subset",   type=float, default=None,
                   help="Override subset_fraction (e.g. 0.05 for quick test)")
    p.add_argument("--epochs",   type=int,   default=None, help="Override epochs")
    p.add_argument("--resume",   type=Path,  default=None,
                   help="Path to checkpoint to resume from")
    return p.parse_args()


def train_resnet(config: Config, resume_path: Path | None) -> None:
    """Full ResNet training pipeline."""
    from src.dataset import build_dataloaders
    from src.models.resnet_classifier import build_resnet
    from src.trainer import Trainer
    from src.evaluator import Evaluator

    log.info(f"Model        : {config.model_name}")
    log.info(f"Device       : {config.device}")
    log.info(f"Subset frac  : {config.subset_fraction}")
    log.info(f"Data root    : {config.data_root}")
    log.info(f"Splits dir   : {config.splits_dir}")

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(config)
    log.info(
        f"Datasets — train:{len(train_loader.dataset)} | "
        f"val:{len(val_loader.dataset)} | "
        f"test:{len(test_loader.dataset)}"
    )

    # Model
    model = build_resnet(config)
    log.info(
        f"Parameters   : {model.num_trainable_params():,} trainable"
    )

    # Trainer
    trainer = Trainer(model, config, train_loader, val_loader)

    if resume_path is not None:
        trainer.load_checkpoint(resume_path)

    history = trainer.train()

    # Evaluate best checkpoint on test set
    log.info("Loading best checkpoint for test evaluation...")
    best_ckpt = config.run_dir / "weights" / "best.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=config.device)
        model.load_state_dict(ckpt["model_state"])
    else:
        log.warning("best.pt not found — evaluating last model state.")

    evaluator = Evaluator(model, test_loader, config, config.run_dir)
    metrics = evaluator.evaluate()
    evaluator.save_report(metrics)
    evaluator.plot_confusion_matrix()
    evaluator.plot_training_curves(
        config.run_dir / "train_log.csv"
    )

    log.info(
        f"\nFinal results — "
        f"Acc={metrics['accuracy']} | F1={metrics['f1']} | "
        f"P={metrics['precision']} | R={metrics['recall']}"
    )


def train_yolo(config: Config) -> None:
    """YOLO training pipeline."""
    from src.models.yolo_classifier import YOLOClassifier

    yolo = YOLOClassifier(config.model_name, config)
    run_dir = yolo.train()

    log.info("Evaluating YOLO model on test set...")
    test_dir = config.processed_dir
    metrics = yolo.evaluate(test_dir)

    import json
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved to {metrics_file}")
    log.info(
        f"YOLO results — "
        f"Acc={metrics['accuracy']} | F1={metrics['f1']} | "
        f"P={metrics['precision']} | R={metrics['recall']}"
    )

    # FPS
    fps = yolo.measure_fps()
    from src.utils import save_fps_report
    save_fps_report(fps, run_dir, config.model_name)
    log.info(f"FPS: {fps['fps']} | Latency: {fps['mean_ms']}ms")


def main():
    args = parse_args()

    overrides = {}
    if args.run_name:
        overrides["run_name"] = args.run_name
    if args.subset is not None:
        overrides["subset_fraction"] = args.subset
    if args.epochs is not None:
        overrides["epochs"] = args.epochs

    config = load_config(args.config, overrides=overrides)
    set_seed(config.seed)

    log.info(f"Run name : {config.run_name}")
    log.info(f"Env      : {config.env}")

    if config.model_name in YOLO_MODELS:
        train_yolo(config)
    else:
        train_resnet(config, args.resume)


if __name__ == "__main__":
    main()
