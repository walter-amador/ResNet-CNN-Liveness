"""
Standalone evaluation script.

Loads a saved checkpoint and runs full evaluation on the test set.
Also supports cross-experiment comparison across all runs.

Usage — evaluate a single run:
    python evaluate.py \
        --config experiments/configs/resnet18_baseline.yaml \
        --checkpoint experiments/results/resnet18_run1/weights/best.pt

Usage — compare all experiments:
    python evaluate.py --compare experiments/results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from src.config import load_config
from src.utils import set_seed, get_logger

log = get_logger("evaluate")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate liveness detector")
    p.add_argument("--config",     type=Path, help="YAML experiment config")
    p.add_argument("--checkpoint", type=Path, help="Path to best.pt checkpoint")
    p.add_argument("--output",     type=Path, help="Override output directory")
    p.add_argument(
        "--compare", type=Path, metavar="RESULTS_DIR",
        help="Walk a results directory and produce comparison_table.csv"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Cross-experiment comparison mode
    if args.compare:
        from src.evaluator import compare_experiments
        compare_experiments(args.compare)
        return

    # Single-run evaluation mode
    if not args.config or not args.checkpoint:
        log.error("--config and --checkpoint are required for single-run evaluation.")
        sys.exit(1)

    config = load_config(args.config)
    set_seed(config.seed)

    run_dir = args.output or config.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    from src.dataset import build_dataloaders
    from src.models.resnet_classifier import build_resnet
    from src.evaluator import Evaluator

    _, _, test_loader = build_dataloaders(config)

    model = build_resnet(config)
    ckpt = torch.load(args.checkpoint, map_location=config.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    log.info(f"Evaluating {args.checkpoint} on test set ({len(test_loader.dataset)} samples)")

    evaluator = Evaluator(model, test_loader, config, run_dir)
    metrics = evaluator.evaluate()
    evaluator.save_report(metrics)
    evaluator.plot_confusion_matrix()

    log_csv = run_dir / "train_log.csv"
    if log_csv.exists():
        evaluator.plot_training_curves(log_csv)
    else:
        log.info("No train_log.csv found — skipping training curves plot.")


if __name__ == "__main__":
    main()
