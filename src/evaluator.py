"""
Evaluation, metrics, and cross-experiment comparison.

Usage — evaluate a single model:
    evaluator = Evaluator(model, test_loader, config, run_dir)
    metrics = evaluator.evaluate()
    evaluator.save_report(metrics)
    evaluator.plot_confusion_matrix(run_dir / "confusion_matrix.png")
    evaluator.plot_training_curves(run_dir / "train_log.csv", run_dir / "training_curves.png")

Usage — compare all experiments:
    df = compare_experiments("experiments/results/")
    df.to_csv("experiments/results/comparison_table.csv")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config, CLASS_NAMES


class Evaluator:
    """Compute metrics and produce visualizations for a trained model.

    Args:
        model       : Trained PyTorch model in eval mode (or will be set).
        test_loader : DataLoader for the held-out test set.
        config      : Experiment configuration.
        run_dir     : Directory where outputs (metrics.json, plots) are saved.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        config: Config,
        run_dir: str | Path,
    ) -> None:
        self.model = model
        self.test_loader = test_loader
        self.config = config
        self.run_dir = Path(run_dir)
        self.device = torch.device(config.device)

        self._y_true: list[int] = []
        self._y_pred: list[int] = []
        self._y_prob: list[float] = []   # confidence for positive class

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> dict:
        """Run inference on test_loader and compute all metrics.

        Returns:
            metrics dict: accuracy, precision, recall, f1, class_report,
                          confusion_matrix (as nested list)
        """
        self.model.eval()
        self.model.to(self.device)

        self._y_true, self._y_pred, self._y_prob = [], [], []

        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating", leave=False):
                images = images.to(self.device, non_blocking=True)
                logits = self.model(images)
                probs  = torch.softmax(logits, dim=1)

                preds = logits.argmax(dim=1).cpu().tolist()
                trues = labels.tolist()
                confs = probs[:, 1].cpu().tolist()   # confidence for "live" class

                self._y_true.extend(trues)
                self._y_pred.extend(preds)
                self._y_prob.extend(confs)

        acc = accuracy_score(self._y_true, self._y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            self._y_true, self._y_pred, average="macro", zero_division=0
        )
        cm = confusion_matrix(self._y_true, self._y_pred).tolist()
        report = classification_report(
            self._y_true, self._y_pred,
            target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
            output_dict=True,
        )

        metrics = {
            "model": self.config.model_name,
            "run_name": self.config.run_name,
            "accuracy":  round(float(acc),  4),
            "precision": round(float(prec), 4),
            "recall":    round(float(rec),  4),
            "f1":        round(float(f1),   4),
            "confusion_matrix": cm,
            "class_report": report,
        }
        return metrics

    # ------------------------------------------------------------------
    # Saving outputs
    # ------------------------------------------------------------------

    def save_report(self, metrics: dict) -> None:
        """Write metrics to metrics.json and print a classification report."""
        out = self.run_dir / "metrics.json"
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {out}")
        print(f"\n  Accuracy  : {metrics['accuracy']}")
        print(f"  Precision : {metrics['precision']}")
        print(f"  Recall    : {metrics['recall']}")
        print(f"  F1        : {metrics['f1']}")

    def plot_confusion_matrix(
        self,
        output_path: Optional[str | Path] = None,
    ) -> None:
        """Save a seaborn heatmap of the confusion matrix."""
        if not self._y_true:
            raise RuntimeError("Call evaluate() before plotting.")

        cm = confusion_matrix(self._y_true, self._y_pred)
        labels = [CLASS_NAMES[0], CLASS_NAMES[1]]

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels, ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {self.config.run_name}")
        plt.tight_layout()

        out = output_path or self.run_dir / "confusion_matrix.png"
        fig.savefig(str(out), dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved to {out}")

    def plot_training_curves(
        self,
        log_csv_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> None:
        """Plot train/val loss and val accuracy from train_log.csv."""
        df = pd.read_csv(log_csv_path)
        if df.empty:
            print("Training log is empty — skipping curves plot.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss curves
        axes[0].plot(df["epoch"], df["train_loss"], label="Train loss")
        axes[0].plot(df["epoch"], df["val_loss"],   label="Val loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curves")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(df["epoch"], df["val_acc"], color="green", label="Val accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Validation Accuracy")
        axes[1].set_ylim(0, 1)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(self.config.run_name, fontsize=12)
        plt.tight_layout()

        out = output_path or self.run_dir / "training_curves.png"
        fig.savefig(str(out), dpi=150)
        plt.close(fig)
        print(f"Training curves saved to {out}")


# ---------------------------------------------------------------------------
# Cross-experiment comparison
# ---------------------------------------------------------------------------

def compare_experiments(results_dir: str | Path) -> pd.DataFrame:
    """Walk all run directories, read metrics.json, assemble a comparison table.

    Also reads fps_report.txt when available.

    Returns:
        DataFrame with one row per run, sorted by F1 descending.
    """
    results_dir = Path(results_dir)
    rows = []

    for metrics_file in sorted(results_dir.rglob("metrics.json")):
        run_dir = metrics_file.parent
        run_name = run_dir.name

        with open(metrics_file) as f:
            m = json.load(f)

        row = {
            "run_name":  m.get("run_name", run_name),
            "model":     m.get("model", "unknown"),
            "accuracy":  m.get("accuracy", None),
            "precision": m.get("precision", None),
            "recall":    m.get("recall", None),
            "f1":        m.get("f1", None),
        }

        # FPS report (optional)
        fps_file = run_dir / "fps_report.txt"
        if fps_file.exists():
            for line in fps_file.read_text().splitlines():
                if "FPS" in line:
                    try:
                        row["fps"] = float(line.split(":")[-1].strip())
                    except ValueError:
                        pass
                if "Mean latency" in line:
                    try:
                        row["mean_latency_ms"] = float(line.split(":")[-1].strip().split()[0])
                    except ValueError:
                        pass

        # Snapshot config — pick up subset_fraction for Experiment 2
        cfg_file = run_dir / "config_snapshot.yaml"
        if cfg_file.exists():
            import yaml
            with open(cfg_file) as f:
                cfg = yaml.safe_load(f) or {}
            row["subset_fraction"] = cfg.get("subset_fraction", None)
            row["epochs"] = cfg.get("epochs", None)

        rows.append(row)

    if not rows:
        print(f"No metrics.json files found under {results_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

    out_csv = results_dir / "comparison_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nExperiment comparison saved to: {out_csv}")
    print(df.to_string(index=False))
    return df
