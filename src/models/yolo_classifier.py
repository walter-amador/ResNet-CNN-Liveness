"""
YOLO11 classification wrapper for face liveness detection.

Wraps the Ultralytics YOLO API behind the same interface used by the ResNet
evaluator so results can be compared directly.

Supported variants: "yolo11n-cls", "yolo11s-cls"

Note: YOLO classification models have their own training loop managed by
Ultralytics. This module:
  1. Generates the Ultralytics-format data YAML from the processed/ directory.
  2. Runs model.train() with the correct hyperparameters.
  3. Copies Ultralytics output artifacts to the standard run directory.
  4. Provides an evaluate() method that returns the same metrics dict format
     as src/evaluator.py so compare_experiments() works uniformly.
  5. Provides export_to_onnx() for the export pipeline.

Usage:
    yolo = YOLOClassifier("yolo11n-cls", config)
    yolo.train()
    metrics = yolo.evaluate(test_data_dir)
    yolo.export_to_onnx("export/model_yolo.onnx")
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from src.config import Config, CLASS_NAMES


_SUPPORTED = {"yolo11n-cls", "yolo11s-cls"}


class YOLOClassifier:
    """Ultralytics YOLO classification model wrapper.

    Args:
        model_variant : "yolo11n-cls" | "yolo11s-cls"
        config        : Experiment config (for paths, device, hyperparams).
    """

    def __init__(self, model_variant: str, config: Config) -> None:
        if model_variant not in _SUPPORTED:
            raise ValueError(
                f"Unsupported YOLO variant '{model_variant}'. "
                f"Choose from: {_SUPPORTED}"
            )

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for YOLO models. "
                "Install with: pip install ultralytics"
            ) from e

        self.model_variant = model_variant
        self.config = config
        self.model = YOLO(f"{model_variant}.pt")  # downloads pretrained weights
        self._run_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # Data YAML
    # ------------------------------------------------------------------

    def build_data_yaml(self, processed_root: Optional[str | Path] = None) -> Path:
        """Write a temporary Ultralytics classification data YAML.

        The YAML points at the processed/ directory which already has
        real/ and fake/ subdirectories — exactly the layout Ultralytics expects.

        Returns:
            Path to the written YAML file.
        """
        root = Path(processed_root or self.config.processed_dir)

        # Ultralytics classification expects:
        #   path: <root>
        #   train: real, fake   (relative subdirs for each class)
        # But actually for classification it uses a single dataset root
        # with class-named subdirectories — we point it at processed/ directly.

        data = {
            "path": str(root),
            "train": ".",    # subdirs real/ fake/ are auto-discovered
            "val":   ".",
            "nc": self.config.num_classes,
            "names": {v: k for k, v in {"fake": 0, "real": 1}.items()},
        }

        tmp = Path(tempfile.mkdtemp()) / "liveness_data.yaml"
        with open(tmp, "w") as f:
            yaml.dump(data, f)

        return tmp

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, processed_root: Optional[str | Path] = None) -> Path:
        """Run YOLO classification training.

        Returns:
            Path to the Ultralytics output directory for this run.
        """
        from src.utils import setup_run_dir

        data_yaml = self.build_data_yaml(processed_root)
        run_dir = setup_run_dir(self.config.experiment_root, self.config.run_name)
        self._run_dir = run_dir

        device_str = self.config.device
        # Ultralytics uses "mps" as-is, "cuda" for GPU, "cpu" for CPU
        if device_str == "mps":
            device_str = "mps"

        print(f"Starting YOLO training: {self.model_variant}")
        print(f"  Device : {device_str}")
        print(f"  Epochs : {self.config.epochs}")
        print(f"  Batch  : {self.config.batch_size}")
        print(f"  Data   : {data_yaml}")

        self.model.train(
            data=str(data_yaml),
            epochs=self.config.epochs,
            imgsz=self.config.image_size,
            batch=self.config.batch_size,
            device=device_str,
            project=str(run_dir),
            name="weights",
            exist_ok=True,
            verbose=True,
            patience=self.config.patience,
            seed=self.config.seed,
        )

        # Save config snapshot
        from src.config import save_config_snapshot
        save_config_snapshot(self.config, run_dir)

        return run_dir

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, test_dir: Optional[str | Path] = None) -> dict:
        """Run prediction on a directory and compute metrics.

        Returns a dict matching the format produced by src/evaluator.py:
            accuracy, precision, recall, f1, class_report (dict)
        """
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            classification_report,
        )

        if test_dir is None:
            test_dir = self.config.processed_dir

        test_dir = Path(test_dir)
        all_paths, all_true = self._collect_test_images(test_dir)

        if not all_paths:
            raise RuntimeError(f"No test images found under {test_dir}")

        results = self.model.predict(all_paths, verbose=False)

        # Map YOLO class names to our label indices
        # Ultralytics returns class indices based on the training order
        # which follows sorted directory names: fake=0, real=1
        all_pred = [int(r.probs.top1) for r in results]

        acc = accuracy_score(all_true, all_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            all_true, all_pred, average="macro", zero_division=0
        )
        report = classification_report(
            all_true, all_pred,
            target_names=[CLASS_NAMES[0], CLASS_NAMES[1]],
            output_dict=True,
        )

        metrics = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
            "class_report": report,
            "model": self.model_variant,
        }
        return metrics

    def _collect_test_images(
        self, root: Path
    ) -> tuple[list[str], list[int]]:
        """Walk root/{real,fake}/ and collect (path, label) pairs."""
        IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
        paths, labels = [], []
        for cls_name, label in [("fake", 0), ("real", 1)]:
            cls_dir = root / cls_name
            if not cls_dir.exists():
                continue
            for p in cls_dir.rglob("*"):
                if p.suffix.lower() in IMAGE_EXT:
                    paths.append(str(p))
                    labels.append(label)
        return paths, labels

    # ------------------------------------------------------------------
    # ONNX Export
    # ------------------------------------------------------------------

    def export_to_onnx(self, output_path: str | Path) -> Path:
        """Export the best YOLO checkpoint to ONNX format.

        Returns:
            Path to the exported .onnx file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        exported = self.model.export(
            format="onnx",
            imgsz=self.config.image_size,
            opset=self.config.onnx_opset,
            dynamic=True,
        )

        src_path = Path(exported)
        shutil.copy2(src_path, output_path)
        print(f"YOLO ONNX model saved to: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # FPS measurement
    # ------------------------------------------------------------------

    def measure_fps(self, n_iters: int = 100, warmup: int = 10) -> dict:
        """Measure inference throughput on a dummy image."""
        import torch
        dummy_path = None

        # Create a temporary dummy image
        import cv2, tempfile
        img = np.zeros((self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
        tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp_img.name, img)
        dummy_path = tmp_img.name

        # Warmup
        for _ in range(warmup):
            self.model.predict(dummy_path, verbose=False)

        # Timed
        latencies = []
        for _ in range(n_iters):
            t0 = time.perf_counter()
            self.model.predict(dummy_path, verbose=False)
            latencies.append((time.perf_counter() - t0) * 1000)

        Path(dummy_path).unlink(missing_ok=True)

        mean_ms = float(np.mean(latencies))
        return {
            "fps": round(1000.0 / mean_ms, 2),
            "mean_ms": round(mean_ms, 3),
            "min_ms": round(float(np.min(latencies)), 3),
            "max_ms": round(float(np.max(latencies)), 3),
        }
