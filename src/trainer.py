"""
Training loop for ResNet-based liveness classifiers.

Features:
  - Automatic Mixed Precision (AMP) on CUDA; skipped on MPS/CPU
  - CosineAnnealingLR scheduler
  - Early stopping based on validation loss
  - Best-model checkpoint saving (best.pt) and last-epoch saving (last.pt)
  - Per-epoch CSV logging: epoch, train_loss, val_loss, val_acc, lr, elapsed_sec
  - tqdm progress bars compatible with both terminal and Colab

Usage:
    trainer = Trainer(model, config, train_loader, val_loader)
    history = trainer.train()
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config, save_config_snapshot
from src.utils import get_logger, setup_run_dir, save_fps_report, compute_fps


class Trainer:
    """Full training loop with checkpointing and early stopping.

    Args:
        model       : PyTorch model with a get_optimizer() method.
        config      : Experiment configuration.
        train_loader: DataLoader for training data.
        val_loader  : DataLoader for validation data.
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.run_dir = setup_run_dir(config.experiment_root, config.run_name)
        self.logger = get_logger(
            config.run_name, log_file=self.run_dir / "training.log"
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = model.get_optimizer(lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=1e-6
        )

        # AMP scaler (CUDA only)
        self.use_amp = (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self._best_val_loss = float("inf")
        self._patience_counter = 0
        self._start_epoch = 1

        # CSV log setup
        self._log_path = self.run_dir / "train_log.csv"
        self._log_fields = [
            "epoch", "train_loss", "val_loss", "val_acc", "lr", "elapsed_sec"
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            history dict with lists: train_loss, val_loss, val_acc, lr
        """
        save_config_snapshot(self.config, self.run_dir)
        self._init_log_csv()

        history: dict[str, list] = {
            "train_loss": [], "val_loss": [], "val_acc": [], "lr": []
        }

        self.logger.info(
            f"Starting training: {self.config.run_name} | "
            f"device={self.config.device} | epochs={self.config.epochs}"
        )
        self.logger.info(
            f"Train batches={len(self.train_loader)} | "
            f"Val batches={len(self.val_loader)}"
        )

        t_start = time.time()

        for epoch in range(self._start_epoch, self.config.epochs + 1):
            ep_start = time.time()

            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss,   val_acc   = self._run_epoch(self.val_loader,   train=False)

            self.scheduler.step()
            current_lr = self.optimizer.param_groups[-1]["lr"]
            elapsed = round(time.time() - ep_start, 1)

            # Record
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(current_lr)

            self._append_log_row(epoch, train_loss, val_loss, val_acc, current_lr, elapsed)

            self.logger.info(
                f"Epoch {epoch:3d}/{self.config.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | lr={current_lr:.2e} | {elapsed}s"
            )

            # Checkpoint
            self._save_checkpoint(self.run_dir / "weights" / "last.pt", epoch, val_loss)

            if val_loss < self._best_val_loss:
                self._best_val_loss = val_loss
                self._patience_counter = 0
                self._save_checkpoint(self.run_dir / "weights" / "best.pt", epoch, val_loss)
                self.logger.info(f"  → New best checkpoint saved (val_loss={val_loss:.4f})")
            else:
                self._patience_counter += 1
                if self._patience_counter >= self.config.patience:
                    self.logger.info(
                        f"Early stopping triggered after {epoch} epochs "
                        f"(no improvement for {self.config.patience} epochs)"
                    )
                    break

        total_time = round(time.time() - t_start, 1)
        self.logger.info(f"Training complete in {total_time}s")

        # FPS measurement on best model
        self._measure_and_save_fps()

        return history

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self._best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self._start_epoch = ckpt.get("epoch", 1) + 1
        self.logger.info(f"Resumed from {path} (epoch {self._start_epoch - 1})")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_epoch(
        self, loader: DataLoader, train: bool
    ) -> tuple[float, float]:
        """Run one epoch (train or eval). Returns (mean_loss, accuracy)."""
        self.model.train(train)
        total_loss = 0.0
        correct = 0
        total = 0

        desc = "Train" if train else "Val  "
        with torch.set_grad_enabled(train):
            for images, labels in tqdm(loader, desc=desc, leave=False):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        mean_loss = total_loss / total if total > 0 else 0.0
        accuracy  = correct / total    if total > 0 else 0.0
        return round(mean_loss, 6), round(accuracy, 6)

    def _save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "best_val_loss": self._best_val_loss,
                "config": self.config.to_dict(),
            },
            path,
        )

    def _init_log_csv(self) -> None:
        with open(self._log_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self._log_fields).writeheader()

    def _append_log_row(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_acc: float,
        lr: float,
        elapsed: float,
    ) -> None:
        with open(self._log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._log_fields)
            writer.writerow({
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "lr": f"{lr:.2e}",
                "elapsed_sec": elapsed,
            })

    def _measure_and_save_fps(self) -> None:
        """Measure throughput and save fps_report.txt."""
        try:
            self.model.eval()
            fps_data = compute_fps(
                self.model,
                self.device,
                image_size=self.config.image_size,
                n_iters=100,
            )
            save_fps_report(fps_data, self.run_dir, self.config.model_name)
            self.logger.info(
                f"FPS: {fps_data['fps']} | "
                f"Latency: {fps_data['mean_ms']}ms avg"
            )
        except Exception as e:
            self.logger.warning(f"FPS measurement failed: {e}")
