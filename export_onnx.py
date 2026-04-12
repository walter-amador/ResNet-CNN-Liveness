"""
Export a trained ResNet checkpoint to ONNX format.

Steps:
  1. Load PyTorch checkpoint
  2. torch.onnx.export() with dynamic batch axis, opset 17
  3. onnx.checker.check_model() — validates graph structure
  4. onnxruntime numerical consistency check (atol 1e-4)
  5. Save preprocessing_meta.json alongside the .onnx

Usage:
    python export_onnx.py \
        --checkpoint experiments/results/resnet18_run1/weights/best.pt \
        --model resnet18 \
        --output export/model.onnx

    # With explicit opset:
    python export_onnx.py \
        --checkpoint experiments/results/resnet18_run1/weights/best.pt \
        --model resnet18 \
        --output export/model.onnx \
        --opset 17
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from src.config import IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES
from src.utils import get_logger

log = get_logger("export_onnx")


def export_to_onnx(
    checkpoint_path: Path,
    model_name: str,
    output_path: Path,
    image_size: int = 224,
    opset: int = 17,
    num_classes: int = 2,
) -> Path:
    """Export a ResNet checkpoint to ONNX.

    Returns:
        Path to the saved .onnx file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    from src.models.resnet_classifier import ResNetClassifier
    model = ResNetClassifier(model_name=model_name, num_classes=num_classes, pretrained=False)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    log.info(f"Loaded checkpoint: {checkpoint_path}")
    log.info(f"Model: {model_name} | Output: {output_path}")

    # --- Dummy input ---
    dummy = torch.randn(1, 3, image_size, image_size)

    # --- Export ---
    log.info(f"Exporting to ONNX (opset {opset}) ...")
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        verbose=False,
    )
    log.info(f"ONNX model saved to: {output_path}")

    # --- Validate graph ---
    try:
        import onnx
        model_onnx = onnx.load(str(output_path))
        onnx.checker.check_model(model_onnx)
        log.info("ONNX graph validation: PASSED")
    except ImportError:
        log.warning("onnx package not installed — skipping graph validation.")

    # --- Numerical consistency check ---
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(output_path),
            providers=["CPUExecutionProvider"],
        )

        dummy_np = dummy.numpy()
        ort_out = session.run(["logits"], {"input": dummy_np})[0]
        with torch.no_grad():
            pt_out = model(dummy).numpy()

        max_diff = float(np.abs(ort_out - pt_out).max())
        if max_diff < 1e-4:
            log.info(f"Numerical consistency: PASSED (max_diff={max_diff:.2e})")
        else:
            log.warning(
                f"Numerical consistency: LARGE DIFF ({max_diff:.2e}) — "
                "outputs may differ between PyTorch and ONNX Runtime."
            )
    except ImportError:
        log.warning("onnxruntime not installed — skipping consistency check.")

    # --- Save preprocessing metadata ---
    meta = {
        "mean": IMAGENET_MEAN,
        "std":  IMAGENET_STD,
        "image_size": image_size,
        "class_map": {str(k): v for k, v in CLASS_NAMES.items()},
        "input_name":  "input",
        "output_name": "logits",
        "model": model_name,
        "opset": opset,
    }
    meta_path = output_path.with_suffix("").parent / "preprocessing_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Preprocessing metadata saved to: {meta_path}")

    return output_path


def parse_args():
    p = argparse.ArgumentParser(description="Export ResNet checkpoint to ONNX")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--model", default="resnet18",
                   choices=["resnet18", "resnet34", "resnet50"])
    p.add_argument("--output", type=Path, default=Path("export/model.onnx"))
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def main():
    args = parse_args()
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        output_path=args.output,
        image_size=args.image_size,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
