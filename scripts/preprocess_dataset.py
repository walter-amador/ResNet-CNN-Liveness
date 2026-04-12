"""
Preprocess extracted frames: detect face, crop, resize to 224×224.

Reads from data/frames/{real,fake}/ (or any input-root with class subdirs),
runs face detection on each image, crops the largest detected face with padding,
resizes to the target image size, and saves to data/processed/{real,fake}/.

Images where no face is detected are logged to skipped.log and excluded.

Usage:
    python scripts/preprocess_dataset.py \
        --input-root  data/frames \
        --output-root data/processed \
        --backend mediapipe \
        --image-size 224

    # Include manually collected samples:
    python scripts/preprocess_dataset.py \
        --input-root  data/raw/manual_samples \
        --output-root data/processed \
        --source manual \
        --backend mediapipe
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.face_detector import FaceDetector

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess_directory(
    input_dir: Path,
    output_dir: Path,
    detector: FaceDetector,
    image_size: int = 224,
    source: str = "",
    jpeg_quality: int = 95,
) -> dict:
    """Face-crop and resize all images in input_dir, save to output_dir.

    Returns:
        dict with keys: total, saved, skipped
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    skip_log = output_dir.parent / "skipped.log"

    images = [p for p in input_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        log.warning(f"No images found in {input_dir}")
        return {"total": 0, "saved": 0, "skipped": 0}

    log.info(f"Processing {len(images)} image(s) from {input_dir}")

    saved = 0
    skipped = 0
    skipped_paths: list[str] = []

    for i, img_path in enumerate(images):
        if i % 500 == 0 and i > 0:
            log.info(f"  Progress: {i}/{len(images)} ({saved} saved, {skipped} skipped)")

        frame = cv2.imread(str(img_path))
        if frame is None:
            skipped += 1
            skipped_paths.append(str(img_path) + "  [read error]")
            continue

        crop = detector.crop_largest(frame, output_size=image_size)
        if crop is None:
            skipped += 1
            skipped_paths.append(str(img_path) + "  [no face]")
            continue

        # Preserve original stem, prepend source if provided
        stem = img_path.stem
        prefix = f"{source}_" if source else ""
        out_name = f"{prefix}{stem}.jpg"
        out_path = output_dir / out_name

        # Avoid name collisions
        if out_path.exists():
            out_name = f"{prefix}{stem}_{saved:05d}.jpg"
            out_path = output_dir / out_name

        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        saved += 1

    # Write skip log
    if skipped_paths:
        with open(skip_log, "a") as f:
            f.write("\n".join(skipped_paths) + "\n")

    return {"total": len(images), "saved": saved, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser(
        description="Face-detect, crop, and resize dataset images"
    )
    parser.add_argument(
        "--input-root", type=Path, required=True,
        help="Root directory with real/ and fake/ subdirectories"
    )
    parser.add_argument(
        "--output-root", type=Path, required=True,
        help="Root directory for processed output (will create real/ and fake/)"
    )
    parser.add_argument(
        "--backend", default="mediapipe", choices=["mediapipe", "haar", "mtcnn"],
        help="Face detection backend"
    )
    parser.add_argument("--image-size", type=int, default=224, help="Output image size (px)")
    parser.add_argument(
        "--source", default="",
        help="Optional prefix for output filenames (e.g. 'kaggle' or 'manual')"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.7,
        help="Minimum face detection confidence (mediapipe/mtcnn)"
    )
    parser.add_argument(
        "--padding", type=float, default=0.10,
        help="Fractional padding around detected face box"
    )
    args = parser.parse_args()

    log.info(f"Loading face detector backend='{args.backend}'")
    detector = FaceDetector(
        backend=args.backend,
        min_confidence=args.confidence,
        padding_frac=args.padding,
    )

    grand_total = grand_saved = grand_skipped = 0

    for cls in ("real", "fake"):
        in_dir = args.input_root / cls
        out_dir = args.output_root / cls

        if not in_dir.exists():
            log.warning(f"Skipping missing directory: {in_dir}")
            continue

        log.info(f"\n=== Class: {cls} ===")
        stats = preprocess_directory(
            in_dir, out_dir, detector,
            image_size=args.image_size,
            source=args.source,
        )
        log.info(
            f"  Total={stats['total']}, Saved={stats['saved']}, "
            f"Skipped={stats['skipped']}"
        )
        grand_total   += stats["total"]
        grand_saved   += stats["saved"]
        grand_skipped += stats["skipped"]

    log.info(
        f"\nOverall: {grand_total} images → "
        f"{grand_saved} saved, {grand_skipped} skipped "
        f"(skip rate {grand_skipped/max(grand_total,1)*100:.1f}%)"
    )
    if grand_skipped > 0:
        log.info(f"Skipped images logged to: {args.output_root}/skipped.log")


if __name__ == "__main__":
    main()
