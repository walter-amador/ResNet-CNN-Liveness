"""
Webcam-based manual sample collection for the liveness dataset.

Lets you capture labeled face frames directly from your webcam to build
the manually-collected portion of the dataset (data/raw/manual_samples/).

Controls during preview:
    R  — switch to REAL label (live face)
    F  — switch to FAKE label (phone replay or printed photo)
    SPACE — start a capture burst (saves N frames with configured delay)
    Q  — quit

Usage:
    # Collect 100 real samples, 100 fake samples:
    python scripts/collect_samples.py \
        --output data/raw/manual_samples \
        --n-samples 100 \
        --delay 0.3

    # Collect only fake samples:
    python scripts/collect_samples.py \
        --output data/raw/manual_samples \
        --label fake \
        --n-samples 150
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.face_detector import FaceDetector

FONT = cv2.FONT_HERSHEY_SIMPLEX

COLORS = {"real": (0, 200, 0), "fake": (0, 0, 220)}
LABEL_KEYS = {ord("r"): "real", ord("f"): "fake"}


def preview_loop(
    cap: cv2.VideoCapture,
    detector: FaceDetector,
    output_root: Path,
    n_samples: int,
    delay: float,
) -> None:
    """Interactive preview + capture loop."""
    current_label = "real"
    counts = {"real": 0, "fake": 0}
    capturing = False
    last_capture_time = 0.0

    print("\nControls:")
    print("  R      — label = REAL  (live face)")
    print("  F      — label = FAKE  (phone replay / printed photo)")
    print("  SPACE  — start capture burst")
    print("  Q      — quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed.")
            break

        # Face detection for live preview
        boxes = detector.detect(frame)
        vis = frame.copy()
        color = COLORS[current_label]

        if boxes:
            for x, y, w, h in boxes:
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        # Status bar
        status = (
            f"Label: {current_label.upper()}  |  "
            f"real={counts['real']}  fake={counts['fake']}  |  "
            f"Target: {n_samples}"
        )
        cv2.putText(vis, status, (10, 30), FONT, 0.7, color, 2, cv2.LINE_AA)

        if capturing:
            cv2.putText(vis, "● CAPTURING", (10, 65), FONT, 0.7, (0, 0, 255), 2)

        cv2.imshow("Sample Collector (R=real F=fake SPACE=capture Q=quit)", vis)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key in LABEL_KEYS:
            current_label = LABEL_KEYS[key]
            print(f"  Label → {current_label.upper()}")
        elif key == ord(" "):
            capturing = True
            last_capture_time = 0.0
            print(f"  Starting capture burst: {n_samples} {current_label} frames...")

        # Capture logic
        if capturing:
            now = time.time()
            if now - last_capture_time >= delay:
                if not boxes:
                    print("  [WARN] No face detected — skipping frame")
                else:
                    crop = detector.crop_largest(frame)
                    if crop is not None:
                        save_dir = output_root / current_label
                        save_dir.mkdir(parents=True, exist_ok=True)
                        fname = f"manual_{current_label}_{counts[current_label]:05d}.jpg"
                        cv2.imwrite(str(save_dir / fname), crop)
                        counts[current_label] += 1
                        print(
                            f"  Saved: {fname} "
                            f"({counts[current_label]}/{n_samples})"
                        )

                last_capture_time = now
                if counts[current_label] >= n_samples:
                    capturing = False
                    print(
                        f"  Capture complete: {n_samples} {current_label} frames saved."
                    )

    print(f"\nFinal counts — real: {counts['real']}, fake: {counts['fake']}")
    print(f"Saved to: {output_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect manual liveness samples from webcam"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/raw/manual_samples"),
        help="Root output directory (real/ and fake/ subdirs created automatically)"
    )
    parser.add_argument(
        "--n-samples", type=int, default=100,
        help="Number of frames to capture per burst"
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="Seconds between captured frames"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera device index"
    )
    parser.add_argument(
        "--face-backend", default="mediapipe",
        choices=["mediapipe", "haar", "mtcnn"],
    )
    parser.add_argument(
        "--label", default=None, choices=["real", "fake"],
        help="If set, starts in this label mode (still changeable with R/F)"
    )
    args = parser.parse_args()

    # Resolve path from repo root if relative
    repo_root = Path(__file__).resolve().parent.parent
    output_root = args.output if args.output.is_absolute() else repo_root / args.output

    print(f"Loading face detector backend='{args.face_backend}' ...")
    detector = FaceDetector(backend=args.face_backend)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    preview_loop(cap, detector, output_root, args.n_samples, args.delay)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
