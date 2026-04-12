"""
Extract frames from videos for the liveness detection dataset.

Reads .mp4/.avi video files from an input directory (organized by class)
and writes individual JPEG frames to the output directory.

Blur filtering: frames with Laplacian variance < blur_threshold are skipped.
FPS sampling: only every Nth frame is saved to avoid redundancy.

Usage — single class:
    python scripts/extract_frames.py \
        --input  data/raw/kaggle_videos/real \
        --output data/frames/real \
        --source kaggle \
        --fps-sample 2 \
        --max-frames 500

Usage — both classes at once:
    python scripts/extract_frames.py \
        --input-root data/raw/kaggle_videos \
        --output-root data/frames \
        --source kaggle \
        --fps-sample 2 \
        --max-frames 500
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def is_blurry(frame: np.ndarray, threshold: float = 100.0) -> bool:
    """Return True if the frame's Laplacian variance is below threshold."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def extract_frames(
    video_path: Path,
    output_dir: Path,
    source: str = "kaggle",
    fps_sample: int = 2,
    max_frames: int = 500,
    blur_threshold: float = 100.0,
    jpeg_quality: int = 95,
) -> tuple[int, int]:
    """Extract frames from a single video file.

    Args:
        video_path      : Path to the video file.
        output_dir      : Directory to save extracted frames.
        source          : Source tag prepended to filenames ("kaggle" or "manual").
        fps_sample      : Save 1 frame every fps_sample frames of source video.
        max_frames      : Maximum frames to extract per video.
        blur_threshold  : Laplacian variance threshold; blurrier frames are skipped.
        jpeg_quality    : JPEG compression quality (0–100).

    Returns:
        (saved_count, skipped_count) — number of frames saved and skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.warning(f"Cannot open video: {video_path}")
        return 0, 0

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # How many source video frames to skip between saves
    step = max(1, round(video_fps / fps_sample))

    stem = video_path.stem
    saved = 0
    skipped = 0
    frame_idx = 0

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            if is_blurry(frame, blur_threshold):
                skipped += 1
            else:
                filename = f"{source}_{stem}_{saved:05d}.jpg"
                out_path = output_dir / filename
                cv2.imwrite(
                    str(out_path),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
                )
                saved += 1

        frame_idx += 1

    cap.release()
    return saved, skipped


def process_directory(
    input_dir: Path,
    output_dir: Path,
    source: str,
    fps_sample: int,
    max_frames: int,
    blur_threshold: float,
) -> dict:
    """Process all videos in a directory."""
    videos = [p for p in input_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS]

    if not videos:
        log.warning(f"No videos found in {input_dir}")
        return {"videos": 0, "saved": 0, "skipped": 0}

    log.info(f"Found {len(videos)} video(s) in {input_dir}")

    total_saved = 0
    total_skipped = 0

    for i, vp in enumerate(videos, 1):
        log.info(f"[{i}/{len(videos)}] Processing {vp.name}")
        saved, skipped = extract_frames(
            vp, output_dir, source, fps_sample, max_frames, blur_threshold
        )
        log.info(f"  → saved={saved}, blurry_skipped={skipped}")
        total_saved += saved
        total_skipped += skipped

    return {"videos": len(videos), "saved": total_saved, "skipped": total_skipped}


def main():
    parser = argparse.ArgumentParser(description="Extract frames from liveness videos")

    # Single-class mode
    parser.add_argument("--input", type=Path, help="Input video directory (single class)")
    parser.add_argument("--output", type=Path, help="Output frame directory (single class)")

    # Dual-class mode
    parser.add_argument("--input-root", type=Path, help="Root with real/ and fake/ subdirs")
    parser.add_argument("--output-root", type=Path, help="Root for output real/ and fake/ dirs")

    parser.add_argument("--source", default="kaggle", help="Source tag: kaggle | manual")
    parser.add_argument("--fps-sample", type=int, default=2, help="Saved frames per second")
    parser.add_argument("--max-frames", type=int, default=500, help="Max frames per video")
    parser.add_argument("--blur-threshold", type=float, default=100.0,
                        help="Laplacian variance threshold for blur detection")

    args = parser.parse_args()

    if args.input_root:
        # Process both classes
        for cls in ("real", "fake"):
            in_dir = args.input_root / cls
            out_dir = (args.output_root or args.input_root.parent / "frames") / cls
            if not in_dir.exists():
                log.warning(f"Skipping missing directory: {in_dir}")
                continue
            log.info(f"\n--- Class: {cls} ---")
            stats = process_directory(
                in_dir, out_dir, args.source, args.fps_sample,
                args.max_frames, args.blur_threshold
            )
            log.info(
                f"Class '{cls}' done: {stats['videos']} videos, "
                f"{stats['saved']} frames saved, {stats['skipped']} blurry skipped"
            )
    elif args.input and args.output:
        stats = process_directory(
            args.input, args.output, args.source, args.fps_sample,
            args.max_frames, args.blur_threshold
        )
        log.info(
            f"Done: {stats['videos']} videos, "
            f"{stats['saved']} frames saved, {stats['skipped']} blurry skipped"
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
