"""
Real-time webcam liveness detection inference.

Opens the webcam, detects faces in each frame, and classifies each detected
face as "live" or "spoof". Displays a color-coded bounding box and prediction
with confidence score. Shows a rolling FPS counter.

Usage:
    # ResNet model:
    python infer_webcam.py \
        --checkpoint experiments/results/resnet18_run1/weights/best.pt \
        --model resnet18

    # With a specific face detector backend:
    python infer_webcam.py \
        --checkpoint experiments/results/resnet18_run1/weights/best.pt \
        --model resnet18 \
        --face-backend haar

Controls:
    Q  — quit
    S  — save current frame with prediction overlay to disk
    P  — pause / resume
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2
import numpy as np
import torch
from PIL import Image

from src.config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from src.face_detector import FaceDetector
from src.transforms import get_inference_transforms
from src.utils import RollingFPS, get_logger

log = get_logger("webcam")

# Visual style
COLORS = {
    "live":    (0, 200, 0),    # green
    "spoof":   (0, 0, 220),    # red (BGR)
    "no_face": (120, 120, 120), # gray
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


class LivenessInferenceEngine:
    """Load a trained model and classify individual webcam frames.

    Args:
        checkpoint_path : Path to best.pt checkpoint.
        model_name      : "resnet18" | "resnet34" | "resnet50"
        face_backend    : "mediapipe" | "haar" | "mtcnn"
        image_size      : Input size expected by the model.
        device          : "cuda" | "mps" | "cpu" (auto-detected if None)
        confidence_threshold: Minimum confidence to display a prediction.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        model_name: str = "resnet18",
        face_backend: str = "mediapipe",
        image_size: int = 224,
        device: str | None = None,
        confidence_threshold: float = 0.55,
    ) -> None:
        from src.utils import get_device
        self.device = torch.device(device) if device else get_device()
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold

        # Face detector
        self.detector = FaceDetector(backend=face_backend)

        # Transform
        self.transform = get_inference_transforms(image_size)

        # Model
        self.model = self._load_model(checkpoint_path, model_name)
        self.model.eval()

        # Latency tracking
        self._latencies: list[float] = []

    # ------------------------------------------------------------------
    # Single-frame inference
    # ------------------------------------------------------------------

    def predict_frame(
        self, frame_bgr: np.ndarray
    ) -> tuple[str, float, list | None]:
        """Detect face, classify it.

        Returns:
            (label, confidence, boxes)
            label      : "live" | "spoof" | "no_face"
            confidence : float in [0, 1]  (0 if no_face)
            boxes      : list of (x,y,w,h) or None
        """
        t0 = time.perf_counter()

        boxes = self.detector.detect(frame_bgr)
        if not boxes:
            return "no_face", 0.0, None

        crop = self.detector.crop_largest(frame_bgr, output_size=self.image_size)
        if crop is None:
            return "no_face", 0.0, boxes

        # BGR → RGB → PIL → tensor
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        pred_idx = int(probs.argmax().item())
        confidence = float(probs[pred_idx].item())
        label = CLASS_NAMES[pred_idx]

        # Low-confidence → treat as no_face
        if confidence < self.confidence_threshold:
            label = "no_face"

        self._latencies.append((time.perf_counter() - t0) * 1000)
        return label, confidence, boxes

    # ------------------------------------------------------------------
    # Main webcam loop
    # ------------------------------------------------------------------

    def run_webcam_loop(self, camera_id: int = 0, save_dir: str | None = None) -> None:
        """Open webcam and run continuous liveness inference."""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            log.error(f"Cannot open camera {camera_id}")
            return

        # Prefer higher resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        fps_counter = RollingFPS(window=30)
        paused = False
        frame_count = 0

        log.info("Webcam started. Press Q to quit, S to save frame, P to pause.")

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    log.warning("Frame read failed.")
                    break

                label, confidence, boxes = self.predict_frame(frame)
                vis = self._draw_overlay(frame, label, confidence, boxes, fps_counter.tick())
            else:
                # Show paused frame
                cv2.putText(vis, "PAUSED", (10, 60), FONT, 1, (0, 255, 255), 2)

            cv2.imshow("Liveness Detection (Q=quit S=save P=pause)", vis)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused
            elif key == ord("s") and save_dir is not None:
                out = Path(save_dir) / f"frame_{frame_count:05d}_{label}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), vis)
                log.info(f"Saved: {out}")

        cap.release()
        cv2.destroyAllWindows()
        self._print_latency_summary()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, checkpoint_path: str | Path, model_name: str):
        from src.models.resnet_classifier import ResNetClassifier
        model = ResNetClassifier(model_name=model_name, num_classes=2, pretrained=False)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(ckpt["model_state"])
        return model.to(self.device)

    def _draw_overlay(
        self,
        frame: np.ndarray,
        label: str,
        confidence: float,
        boxes: list | None,
        fps: float,
    ) -> np.ndarray:
        vis = frame.copy()
        color = COLORS.get(label, COLORS["no_face"])

        # Draw face boxes
        if boxes:
            for x, y, w, h in boxes:
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        # Prediction text
        if label != "no_face":
            text = f"{label.upper()}  {confidence:.1%}"
        else:
            text = "NO FACE"
        cv2.putText(vis, text, (10, 35), FONT, 1.1, color, 2, cv2.LINE_AA)

        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(vis, fps_text, (10, frame.shape[0] - 12),
                    FONT, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

        return vis

    def _print_latency_summary(self) -> None:
        if not self._latencies:
            return
        arr = np.array(self._latencies)
        print(
            f"\nLatency summary ({len(arr)} frames):\n"
            f"  Mean : {arr.mean():.1f} ms\n"
            f"  Min  : {arr.min():.1f} ms\n"
            f"  Max  : {arr.max():.1f} ms\n"
            f"  FPS  : {1000/arr.mean():.1f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Real-time liveness detection from webcam")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to trained model checkpoint (best.pt)")
    p.add_argument("--model", default="resnet18",
                   choices=["resnet18", "resnet34", "resnet50"],
                   help="Model architecture")
    p.add_argument("--face-backend", default="mediapipe",
                   choices=["mediapipe", "haar", "mtcnn"],
                   help="Face detection backend")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--camera", type=int, default=0, help="Camera device index")
    p.add_argument("--save-dir", type=Path, default=None,
                   help="Directory to save frames on S key press")
    p.add_argument("--device", default=None,
                   help="Force device: cuda | mps | cpu")
    p.add_argument("--threshold", type=float, default=0.55,
                   help="Minimum confidence to label (below → 'no_face')")
    return p.parse_args()


def main():
    args = parse_args()
    engine = LivenessInferenceEngine(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        face_backend=args.face_backend,
        image_size=args.image_size,
        device=args.device,
        confidence_threshold=args.threshold,
    )
    engine.run_webcam_loop(
        camera_id=args.camera,
        save_dir=str(args.save_dir) if args.save_dir else None,
    )


if __name__ == "__main__":
    main()
