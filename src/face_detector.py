"""
Face detection abstraction for the liveness detection pipeline.

Supports three backends behind a single interface:
  - "mediapipe"  (default) — best accuracy, real-time speed
  - "haar"                  — fast, offline, lower accuracy
  - "mtcnn"                 — most robust, slower

Usage:
    detector = FaceDetector(backend="mediapipe")
    crop = detector.crop_largest(frame_bgr)
    if crop is not None:
        # face found — use crop for classification
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class FaceDetector:
    """Unified face detection interface.

    All backends expose the same two public methods:
        detect(frame_bgr)          → list of (x, y, w, h) boxes or None
        crop_largest(frame_bgr)    → BGR crop of the largest face or None
    """

    def __init__(
        self,
        backend: str = "mediapipe",
        min_confidence: float = 0.7,
        padding_frac: float = 0.10,
    ) -> None:
        self.backend = backend.lower()
        self.min_confidence = min_confidence
        self.padding_frac = padding_frac
        self._detector = self._load_backend()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> Optional[list[tuple[int, int, int, int]]]:
        """Detect faces in a BGR frame.

        Returns:
            List of (x, y, w, h) bounding boxes, or None if no faces found.
        """
        if self.backend == "mediapipe":
            return self._detect_mediapipe(frame_bgr)
        if self.backend == "haar":
            return self._detect_haar(frame_bgr)
        if self.backend == "mtcnn":
            return self._detect_mtcnn(frame_bgr)
        raise ValueError(f"Unknown backend: {self.backend}")

    def crop_largest(
        self,
        frame_bgr: np.ndarray,
        padding_frac: Optional[float] = None,
        output_size: int = 224,
    ) -> Optional[np.ndarray]:
        """Detect, crop, and resize the largest face in the frame.

        Adds fractional padding around the detected box, clamps to image
        boundaries, then resizes to output_size × output_size.

        Returns:
            BGR face crop as np.ndarray, or None if no face was detected.
        """
        boxes = self.detect(frame_bgr)
        if not boxes:
            return None

        pad = padding_frac if padding_frac is not None else self.padding_frac

        # Pick the largest box by area
        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])

        ih, iw = frame_bgr.shape[:2]
        px = int(w * pad)
        py = int(h * pad)

        x1 = max(0, x - px)
        y1 = max(0, y - py)
        x2 = min(iw, x + w + px)
        y2 = min(ih, y + h + py)

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)

    def draw_boxes(
        self,
        frame_bgr: np.ndarray,
        boxes: Optional[list[tuple[int, int, int, int]]],
        color: tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Draw detected face boxes on a copy of the frame."""
        out = frame_bgr.copy()
        if boxes:
            for x, y, w, h in boxes:
                cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        return out

    # ------------------------------------------------------------------
    # Backend loading
    # ------------------------------------------------------------------

    def _load_backend(self):
        if self.backend == "mediapipe":
            return self._load_mediapipe()
        if self.backend == "haar":
            return self._load_haar()
        if self.backend == "mtcnn":
            return self._load_mtcnn()
        raise ValueError(
            f"Unknown face detector backend '{self.backend}'. "
            "Choose from: mediapipe, haar, mtcnn"
        )

    def _load_mediapipe(self):
        try:
            import mediapipe as mp
            fd = mp.solutions.face_detection.FaceDetection(
                model_selection=1,  # 1 = full-range model (up to 5m)
                min_detection_confidence=self.min_confidence,
            )
            return fd
        except ImportError as e:
            raise ImportError(
                "mediapipe is required for backend='mediapipe'. "
                "Install with: pip install mediapipe"
            ) from e

    def _load_haar(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
        return cascade

    def _load_mtcnn(self):
        try:
            from mtcnn import MTCNN
            return MTCNN()
        except ImportError as e:
            raise ImportError(
                "mtcnn is required for backend='mtcnn'. "
                "Install with: pip install mtcnn"
            ) from e

    # ------------------------------------------------------------------
    # Backend-specific detection
    # ------------------------------------------------------------------

    def _detect_mediapipe(
        self, frame_bgr: np.ndarray
    ) -> Optional[list[tuple[int, int, int, int]]]:
        import mediapipe as mp

        ih, iw = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(rgb)

        if not results.detections:
            return None

        boxes = []
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * iw))
            y = max(0, int(bb.ymin * ih))
            w = int(bb.width * iw)
            h = int(bb.height * ih)
            boxes.append((x, y, w, h))

        return boxes if boxes else None

    def _detect_haar(
        self, frame_bgr: np.ndarray
    ) -> Optional[list[tuple[int, int, int, int]]]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return None
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    def _detect_mtcnn(
        self, frame_bgr: np.ndarray
    ) -> Optional[list[tuple[int, int, int, int]]]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.detect_faces(rgb)
        if not results:
            return None

        boxes = []
        for r in results:
            if r["confidence"] >= self.min_confidence:
                x, y, w, h = r["box"]
                x, y = max(0, x), max(0, y)
                boxes.append((x, y, w, h))

        return boxes if boxes else None

    def __del__(self):
        # MediaPipe requires explicit close
        if self.backend == "mediapipe" and self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Quick test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    backend = sys.argv[1] if len(sys.argv) > 1 else "mediapipe"
    print(f"Testing FaceDetector with backend='{backend}'")

    detector = FaceDetector(backend=backend)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        sys.exit(1)

    print("Press Q to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes = detector.detect(frame)
        vis = detector.draw_boxes(frame, boxes)
        label = f"Faces: {len(boxes) if boxes else 0}"
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Face Detector Test", vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
