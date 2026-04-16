"""
COSC-4117EL Assignment 3 — Group 8
Real-Time Face Liveness Detection — Webcam Inference

Usage:
    python COSC_4117EL_A3_G8-inference.py
    python COSC_4117EL_A3_G8-inference.py --model models/resnet18_best.pth
    python COSC_4117EL_A3_G8-inference.py --model models/resnet34_best.pth --cam 0

Controls:
    Q  — quit
    S  — save current frame to saved_frames/

Code assisted with Claude Code (Anthropic) — cited per course AI-use policy.
"""

import argparse
import os
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLASS_NAMES   = {0: 'spoof', 1: 'live'}
CLASS_COLORS  = {0: (0, 0, 220), 1: (0, 200, 0)}   # BGR: red=spoof, green=live


# ─────────────────────────────────────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, device: torch.device) -> tuple[nn.Module, int]:
    """
    Load a ResNet-18/34/50 liveness model from a checkpoint.
    Returns (model, image_size).
    """
    ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
    image_size = ckpt.get('image_size', 224)

    # Detect architecture from the state dict keys / layer sizes
    sd = ckpt['state_dict']
    # ResNet-50 has a different layer structure (Bottleneck vs BasicBlock)
    # Distinguish by checking the fc weight shape: 512→ResNet18/34, 2048→ResNet50
    fc_in = None
    for k, v in sd.items():
        if k == 'fc.1.weight':   # our head: Sequential(Dropout, Linear)
            fc_in = v.shape[1]
            break

    if fc_in == 2048:
        base = tv_models.resnet50(weights=None)
    elif fc_in == 512:
        # Could be ResNet-18 or ResNet-34 — both have 512 fc_in
        # Try to detect by layer4 conv weight shape
        l4_key = 'layer4.1.conv1.weight'
        if l4_key in sd and sd[l4_key].shape == (512, 256, 3, 3):
            base = tv_models.resnet34(weights=None)
        else:
            base = tv_models.resnet18(weights=None)
    else:
        # Fallback
        base = tv_models.resnet18(weights=None)

    base.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(base.fc.in_features, 2))
    base.load_state_dict(sd)
    base.eval()
    return base.to(device), image_size


# ─────────────────────────────────────────────────────────────────────────────
# Inference transform
# ─────────────────────────────────────────────────────────────────────────────
def get_transform(image_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Classify a single face crop
# ─────────────────────────────────────────────────────────────────────────────
def classify_face(
    face_rgb: np.ndarray,
    model: nn.Module,
    transform: T.Compose,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[int, float, float]:
    """
    Run liveness classification on a (H, W, 3) RGB numpy array.
    Returns (predicted_label, spoof_prob, live_prob).

    threshold — minimum live probability required to predict 'live'.
    Lower values (e.g. 0.40) make the classifier more permissive for
    unseen faces; higher values (e.g. 0.60) make it stricter.
    """
    pil_img = Image.fromarray(face_rgb)
    tensor  = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
    spoof_prob = float(probs[0].item())
    live_prob  = float(probs[1].item())
    pred       = 1 if live_prob >= threshold else 0
    return pred, spoof_prob, live_prob


# ─────────────────────────────────────────────────────────────────────────────
# Rolling FPS counter
# ─────────────────────────────────────────────────────────────────────────────
class RollingFPS:
    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ─────────────────────────────────────────────────────────────────────────────
# Main webcam loop
# ─────────────────────────────────────────────────────────────────────────────
def run_webcam(
    model: nn.Module,
    transform: T.Compose,
    device: torch.device,
    cam_idx: int = 0,
    padding: float = 0.10,
    threshold: float = 0.5,
) -> None:
    """
    Open webcam, detect faces with OpenCV Haar Cascade, classify each as
    live/spoof, and display results.

    Controls:
        Q — quit
        S — save current frame to saved_frames/
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open webcam index {cam_idx}')

    save_dir = Path('saved_frames')
    save_dir.mkdir(exist_ok=True)

    fps_counter = RollingFPS(window=30)
    frame_count = 0

    print('Webcam open. Press Q to quit, S to save frame.')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to grab frame — exiting.')
            break

        frame_count += 1
        fps_counter.tick()
        fps = fps_counter.tick()

        h, w = frame.shape[:2]

        # ── Face detection ────────────────────────────────────────────────────
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        if len(faces) > 0:
            for (fx, fy, fw, fh) in faces:
                pad_x = int(padding * fw)
                pad_y = int(padding * fh)
                x1    = max(0, fx - pad_x)
                y1    = max(0, fy - pad_y)
                x2    = min(w, fx + fw + pad_x)
                y2    = min(h, fy + fh + pad_y)

                face_bgr = frame[y1:y2, x1:x2]
                if face_bgr.size == 0:
                    continue
                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

                # ── Classify ─────────────────────────────────────────────────
                label, spoof_p, live_p = classify_face(face_rgb, model, transform, device, threshold)
                cls_name = CLASS_NAMES[label]
                color    = CLASS_COLORS[label]

                # ── Draw bounding box ─────────────────────────────────────────
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # ── Label: prediction + both class probabilities ───────────────
                line1 = f'{cls_name}  (live {live_p*100:.0f}%  spoof {spoof_p*100:.0f}%)'
                font       = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness  = 2
                (tw, th), _ = cv2.getTextSize(line1, font, font_scale, thickness)
                ty = max(y1 - 6, th + 4)
                cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), color, -1)
                cv2.putText(frame, line1, (x1 + 2, ty - 1), font,
                            font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        else:
            cv2.putText(frame, 'No face detected', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

        # ── HUD overlay ───────────────────────────────────────────────────────
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Q=quit  S=save  thr={threshold:.2f}', (w - 230, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        cv2.imshow('Face Liveness Detection', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        if key == ord('s') or key == ord('S'):
            fname = save_dir / f'frame_{frame_count:06d}.png'
            cv2.imwrite(str(fname), frame)
            print(f'Saved: {fname}')

    cap.release()
    cv2.destroyAllWindows()
    print('Done.')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Real-time face liveness detection from webcam.'
    )
    parser.add_argument(
        '--model', default='models/resnet18_best.pth',
        help='Path to trained ResNet checkpoint (default: models/resnet18_best.pth)',
    )
    parser.add_argument(
        '--cam', type=int, default=0,
        help='Webcam device index (default: 0)',
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help=(
            'Minimum live-class probability to predict LIVE (default: 0.5). '
            'Lower this (e.g. 0.40) if unseen faces are wrongly classified as '
            'spoof; raise it (e.g. 0.60) for a stricter detector.'
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f'ERROR: model checkpoint not found at "{args.model}"')
        print('Train the model first by running COSC_4117EL_A3_G8-train.ipynb')
        return

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # Load model
    print(f'Loading model from: {args.model}')
    model, image_size = load_model(args.model, device)
    transform = get_transform(image_size)
    print(f'Model ready (image_size={image_size})')

    print(f'Live threshold : {args.threshold}')

    # Run
    run_webcam(
        model=model,
        transform=transform,
        device=device,
        cam_idx=args.cam,
        threshold=args.threshold,
    )


if __name__ == '__main__':
    main()
