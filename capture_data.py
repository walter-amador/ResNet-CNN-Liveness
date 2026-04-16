"""
COSC-4117EL Assignment 3 — Group 8
Webcam Data Capture Tool

Saves frames directly into the raw dataset folders:
    data/live/   — press L
    data/spoof/  — press S
    quit         — press Q

Filenames are timestamp-based so they never collide with existing files.
"""

import os
import time
from pathlib import Path

import cv2

# ── Destination folders ───────────────────────────────────────────────────────
LIVE_DIR  = Path('data/live')
SPOOF_DIR = Path('data/spoof')
LIVE_DIR.mkdir(parents=True, exist_ok=True)
SPOOF_DIR.mkdir(parents=True, exist_ok=True)

# ── Open webcam ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Cannot open webcam (index 0).')

# ── State ─────────────────────────────────────────────────────────────────────
count_live  = 0
count_spoof = 0
flash_msg   = ''       # short confirmation shown for a few frames
flash_color = (255, 255, 255)
flash_until = 0        # time.time() deadline

print('Capture tool ready.')
print('  L → save LIVE image')
print('  S → save SPOOF image')
print('  Q → quit')

while True:
    ret, frame = cap.read()
    if not ret:
        print('Failed to grab frame.')
        break

    display = frame.copy()
    h, w    = display.shape[:2]
    now     = time.time()

    # ── Flash overlay ─────────────────────────────────────────────────────────
    if now < flash_until:
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), flash_color, -1)
        alpha = 0.25 * (flash_until - now) / 0.4   # fade out
        cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)

        (tw, th), _ = cv2.getTextSize(flash_msg, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        tx = (w - tw) // 2
        ty = (h + th) // 2
        cv2.putText(display, flash_msg, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    # ── HUD ───────────────────────────────────────────────────────────────────
    cv2.rectangle(display, (0, 0), (w, 34), (0, 0, 0), -1)
    hud = (f'LIVE: {count_live}   SPOOF: {count_spoof}'
           f'        L=live  S=spoof  Q=quit')
    cv2.putText(display, hud, (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow('Data Capture', display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == ord('Q'):
        break

    elif key == ord('l') or key == ord('L'):
        ts       = int(time.time() * 1000)
        filename = LIVE_DIR / f'custom_{ts}.png'
        cv2.imwrite(str(filename), frame)
        count_live += 1
        flash_msg   = f'LIVE saved  ({count_live})'
        flash_color = (0, 180, 0)     # green tint
        flash_until = now + 0.4
        print(f'  [LIVE]  {filename}')

    elif key == ord('s') or key == ord('S'):
        ts       = int(time.time() * 1000)
        filename = SPOOF_DIR / f'custom_{ts}.png'
        cv2.imwrite(str(filename), frame)
        count_spoof += 1
        flash_msg   = f'SPOOF saved  ({count_spoof})'
        flash_color = (0, 0, 200)     # red tint
        flash_until = now + 0.4
        print(f'  [SPOOF] {filename}')

cap.release()
cv2.destroyAllWindows()
print(f'\nDone. Captured {count_live} live + {count_spoof} spoof images.')
print(f'  Live  → {LIVE_DIR.resolve()}')
print(f'  Spoof → {SPOOF_DIR.resolve()}')
