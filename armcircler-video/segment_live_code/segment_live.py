"""
Live YOLO → SAM2 Webcam Feed
-----------------------------
Captures frames from a webcam, runs YOLO detection, then SAM2 masking in real time.
A frame-skip optimization reuses the previous SAM2 masks when YOLO bounding boxes
haven't changed significantly, keeping the loop fast on CPU.

Usage:
    python segment_live.py [options]

Options:
    --camera          Webcam index (default: 0)
    --model           SAM2 size: tiny | small | large  (default: tiny)
    --yolo            YOLO weights name or path to .pt file  (default: yolov8n)
    --conf            YOLO confidence threshold 0-1  (default: 0.25)
    --skip-threshold  IoU threshold above which SAM2 is skipped  (default: 0.85)
    --classes         Space-separated class names to keep (default: all)

Keys while running:
    q  quit
    s  save current display frame to disk
"""

import argparse
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_v2 import _blend_mask, _draw_boxes, COLORS, SAM2_CONFIGS


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def auto_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Frame-skip helpers
# ---------------------------------------------------------------------------

def _box_iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def boxes_stable(prev, curr, threshold):
    if len(prev) != len(curr):
        return False
    if not curr:
        return True
    return all(_box_iou(p["box"], c["box"]) >= threshold for p, c in zip(prev, curr))


# ---------------------------------------------------------------------------
# Per-frame inference
# ---------------------------------------------------------------------------

def detect(yolo_model, frame_rgb, conf_thresh, keep_classes):
    results = yolo_model(frame_rgb, conf=conf_thresh, verbose=False)[0]
    detections = []
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        if keep_classes and label not in keep_classes:
            continue
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        detections.append({
            "idx": i,
            "label": label,
            "conf": float(box.conf[0]),
            "box": xyxy,
        })
    return detections


def run_sam2_masks(predictor, frame_rgb, detections):
    with torch.inference_mode():
        predictor.set_image(frame_rgb)
        boxes = np.stack([det["box"].astype(float) for det in detections])  # Nx4
        m, _, _ = predictor.predict(box=boxes, multimask_output=False)
        masks = [m[i] for i in range(len(detections))]
    return masks


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def build_display(frame_rgb, detections, masks):
    box_pil = _draw_boxes(frame_rgb, detections)
    left = cv2.cvtColor(np.array(box_pil), cv2.COLOR_RGB2BGR)

    composite = frame_rgb.copy()
    for det, mask in zip(detections, masks):
        color = COLORS[det["idx"] % len(COLORS)]
        composite = _blend_mask(composite, mask, color, alpha=0.45)
    right = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

    return np.hstack([left, right])


def overlay_hud(display, fps, sam2_cached):
    h, w = display.shape[:2]
    cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    sam_text = "SAM2: CACHED" if sam2_cached else "SAM2: NEW"
    sam_color = (0, 200, 0) if sam2_cached else (0, 120, 255)
    cv2.putText(display, sam_text, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, sam_color, 2)
    cv2.line(display, (w // 2, 0), (w // 2, h), (200, 200, 200), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Live YOLO → SAM2 webcam segmentation."
    )
    parser.add_argument("--camera", type=int, default=0,
                        help="Webcam index (default: 0)")
    parser.add_argument("--model", choices=["tiny", "small", "large"], default="tiny",
                        help="SAM2 model size (default: tiny)")
    parser.add_argument("--yolo", default="best",
                        help="YOLO weights name or path (default: best)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--skip-threshold", type=float, default=0.85,
                        help="IoU above which SAM2 is skipped (default: 0.85)")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="Only keep these YOLO class names")
    args = parser.parse_args()

    keep_classes = set(args.classes) if args.classes else None

    config_file, ckpt_path = SAM2_CONFIGS[args.model]
    if not os.path.isfile(ckpt_path):
        sys.exit(
            f"SAM2 checkpoint not found: {ckpt_path}\n"
            f"Download with PowerShell:\n"
            f"  Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt "
            f"-OutFile checkpoints\\sam2.1_hiera_tiny.pt"
        )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.camera}.")

    # Normalize YOLO path: append .pt only for bare model names
    yolo_path = args.yolo
    if not yolo_path.endswith(".pt") and os.sep not in yolo_path and "/" not in yolo_path:
        yolo_path = yolo_path + ".pt"

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run:  pip install ultralytics")

    device = auto_device()
    print(f"Device: {device}")

    print(f"Loading YOLO ({yolo_path}) ...")
    yolo_model = YOLO(yolo_path)

    print(f"Loading SAM2-{args.model} ...")
    sam2_model = build_sam2(config_file, ckpt_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    prev_boxes = []
    prev_masks = []
    fps_times = deque(maxlen=30)
    frame_count = 0

    print("Running — press 'q' to quit, 's' to save frame.")

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (640, 480))
        t0 = time.perf_counter()

        detections = detect(yolo_model, frame_rgb, args.conf, keep_classes)
        cached = boxes_stable(prev_boxes, detections, args.skip_threshold)

        if not cached:
            masks = run_sam2_masks(predictor, frame_rgb, detections) if detections else []
            prev_boxes = detections
            prev_masks = masks
        else:
            masks = prev_masks

        display = build_display(frame_rgb, detections, masks)
        fps_times.append(time.perf_counter() - t0)
        fps = len(fps_times) / sum(fps_times) if fps_times else 0.0
        overlay_hud(display, fps, cached)

        cv2.imshow("segment_live  [q=quit  s=save]", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"frame_{frame_count:05d}.png"
            cv2.imwrite(fname, display)
            print(f"Saved {fname}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
