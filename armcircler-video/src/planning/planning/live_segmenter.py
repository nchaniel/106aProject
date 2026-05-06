import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Paths — resolve relative to this file so ROS working-dir doesn't matter
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent   # .../src/planning/planning/
_ROOT = _HERE.parents[2]                  # .../armcircler-video/

# Checkpoints live in the sam2 repo directory, not armcircler-video.
# Derive location from the installed sam2 package so the path is always correct.
try:
    import sam2 as _sam2_pkg
    _SAM2_CKPT_DIR = Path(_sam2_pkg.__file__).resolve().parent.parent / "checkpoints"
except ImportError:
    _SAM2_CKPT_DIR = _ROOT / "checkpoints"  # fallback

SAM2_CONFIGS = {
    "tiny":  ("configs/sam2.1/sam2.1_hiera_t.yaml",  str(_SAM2_CKPT_DIR / "sam2.1_hiera_tiny.pt")),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml",  str(_SAM2_CKPT_DIR / "sam2.1_hiera_small.pt")),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml",  str(_SAM2_CKPT_DIR / "sam2.1_hiera_large.pt")),
}

COLORS = [
    [0.0, 1.0, 0.0],
    [1.0, 0.3, 0.0],
    [0.2, 0.4, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [0.8, 0.5, 0.0],
    [0.5, 0.0, 0.8],
]

# Smaller resolution for faster YOLO+SAM2 inference on CPU
INFER_W, INFER_H = 480, 360


# ---------------------------------------------------------------------------
# Display helpers (adapted from segment_v2.py / segment_live.py)
# ---------------------------------------------------------------------------

def _blend_mask(image, mask, color, alpha=0.45):
    out = image.astype(float) / 255.0
    for c in range(3):
        out[:, :, c] = np.where(mask, out[:, :, c] * (1 - alpha) + color[c] * alpha, out[:, :, c])
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _draw_boxes(image, detections):
    pil = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for det in detections:
        x0, y0, x1, y1 = det["box"]
        color_rgb = tuple(int(c * 255) for c in COLORS[det["idx"] % len(COLORS)])
        draw.rectangle([x0, y0, x1, y1], outline=color_rgb, width=3)
        label = f"{det['label']} {det['conf']:.2f}"
        tw, th = 6 * len(label), 14
        draw.rectangle([x0, y0 - th - 2, x0 + tw, y0], fill=color_rgb)
        draw.text((x0 + 2, y0 - th - 1), label, fill=(0, 0, 0))
    return pil


def _build_display(frame_rgb, detections, masks):
    left = cv2.cvtColor(np.array(_draw_boxes(frame_rgb, detections)), cv2.COLOR_RGB2BGR)

    composite = frame_rgb.copy()
    for det, mask in zip(detections, masks):
        composite = _blend_mask(composite, mask, COLORS[det["idx"] % len(COLORS)])
    right = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

    display = np.hstack([left, right])
    h, w = display.shape[:2]
    cv2.putText(display, f"FPS: {0:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.line(display, (w // 2, 0), (w // 2, h), (200, 200, 200), 1)
    return display


def _build_display_with_fps(frame_rgb, detections, masks, fps):
    left = cv2.cvtColor(np.array(_draw_boxes(frame_rgb, detections)), cv2.COLOR_RGB2BGR)

    composite = frame_rgb.copy()
    for det, mask in zip(detections, masks):
        composite = _blend_mask(composite, mask, COLORS[det["idx"] % len(COLORS)])
    right = cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)

    display = np.hstack([left, right])
    h, w = display.shape[:2]
    cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.line(display, (w // 2, 0), (w // 2, h), (200, 200, 200), 1)
    return display


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _detect(yolo_model, frame_rgb, conf_thresh):
    results = yolo_model(frame_rgb, conf=conf_thresh, verbose=False)[0]
    detections = []
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        detections.append({
            "idx":   i,
            "label": results.names[cls_id],
            "conf":  float(box.conf[0]),
            "box":   xyxy,
        })
    return detections


def _run_sam2(predictor, frame_rgb, detections):
    with torch.inference_mode():
        predictor.set_image(frame_rgb)
        boxes = np.stack([det["box"].astype(float) for det in detections])
        m, _, _ = predictor.predict(box=boxes, multimask_output=False)
    return [m[i] for i in range(len(detections))]


# ---------------------------------------------------------------------------
# LiveSegmenter
# ---------------------------------------------------------------------------

class LiveSegmenter:
    """Runs YOLO+SAM2 in a background thread and shows a live cv2.imshow window.

    Call update_frame() from the ROS camera callback.
    Call stop() before rclpy.shutdown().
    """

    def __init__(self, yolo_path=None, sam2_model="tiny", conf=0.25):
        if yolo_path is None:
            yolo_path = str(_ROOT / "best.pt")

        config_file, ckpt_path = SAM2_CONFIGS[sam2_model]
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {ckpt_path}\n"
                f"Download from: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt\n"
                f"Place at: {ckpt_path}"
            )

        from ultralytics import YOLO
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        print("[LiveSegmenter] Loading YOLO ...")
        self._yolo = YOLO(yolo_path)
        self._conf = conf

        print(f"[LiveSegmenter] Loading SAM2-{sam2_model} on CPU ...")
        sam2 = build_sam2(config_file, ckpt_path, device="cpu")
        self._predictor = SAM2ImagePredictor(sam2)

        self._lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._frame = None
        self._running = True

        self._thread = threading.Thread(target=self._loop, name="live_segmenter")
        self._thread.start()
        print("[LiveSegmenter] Stream thread started.")

    def update_frame(self, bgr_frame):
        """Thread-safe: drop the latest BGR frame for the inference thread to process."""
        with self._lock:
            self._frame = bgr_frame
        self._frame_ready.set()

    def stop(self):
        self._running = False
        self._frame_ready.set()  # unblock thread if it's waiting
        self._thread.join(timeout=10.0)  # wait for inference to finish before library teardown
        cv2.destroyAllWindows()

    def _loop(self):
        fps_times = deque(maxlen=20)

        while self._running:
            self._frame_ready.wait()
            self._frame_ready.clear()

            if not self._running:
                break

            with self._lock:
                bgr = self._frame.copy() if self._frame is not None else None

            if bgr is None:
                continue

            t0 = time.perf_counter()

            # Downscale for faster inference — full-res still saved to disk by take_photo()
            small = cv2.resize(bgr, (INFER_W, INFER_H))
            frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            detections = _detect(self._yolo, frame_rgb, self._conf)
            masks = _run_sam2(self._predictor, frame_rgb, detections) if detections else []

            fps_times.append(time.perf_counter() - t0)
            fps = len(fps_times) / sum(fps_times) if fps_times else 0.0

            display = _build_display_with_fps(frame_rgb, detections, masks, fps)
            cv2.imshow("Live Segmentation  [YOLO | SAM2]", display)
            cv2.waitKey(1)
