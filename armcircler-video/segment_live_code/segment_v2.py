"""
V2 YOLO → SAM2 Automated Segmentation
---------------------------------------
YOLO detects every object in the image and outputs bounding boxes.
Each box is passed directly into SAM2 as a prompt to get a precise mask.

Usage:
    python segment_v2.py --image path/to/image.jpg [options]

Options:
    --model   SAM2 size: tiny | small | large  (default: tiny)
    --yolo    YOLO model: yolov8n | yolov8s | yolov8m  (default: yolov8n)
    --conf    YOLO confidence threshold 0-1  (default: 0.25)
    --classes space-separated class names to keep, e.g. --classes apple banana
              (default: keep all detections)

Outputs (saved next to the input image):
    <name>_boxes.png          YOLO bounding boxes drawn on image
    <name>_all_overlay.png    all SAM2 masks composited on image
    <name>_<class>_<i>_mask.png     binary mask per object
    <name>_<class>_<i>_overlay.png  per-object mask overlay

First run: YOLO weights (~6 MB) download automatically from ultralytics.
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM2_CONFIGS = {
    "tiny":  ("configs/sam2.1/sam2.1_hiera_t.yaml",  "checkpoints/sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml",  "checkpoints/sam2.1_hiera_small.pt"),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml",  "checkpoints/sam2.1_hiera_large.pt"),
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


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _blend_mask(image, mask, color, alpha=0.5):
    out = image.astype(float) / 255.0
    for c in range(3):
        out[:, :, c] = np.where(mask, out[:, :, c] * (1 - alpha) + color[c] * alpha, out[:, :, c])
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _draw_boxes(image, detections):
    """Draw YOLO boxes + labels on a copy of image. Returns PIL Image."""
    pil = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for det in detections:
        x0, y0, x1, y1 = det["box"]
        color_01 = COLORS[det["idx"] % len(COLORS)]
        color_rgb = tuple(int(c * 255) for c in color_01)
        draw.rectangle([x0, y0, x1, y1], outline=color_rgb, width=3)
        label = f"{det['label']} {det['conf']:.2f}"
        # small filled rect behind text
        tw, th = 6 * len(label), 14
        draw.rectangle([x0, y0 - th - 2, x0 + tw, y0], fill=color_rgb)
        draw.text((x0 + 2, y0 - th - 1), label, fill=(0, 0, 0))
    return pil


def _show_results(image, detections, masks):
    """Display YOLO boxes and SAM2 masks side-by-side in a matplotlib window."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor("#1e1e1e")

    # Left: YOLO boxes
    box_img = _draw_boxes(image, detections)
    axes[0].imshow(box_img)
    axes[0].set_title("YOLO detections", color="white")
    axes[0].axis("off")

    # Right: all SAM2 masks composited
    composite = image.copy()
    legend_patches = []
    for det, mask in zip(detections, masks):
        color = COLORS[det["idx"] % len(COLORS)]
        composite = _blend_mask(composite, mask, color, alpha=0.45)
        patch = mpatches.Patch(color=color, label=f"{det['label']} ({det['conf']:.2f})")
        legend_patches.append(patch)

    axes[1].imshow(composite)
    axes[1].set_title("SAM2 masks", color="white")
    axes[1].axis("off")
    if legend_patches:
        axes[1].legend(handles=legend_patches, loc="upper right",
                       fontsize=8, framealpha=0.7)

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------
# Core pipeline
# -----------------------------------------------------------------------

def run_yolo(image, yolo_name, conf_thresh, keep_classes):
    """Run YOLO and return a list of detection dicts."""
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run:  pip install ultralytics")

    print(f"Running YOLO ({yolo_name}, conf≥{conf_thresh}) ...")
    yolo = YOLO(f"{yolo_name}.pt")  # auto-downloads weights on first run
    results = yolo(image, conf=conf_thresh, verbose=False)[0]

    detections = []
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        if keep_classes and label not in keep_classes:
            continue
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        detections.append({
            "idx":   i,
            "label": label,
            "conf":  conf,
            "box":   xyxy,       # [x0, y0, x1, y1]
        })
        print(f"  [{i}] {label:20s}  conf={conf:.2f}  box={xyxy.tolist()}")

    if not detections:
        print("No detections found. Try lowering --conf or check --classes filter.")
    return detections


def run_sam2(predictor, detections):
    """For each YOLO box, run SAM2 and return list of boolean masks (HxW)."""
    masks = []
    with torch.inference_mode():
        for det in detections:
            box = det["box"].astype(float)
            m, scores, _ = predictor.predict(
                box=box,
                multimask_output=False,  # single best mask for box prompts
            )
            mask = m[0]  # HxW boolean
            masks.append(mask)
            print(f"  [{det['idx']}] {det['label']:20s}  SAM2 score={scores[0]:.3f}  "
                  f"pixels={mask.sum():,}")
    return masks


def save_outputs(image, image_path, detections, masks):
    stem = os.path.splitext(image_path)[0]

    # YOLO boxes image
    box_img = _draw_boxes(image, detections)
    box_path = f"{stem}_boxes.png"
    box_img.save(box_path)
    print(f"Saved  {box_path}")

    # All masks composited
    composite = image.copy()
    for det, mask in zip(detections, masks):
        color = COLORS[det["idx"] % len(COLORS)]
        composite = _blend_mask(composite, mask, color, alpha=0.45)
    all_path = f"{stem}_all_overlay.png"
    Image.fromarray(composite).save(all_path)
    print(f"Saved  {all_path}")

    # Per-object mask + overlay
    label_counts = {}
    for det, mask in zip(detections, masks):
        lbl = det["label"].replace(" ", "_")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        suffix = f"{lbl}_{label_counts[lbl]}"

        mask_arr = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_arr).save(f"{stem}_{suffix}_mask.png")

        color = COLORS[det["idx"] % len(COLORS)]
        overlay = _blend_mask(image, mask, color, alpha=0.5)
        Image.fromarray(overlay).save(f"{stem}_{suffix}_overlay.png")

        print(f"Saved  {stem}_{suffix}_mask.png  +  {stem}_{suffix}_overlay.png")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="V2: YOLO detects objects → SAM2 refines masks automatically."
    )
    parser.add_argument("--image",   required=True, help="Path to input RGB image")
    parser.add_argument("--model",   choices=["tiny", "small", "large"], default="tiny",
                        help="SAM2 model size (default: tiny)")
    parser.add_argument("--yolo",    default="yolov8n",
                        help="YOLO weights name: yolov8n | yolov8s | yolov8m (default: yolov8n)")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--classes", nargs="*", default=None,
                        help="Only keep these YOLO class names, e.g. --classes apple banana")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        sys.exit(f"Image not found: {args.image}")

    config_file, ckpt_path = SAM2_CONFIGS[args.model]
    if not os.path.isfile(ckpt_path):
        sys.exit(
            f"SAM2 checkpoint not found: {ckpt_path}\n"
            f"Download with PowerShell:\n"
            f"  Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt "
            f"-OutFile checkpoints\\sam2.1_hiera_tiny.pt"
        )

    # Load image
    image = np.array(Image.open(args.image).convert("RGB"))
    print(f"Image: {image.shape[1]}x{image.shape[0]} px")

    # Step 1: YOLO detection
    detections = run_yolo(image, args.yolo, args.conf, set(args.classes) if args.classes else None)
    if not detections:
        sys.exit(0)

    # Step 2: SAM2 masking
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading SAM2-{args.model} on {device} ...")
    model = build_sam2(config_file, ckpt_path, device=device)
    predictor = SAM2ImagePredictor(model)

    print("Embedding image ...")
    with torch.inference_mode():
        predictor.set_image(image)

    print("Running SAM2 on each box ...")
    masks = run_sam2(predictor, detections)

    # Step 3: Save and display
    print("\nSaving outputs ...")
    save_outputs(image, args.image, detections, masks)

    print(f"\nDone. {len(masks)} object(s) segmented.")
    _show_results(image, detections, masks)


if __name__ == "__main__":
    main()
