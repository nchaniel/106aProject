"""
YOLO & SAM2 Segmentation


Takes in images from: /armcircler/captured_images \
output: /armcircler/segmented

Current Models:
    SAM2: tiny
    YOLO: best.pt (custom trained)
    --conf        YOLO confidence threshold 0-1  (default: 0.25)
    --classes     Space-separated class names to keep, e.g. --classes person cup

Output layout for each image <stem>:
    <output_dir>/<stem>/<stem>_boxes.png
    <output_dir>/<stem>/<stem>_all_overlay.png
    <output_dir>/<stem>/<stem>_<class>_<i>_mask.png
    <output_dir>/<stem>/<stem>_<class>_<i>_overlay.png
"""

import argparse
import os
import sys
import glob

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# figure out where the sam2 repo folder is relative to this script file,
# so paths work no matter what directory you run from or which machine you're on
_HERE = os.path.dirname(os.path.abspath(__file__))
SAM2_DIR = _HERE if os.path.isdir(os.path.join(_HERE, "checkpoints")) else os.path.join(_HERE, "..", "sam2")

# config_file is a Hydra config name (resolved internally via pkg://sam2, not a filesystem path)
# ckpt_path is a real filesystem path — we make it absolute using SAM2_DIR
BEST_PT = os.path.join(_HERE, "best.pt")

SAM2_CONFIGS = {
    "tiny":  ("configs/sam2.1/sam2.1_hiera_t.yaml",  os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_tiny.pt")),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml",  os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_small.pt")),
    "large": ("configs/sam2.1/sam2.1_hiera_l.yaml",  os.path.join(SAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")),
}

# 8 distinct colors (RGB 0-1) so each detected object gets its own color in the overlay
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


def _blend_mask(image, mask, color, alpha=0.5):
    # takes the raw image and paints the mask region with a semi-transparent color
    # alpha controls how opaque the overlay is — 0.5 means 50% original, 50% color
    out = image.astype(float) / 255.0
    for c in range(3):  # loop over R, G, B channels separately
        out[:, :, c] = np.where(mask, out[:, :, c] * (1 - alpha) + color[c] * alpha, out[:, :, c])
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def _draw_boxes(image, detections):
    # draws the YOLO bounding boxes and labels directly onto a copy of the image
    pil = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(pil)
    for det in detections:
        x0, y0, x1, y1 = det["box"]
        # each object gets a color from our list, cycling if there are more than 8
        color_01 = COLORS[det["idx"] % len(COLORS)]
        color_rgb = tuple(int(c * 255) for c in color_01)
        draw.rectangle([x0, y0, x1, y1], outline=color_rgb, width=3)
        label = f"{det['label']} {det['conf']:.2f}"
        # draw a filled rectangle behind the text so it's readable over any background
        tw, th = 6 * len(label), 14
        draw.rectangle([x0, y0 - th - 2, x0 + tw, y0], fill=color_rgb)
        draw.text((x0 + 2, y0 - th - 1), label, fill=(0, 0, 0))
    return pil


def run_yolo(yolo_model, image, conf_thresh, keep_classes):
    # run YOLO on one image — we pass in the already-loaded model so we don't reload it each time
    results = yolo_model(image, conf=conf_thresh, verbose=False)[0]
    detections = []
    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        label = results.names[cls_id]
        # if the user specified --classes, skip anything not in that list
        if keep_classes and label not in keep_classes:
            continue
        xyxy = box.xyxy[0].cpu().numpy().astype(int)  # bounding box as [x0, y0, x1, y1]
        conf = float(box.conf[0])
        detections.append({
            "idx":   i,
            "label": label,
            "conf":  conf,
            "box":   xyxy,
        })
        print(f"    [{i}] {label:20s}  conf={conf:.2f}  box={xyxy.tolist()}")
    return detections


def run_sam2(predictor, detections):
    # for each YOLO bounding box, ask SAM2 to produce a pixel-level mask
    # we already called predictor.set_image() before this so SAM2 has the image embedded
    masks = []
    with torch.inference_mode():  # no gradients needed, saves memory
        for det in detections:
            box = det["box"].astype(float)
            # multimask_output=False means give us the single best mask, not 3 candidates
            m, scores, _ = predictor.predict(box=box, multimask_output=False)
            mask = m[0]  # boolean HxW array — True where the object is
            masks.append(mask)
            print(f"    [{det['idx']}] {det['label']:20s}  SAM2 score={scores[0]:.3f}  pixels={mask.sum():,}")
    return masks


def save_outputs(image, stem, detections, masks, out_dir):
    # saves all the output images for one input image into its own subfolder
    os.makedirs(out_dir, exist_ok=True)

    # first save the YOLO boxes image so we can see what YOLO detected
    box_img = _draw_boxes(image, detections)
    box_img.save(os.path.join(out_dir, f"{stem}_boxes.png"))

    # then build the composite overlay — all SAM2 masks on top of the original image at once
    composite = image.copy()
    for det, mask in zip(detections, masks):
        composite = _blend_mask(composite, mask, COLORS[det["idx"] % len(COLORS)], alpha=0.45)
    Image.fromarray(composite).save(os.path.join(out_dir, f"{stem}_all_overlay.png"))

    # also save individual mask + overlay per object so we can look at each one in isolation
    label_counts = {}  # tracks how many of each class we've seen (e.g. two cups → cup_1, cup_2)
    for det, mask in zip(detections, masks):
        lbl = det["label"].replace(" ", "_")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        suffix = f"{lbl}_{label_counts[lbl]}"

        # binary mask image — white where the object is, black everywhere else
        mask_arr = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_arr).save(os.path.join(out_dir, f"{stem}_{suffix}_mask.png"))

        # colored overlay just for this one object
        color = COLORS[det["idx"] % len(COLORS)]
        overlay = _blend_mask(image, mask, color, alpha=0.5)
        Image.fromarray(overlay).save(os.path.join(out_dir, f"{stem}_{suffix}_overlay.png"))


def collect_images(input_dir):
    # grab all jpg/jpeg/png files in the folder, handling both upper and lowercase extensions
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(input_dir, ext)))
    # sort and deduplicate in case any extension matched twice
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(
        description="Batch YOLO→SAM2 segmentation for a folder of images."
    )
    # these are the same knobs as segment_v2.py, just with an input folder instead of a single image
    parser.add_argument("--input_dir",  default="../armcircler/captured_images",
                        help="Folder of input images (default: ../armcircler/captured_images)")
    parser.add_argument("--output_dir", default="../armcircler/segmented",
                        help="Output root folder (default: ../armcircler/segmented)")
    parser.add_argument("--model",   choices=["tiny", "small", "large"], default="tiny")
    parser.add_argument("--yolo",    default=BEST_PT)
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--classes", nargs="*", default=None)
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"Input directory not found: {args.input_dir}")

    image_paths = collect_images(args.input_dir)
    if not image_paths:
        sys.exit(f"No images found in: {args.input_dir}")

    # check the checkpoint exists before doing any heavy loading
    config_file, ckpt_path = SAM2_CONFIGS[args.model]
    if not os.path.isfile(ckpt_path):
        sys.exit(
            f"SAM2 checkpoint not found: {ckpt_path}\n"
            f"Download with PowerShell:\n"
            f"  Invoke-WebRequest -Uri https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt "
            f"-OutFile checkpoints\\sam2.1_hiera_tiny.pt"
        )

    keep_classes = set(args.classes) if args.classes else None

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install ultralytics")

    # load both models once here — this is the main advantage over running segment_v2.py in a loop,
    # which would reload the models from disk for every single image
    print(f"Loading YOLO ({args.yolo}) ...")
    yolo_model = YOLO(args.yolo)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM2-{args.model} on {device} ...")
    sam2_model = build_sam2(config_file, ckpt_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    print(f"\nFound {len(image_paths)} image(s) in {args.input_dir}")
    print(f"Outputs → {args.output_dir}\n")

    skipped = 0
    for i, img_path in enumerate(image_paths, 1):
        stem = os.path.splitext(os.path.basename(img_path))[0]  # filename without extension
        print(f"[{i}/{len(image_paths)}] {stem}")

        image = np.array(Image.open(img_path).convert("RGB"))

        # step 1: YOLO tells us where the objects are (bounding boxes)
        detections = run_yolo(yolo_model, image, args.conf, keep_classes)
        if not detections:
            print("  No detections — skipping.\n")
            skipped += 1
            continue

        # step 2: give SAM2 the image so it can compute its internal embedding
        # this is separate from predicting masks — SAM2 needs to "see" the image first
        with torch.inference_mode():
            predictor.set_image(image)

        # step 3: for each bounding box from YOLO, SAM2 produces a precise pixel mask
        masks = run_sam2(predictor, detections)

        # step 4: save everything into its own subfolder under the output directory
        img_out_dir = os.path.join(args.output_dir, stem)
        save_outputs(image, stem, detections, masks, img_out_dir)
        print(f"  Saved {len(masks)} mask(s) to {img_out_dir}\n")

    processed = len(image_paths) - skipped
    print(f"Done. {processed}/{len(image_paths)} image(s) segmented, {skipped} skipped (no detections).")


if __name__ == "__main__":
    main()
