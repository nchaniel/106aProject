import os
import re
import shutil
from urllib.parse import unquote

# this will not run in the main repository until these are fixed; it runs on my computer only
LABEL_DIR = r"C:\Users\natha\Documents\MEC106A\cake_yolo\broken_labels"
IMAGE_DIR = r"C:\Users\natha\Documents\MEC106A\cake_yolo\jpg_images_new"
OUTPUT_DIR = r"C:\Users\natha\Documents\MEC106A\cake_yolo\repaired_dataset"

OUT_IMAGES = os.path.join(OUTPUT_DIR, "images")
OUT_LABELS = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(OUT_IMAGES, exist_ok=True)
os.makedirs(OUT_LABELS, exist_ok=True)


# Supports:
# IMG_1234
# captured_image_12
# captured_image_12 (1)
# Also works with URL-encoded names like:
# captured_image_12%20%281%29
# necessary due to how images are named from phone photos and from robot photos

PATTERN = r"(IMG_\d+|captured_image_\d+(?: \(\d+\))?)"

# build image lookup table

image_lookup = {}

for img_file in os.listdir(IMAGE_DIR):

    lower = img_file.lower()

    if not lower.endswith((".jpg", ".jpeg", ".png")):
        continue

    # decode URL-encoded filenames
    decoded_name = unquote(img_file)

    match = re.search(PATTERN, decoded_name)

    if not match:
        print(f"SKIP IMAGE: {img_file}")
        continue

    base = match.group(1)

    full_path = os.path.join(IMAGE_DIR, img_file)

    image_lookup[base] = (
        full_path,
        os.path.splitext(img_file)[1]
    )

# process labels

paired = 0
missing = 0
skipped = 0

for label_file in os.listdir(LABEL_DIR):

    if not label_file.endswith(".txt"):
        continue

    # decode URL-encoded filenames
    decoded_name = unquote(label_file)

    match = re.search(PATTERN, decoded_name)

    if not match:
        print(f"SKIP LABEL: {label_file}")
        skipped += 1
        continue

    base = match.group(1)

    # find matching image
    if base not in image_lookup:
        print(f"MISSING IMAGE: {base}")
        missing += 1
        continue

    img_path, ext = image_lookup[base]

    # make filename Windows-safe and unique
    safe_base = base.replace(" ", "_")
    safe_base = safe_base.replace("(", "")
    safe_base = safe_base.replace(")", "")

    new_img_name = safe_base + ext
    new_label_name = safe_base + ".txt"

    out_img_path = os.path.join(OUT_IMAGES, new_img_name)
    out_label_path = os.path.join(OUT_LABELS, new_label_name)

    # copy image and label
    shutil.copy2(
        img_path,
        out_img_path
    )

    shutil.copy2(
        os.path.join(LABEL_DIR, label_file),
        out_label_path
    )

    paired += 1
    print(f"PAIRED: {base}")

# output feedback

print("\nDONE")
print(f"Paired: {paired}")
print(f"Missing: {missing}")
print(f"Skipped: {skipped}")
print(f"Output: {OUTPUT_DIR}")