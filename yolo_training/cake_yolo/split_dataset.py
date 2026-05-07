import os
import random
import shutil

img_train_dir = "cake_yolo/dataset/images/train"
lbl_train_dir = "cake_yolo/dataset/labels/train"

img_val_dir = "cake_yolo/dataset/images/val"
lbl_val_dir = "cake_yolo/dataset/labels/val"

val_ratio = 0.2  # move 20% to val folder

images = [f for f in os.listdir(img_train_dir)
          if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

random.shuffle(images)

val_count = int(len(images) * val_ratio)
val_images = images[:val_count]

os.makedirs(img_val_dir, exist_ok=True)
os.makedirs(lbl_val_dir, exist_ok=True)

for img in val_images:
    base = os.path.splitext(img)[0]
    lbl = base + ".txt"

    src_img = os.path.join(img_train_dir, img)
    dst_img = os.path.join(img_val_dir, img)

    shutil.move(src_img, dst_img)

    src_lbl = os.path.join(lbl_train_dir, lbl)
    dst_lbl = os.path.join(lbl_val_dir, lbl)

    if os.path.exists(src_lbl):
        shutil.move(src_lbl, dst_lbl)
    else:
        print(f"WARNING: Missing label for {img}")