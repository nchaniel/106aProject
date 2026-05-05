"""
hsv_picker.py
─────────────
Interactive helper: click on any object in your scene image to get its
HSV range. Prints ready-to-paste ObjectConfig values.

Usage:
    python hsv_picker.py scene.jpg

Controls:
    Left-click      Sample colour at that pixel (±10px neighbourhood)
    R               Reset all samples for current object
    N               Next object (saves current, starts new)
    Q / Esc         Quit and print all ranges
"""

import sys
import cv2
import numpy as np


SAMPLE_RADIUS = 10      # pixels around click to sample
HUE_MARGIN    = 8       # extra hue margin added around sampled range
SAT_MARGIN    = 30
VAL_MARGIN    = 40


def sample_region(hsv_img, cx, cy, r):
    h, w = hsv_img.shape[:2]
    x0, x1 = max(0, cx - r), min(w, cx + r)
    y0, y1 = max(0, cy - r), min(h, cy + r)
    region = hsv_img[y0:y1, x0:x1].reshape(-1, 3)
    return region


def compute_range(samples, hue_margin=HUE_MARGIN,
                  sat_margin=SAT_MARGIN, val_margin=VAL_MARGIN):
    if len(samples) == 0:
        return None, None
    h = samples[:, 0]
    s = samples[:, 1]
    v = samples[:, 2]

    h_lo = max(0,   int(h.min()) - hue_margin)
    h_hi = min(179, int(h.max()) + hue_margin)
    s_lo = max(0,   int(s.min()) - sat_margin)
    s_hi = min(255, int(s.max()) + sat_margin)
    v_lo = max(0,   int(v.min()) - val_margin)
    v_hi = min(255, int(v.max()) + val_margin)

    # Detect red hue wrap-around
    wraps = (h_lo <= 10 and h_hi >= 165)
    return (h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi), wraps


def main():
    if len(sys.argv) < 2:
        print("Usage: python hsv_picker.py <image_path>")
        sys.exit(1)

    img_bgr = cv2.imread(sys.argv[1])
    if img_bgr is None:
        print(f"Could not load image: {sys.argv[1]}")
        sys.exit(1)

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    display = img_bgr.copy()

    objects = []
    current_name = ""
    current_samples = []
    current_dots = []

    def get_name():
        nonlocal current_name
        name = input("\nEnter object name (e.g. 'red_sphere'): ").strip()
        current_name = name or f"object_{len(objects)+1}"
        print(f"  Sampling '{current_name}' — click on the object in the window.")
        print("  R=reset  N=next object  Q=quit")

    def mouse_cb(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        region = sample_region(hsv, x, y, SAMPLE_RADIUS)
        current_samples.extend(region.tolist())
        current_dots.append((x, y))
        cv2.circle(display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("HSV Picker", display)
        print(f"  Sampled at ({x},{y})  HSV≈{hsv[y,x]}  "
              f"total samples: {len(current_samples)}")

    cv2.namedWindow("HSV Picker")
    cv2.setMouseCallback("HSV Picker", mouse_cb)

    print("═" * 55)
    print("  HSV Colour Range Picker")
    print("═" * 55)
    get_name()

    while True:
        cv2.imshow("HSV Picker", display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('r') or key == ord('R'):
            current_samples.clear()
            for dot in current_dots:
                cv2.circle(display, dot, 6, (0, 0, 0), -1)
            current_dots.clear()
            display[:] = img_bgr.copy()
            print("  Reset samples for current object.")

        elif key == ord('n') or key == ord('N'):
            if current_samples:
                arr = np.array(current_samples, dtype=np.uint8)
                lo, hi, wraps = compute_range(arr)
                objects.append({
                    "name": current_name,
                    "low": lo, "high": hi, "wraps": wraps,
                    "n_samples": len(current_samples),
                })
                print(f"  Saved '{current_name}'.")
            current_samples.clear()
            current_dots.clear()
            display[:] = img_bgr.copy()
            get_name()

        elif key == ord('q') or key == 27:
            if current_samples:
                arr = np.array(current_samples, dtype=np.uint8)
                lo, hi, wraps = compute_range(arr)
                objects.append({
                    "name": current_name,
                    "low": lo, "high": hi, "wraps": wraps,
                    "n_samples": len(current_samples),
                })
            break

    cv2.destroyAllWindows()

    if not objects:
        print("No objects sampled.")
        return

    print("\n" + "═" * 55)
    print("  Copy-paste into your run script:")
    print("═" * 55 + "\n")

    for obj in objects:
        lo, hi = obj["low"], obj["high"]
        name = obj["name"]

        if obj["wraps"]:
            # Red-like colour: split into two ranges
            print(f"ObjectConfig(")
            print(f"    name        = \"{name}\",")
            print(f"    stl_path    = \"models/{name}.stl\",")
            print(f"    color_rgb   = (???, ???, ???),   # fill in your print colour")
            print(f"    # Red wraps around hue=0 — two ranges needed:")
            print(f"    hsv_low     = (0,   {lo[1]}, {lo[2]}),")
            print(f"    hsv_high    = (10,  {hi[1]}, {hi[2]}),")
            print(f"    hsv_low2    = (165, {lo[1]}, {lo[2]}),")
            print(f"    hsv_high2   = (179, {hi[1]}, {hi[2]}),")
            print(f"),\n")
        else:
            print(f"ObjectConfig(")
            print(f"    name        = \"{name}\",")
            print(f"    stl_path    = \"models/{name}.stl\",")
            print(f"    color_rgb   = (???, ???, ???),   # fill in your print colour")
            print(f"    hsv_low     = {lo},")
            print(f"    hsv_high    = {hi},")
            print(f"),\n")


if __name__ == "__main__":
    main()
