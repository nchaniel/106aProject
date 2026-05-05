"""
visualisation.py
────────────────
Three output figures saved automatically after every process_scene() call:
  1. scene_poses.png    — scene photo with XYZ axes at each estimated pose
  2. comparisons.png    — real crop | render at pose | edge overlay per object
  3. masks_debug.png    — all auto-generated colour masks for verification
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Axes overlay
# ─────────────────────────────────────────────────────────────────────────────

def draw_pose_axes(
    image: np.ndarray,
    pose_w2c: np.ndarray,
    fx: float = 525.0,
    fy: float = 525.0,
    cx: float = None,
    cy: float = None,
    axis_length: float = 0.35,
    thickness: int = 2,
    anchor_px: tuple = None,
) -> np.ndarray:
    """Draws RGB XYZ axes on the image at the object origin. X=red Y=green Z=blue.
    anchor_px: if given as (u, v), the axes are shifted so their origin lands on
    that pixel (e.g. the mask centroid) while the directions come from pose_w2c."""
    H, W = image.shape[:2]
    cx = cx or W / 2;  cy = cy or H / 2
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float64)
    R = pose_w2c[:3,:3];  t = pose_w2c[:3,3]
    rvec, _ = cv2.Rodrigues(R)

    def proj(pts):
        p, _ = cv2.projectPoints(np.float32(pts), rvec, t, K, None)
        return p.reshape(-1,2).astype(int)

    o  = proj([[0,0,0]])[0]
    px = proj([[axis_length,0,0]])[0]
    py = proj([[0,axis_length,0]])[0]
    pz = proj([[0,0,axis_length]])[0]

    # Optionally pin the origin to the mask centroid so axes start on the object
    if anchor_px is not None:
        shift = np.array(anchor_px, dtype=int) - o
        o  = np.array(anchor_px, dtype=int)
        px = px + shift
        py = py + shift
        pz = pz + shift

    vis = image.copy()
    cv2.arrowedLine(vis, tuple(o), tuple(px), (220,50,50),  thickness, tipLength=0.25)
    cv2.arrowedLine(vis, tuple(o), tuple(py), (50,200,50),  thickness, tipLength=0.25)
    cv2.arrowedLine(vis, tuple(o), tuple(pz), (50,100,220), thickness, tipLength=0.25)
    cv2.circle(vis, tuple(o), 4, (255,255,255), -1)
    return vis


def overlay_mask(image: np.ndarray, mask: np.ndarray,
                 color=(0,220,120), alpha=0.30) -> np.ndarray:
    vis = image.copy()
    c = np.array(color, dtype=np.uint8)
    vis[mask>0] = ((1-alpha)*vis[mask>0].astype(np.float32)
                   + alpha*c).astype(np.uint8)
    contours, _ = cv2.findContours((mask>0).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, color, 2)
    return vis


# ─────────────────────────────────────────────────────────────────────────────
# Per-object comparison strip
# ─────────────────────────────────────────────────────────────────────────────

def make_comparison_strip(
    real_crop: np.ndarray,
    render: np.ndarray,
    label: str,
    result: dict,
    size: int = 224,
) -> np.ndarray:
    """
    Returns a (size+28, size*3, 3) image:
    real crop | render at estimated pose | edge overlay
    Green edges = query,  Red edges = render
    """
    q = cv2.resize(real_crop, (size, size))
    r = cv2.resize(render,    (size, size))

    q_edges = cv2.Canny(cv2.cvtColor(q, cv2.COLOR_RGB2GRAY), 50, 150)
    r_edges = cv2.Canny(cv2.cvtColor(r, cv2.COLOR_RGB2GRAY), 50, 150)

    overlay = q.copy()
    overlay[r_edges > 0] = [255, 80,  80]   # render  → red
    overlay[q_edges > 0] = [80,  255, 80]   # query   → green

    panel = np.concatenate([q, r, overlay], axis=1)

    conf = result.get("confidence", 0.0)
    sym  = result.get("is_symmetric", False)
    if sym:
        t = result["translation"]
        info = f"pos=({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})  [rotation skipped — symmetric]"
    else:
        e = result.get("euler_angles_zyx_deg", np.zeros(3))
        info = f"Z={e[0]:.1f}° Y={e[1]:.1f}° X={e[2]:.1f}°"

    bar = np.zeros((28, panel.shape[1], 3), dtype=np.uint8)
    text = f"{label}  |  {info}  |  conf={conf:.2f}"
    cv2.putText(bar, text, (6,19), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (220,220,220), 1, cv2.LINE_AA)
    return np.concatenate([panel, bar], axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Three output figures
# ─────────────────────────────────────────────────────────────────────────────

def save_all_figures(
    image: np.ndarray,
    results: list[dict],
    refiners: dict,             # {name: PoseRefiner}
    output_dir: str = ".",
    fx: float = 525.0,
    fy: float = 525.0,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _save_scene_poses(image, results, out, fx, fy)
    _save_comparisons(image, results, refiners, out, fx, fy)
    _save_masks_debug(image, results, out)


def _save_scene_poses(image, results, out, fx, fy):
    vis = image.copy()
    H, W = image.shape[:2]
    cx_img, cy_img = W / 2.0, H / 2.0
    colours = [
        (0,220,120), (220,180,0), (0,160,220),
        (220,80,80), (160,0,220), (80,220,160),
    ]
    for i, r in enumerate(results):
        if "error" in r or r.get("pose_w2c") is None:
            continue
        col = colours[i % len(colours)]
        mask = r.get("mask")
        if mask is not None:
            vis = overlay_mask(vis, mask, color=col)

        # Compute mask centroid — the true 2D projection of the 3D centre
        anchor_px = None
        com_text = None
        if mask is not None:
            ys, xs = np.where(mask > 0)
            if len(ys):
                u = int(xs.mean())
                v = int(ys.mean())
                anchor_px = (u, v)

                # Metric 3D centre from apparent size + known physical diameter
                diameter_m = r.get("diameter_m")
                if diameter_m:
                    area_px = int(np.count_nonzero(mask))
                    r_px = max(1.0, float(np.sqrt(area_px / np.pi)))
                    z_m = fx * (diameter_m / 2.0) / r_px
                    x_m = (u - cx_img) / fx * z_m
                    y_m = (v - cy_img) / fy * z_m
                    com_text = f"({x_m:.3f},{y_m:.3f},{z_m:.3f})m"

        vis = draw_pose_axes(vis, r["pose_w2c"], fx=fx, fy=fy, anchor_px=anchor_px)

        if anchor_px is not None:
            u, v = anchor_px
            label = r["object_name"]
            if r.get("instance_idx", 0) > 0:
                label += f"[{r['instance_idx']}]"
            ys_arr = np.where(mask > 0)[0]
            cv2.putText(vis, label, (u, max(0, int(ys_arr.min()) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
            # Yellow dot at 3D centre of mass
            cv2.circle(vis, anchor_px, 7, (0, 0, 0), -1)
            cv2.circle(vis, anchor_px, 5, (255, 255, 0), -1)
            if com_text:
                cv2.putText(vis, com_text, (u + 8, v + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, cv2.LINE_AA)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.imshow(vis)
    ax.set_title("Estimated Poses  —  X=red  Y=green  Z=blue", fontsize=11)
    ax.axis("off")
    # Legend
    patches = [
        mpatches.Patch(color=(1,0,0), label="X axis"),
        mpatches.Patch(color=(0,0.8,0), label="Y axis"),
        mpatches.Patch(color=(0,0.4,0.9), label="Z axis"),
    ]
    ax.legend(handles=patches, loc="upper right", fontsize=8)
    plt.tight_layout()
    path = str(out / "scene_poses.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved → {path}")


def _save_comparisons(image, results, refiners, out, fx, fy):
    strips = []
    for r in results:
        if "error" in r:
            continue
        name = r["object_name"]
        mask = r.get("mask")
        pose = r.get("pose_w2c")
        if mask is None or pose is None or name not in refiners:
            continue

        # Crop real image to mask bbox
        ys, xs = np.where(mask > 0)
        if not len(ys):
            continue
        pad = 10
        y0 = max(0, ys.min()-pad);  y1 = min(image.shape[0], ys.max()+pad)
        x0 = max(0, xs.min()-pad);  x1 = min(image.shape[1], xs.max()+pad)
        crop = image[y0:y1, x0:x1]

        render = refiners[name].render_at_pose(pose, fx=fx, fy=fy)
        label = name
        if r.get("instance_idx", 0) > 0:
            label += f"[{r['instance_idx']}]"
        strips.append(make_comparison_strip(crop, render, label, r))

    if not strips:
        return

    fig, axes = plt.subplots(len(strips), 1,
                             figsize=(10, 3.2 * len(strips)))
    if len(strips) == 1:
        axes = [axes]
    for ax, strip in zip(axes, strips):
        ax.imshow(strip)
        ax.set_title("Real crop  |  Render at pose  |  Edge overlay "
                     "(green=real, red=render)", fontsize=8, pad=3)
        ax.axis("off")
    plt.tight_layout()
    path = str(out / "comparisons.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved → {path}")


def _save_masks_debug(image, results, out):
    masks_found = [(r["object_name"], r.get("instance_idx",0), r["mask"])
                   for r in results if "mask" in r and r["mask"] is not None]
    if not masks_found:
        return

    n = len(masks_found)
    fig, axes = plt.subplots(1, n+1, figsize=(4*(n+1), 4))
    if n == 0:
        return
    axes[0].imshow(image)
    axes[0].set_title("Original scene", fontsize=9)
    axes[0].axis("off")

    for ax, (name, idx, mask) in zip(axes[1:], masks_found):
        label = name if idx == 0 else f"{name}[{idx}]"
        vis = image.copy()
        vis = overlay_mask(vis, mask)
        ax.imshow(vis)
        ax.set_title(f"Mask: {label}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    path = str(out / "masks_debug.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_results_table(results: list[dict]):
    print("\n" + "═"*87)
    print(f"{'Object':<20} {'Z°':>7} {'Y°':>7} {'X°':>7}  "
          f"{'tx':>7} {'ty':>7} {'tz':>7}  {'dist(m)':>8}  {'conf':>6}")
    print("─"*87)
    for r in results:
        name = r.get("object_name","?")
        idx  = r.get("instance_idx", 0)
        label = name if idx == 0 else f"{name}[{idx}]"
        if "error" in r:
            print(f"{label:<20}  ERROR: {r['error']}")
            continue
        t = r.get("position_m", r["translation"])
        dist = float(np.linalg.norm(t))
        c = r.get("confidence", 0.0)
        if r.get("is_symmetric"):
            print(f"{label:<20} {'—':>7} {'—':>7} {'—':>7}  "
                  f"{t[0]:>7.3f} {t[1]:>7.3f} {t[2]:>7.3f}  {dist:>8.4f}  {c:>6.2f}  "
                  f"(symmetric)")
        else:
            e = r["euler_angles_zyx_deg"]
            print(f"{label:<20} {e[0]:>7.1f} {e[1]:>7.1f} {e[2]:>7.1f}  "
                  f"{t[0]:>7.3f} {t[1]:>7.3f} {t[2]:>7.3f}  {dist:>8.4f}  {c:>6.2f}")
    print("═"*87)
