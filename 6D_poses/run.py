"""
run.py
──────
Configure objects and camera parameters below, then either:

    python run.py                          # standalone
    from run import run_pose_estimation    # as a library

Returns a list of result dicts — one per detected object instance.
"""

import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from pose_estimator import PoseEstimator
from visualisation import save_all_figures, print_results_table
from constants import (
    OBJECTS, CAMERA_FX, CAMERA_FY, SCENE_IDX, OUTPUT_DIR,
    DB_DIR, N_VIEWS, SEG_NAME_MAP, T_CAM_FROM_EE, NUM_POSES_PER_LAYER
)

def _pose7_to_ee_T_world(p) -> np.ndarray:
    """[x, y, z, qx, qy, qz, qw] → 4×4 ee_T_world."""
    p = np.asarray(p, dtype=float)
    R = Rotation.from_quat(p[3:]).as_matrix()
    world_T_ee = np.eye(4)
    world_T_ee[:3, :3] = R
    world_T_ee[:3,  3] = p[:3]
    return np.linalg.inv(world_T_ee)


def _load_poses(npy_path: str):
    """
    Return (home_ee_T_world, high_arc_list, low_arc_list).

    Image/pose layout:
      captured_image_1      → home pose  (elements 0-6, stored as 7 scalars)
      captured_image_2-21   → high arc   (elements 7-26, trajectory[0:20])
      captured_image_22-41  → low arc    (elements 27-46, trajectory[20:40])
      captured_image_42     → vertical   (no matching pose, skipped)
    """
    raw = np.load(npy_path, allow_pickle=True)
    home = _pose7_to_ee_T_world([float(raw[k]) for k in range(7)])
    traj = [_pose7_to_ee_T_world(raw[i]) for i in range(7, len(raw))]
    return home, traj[:20], traj[20:]

# ═══════════════════════════════════════════════════════════════════════════
# CAMERA MOUNT GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════
# Fixed transform from the end-effector frame to the camera frame.
# Camera is mounted 3 cm in front (+x) and 13.5 cm above (-z) the
# end-effector, with the same orientation.  Adjust axis signs to match
# your robot's convention, and add a rotation block if the camera is
# tilted relative to the end-effector.

def cam_T_world_from_ee(ee_T_world: np.ndarray) -> np.ndarray:
    """Convert a world-to-end-effector pose to a world-to-camera pose."""
    return np.linalg.inv(T_CAM_FROM_EE) @ ee_T_world


def _add_base_link_poses(results: list[dict], cam_T_world: np.ndarray) -> None:
    """Add 'position_base_link_m' and 'pose_base_link' to each result in-place."""
    world_T_cam = np.linalg.inv(cam_T_world)
    for r in results:
        if "error" in r:
            continue
        if "position_m" in r:
            r["position_base_link_m"] = (world_T_cam @ np.append(r["position_m"], 1.0))[:3]
        if r.get("rotation_matrix") is not None and r.get("translation_m") is not None:
            T = np.eye(4)
            T[:3, :3] = r["rotation_matrix"]
            T[:3,  3] = r["translation_m"]
            r["pose_base_link"] = world_T_cam @ T


# ═══════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def run_pose_estimation(
    objects=None,
    camera_fx: float = CAMERA_FX,
    camera_fy: float = CAMERA_FY,
    use_multi_view: bool = False,
    poses_path: str = "poses.npy",
    seg_name_map=None,
    db_dir: str = DB_DIR,
    output_dir: str = OUTPUT_DIR,
    n_views: int = N_VIEWS,
    scene_idx: int = SCENE_IDX,
    force_rebuild: bool = False,
    save_figures: bool = True,
    images_dir: str = "data/captured_images",
    masks_dir: str = "data/segmented",
) -> list[dict]:
    """
    Run 6D pose estimation and return results.

    Parameters
    ----------
    objects       : list of ObjectConfig — defaults to the OBJECTS list above
    camera_fx/fy  : camera focal lengths in pixels
    use_multi_view: True → 9-view triangulated pipeline; False → single image
    poses_path    : path to poses.npy (only used when use_multi_view=True)
    seg_name_map  : {seg_label: config_name} mapping for mask filenames
    db_dir        : directory for cached reference databases
    output_dir    : directory for output figures
    n_views       : number of rendered reference views per object
    scene_idx     : which captured_image_N to use (only when use_multi_view=False)
    force_rebuild : rebuild reference databases even if cached files exist
    save_figures  : write scene_poses/comparisons/masks_debug PNGs to output_dir
    images_dir    : directory containing captured_image_N.jpg files
    masks_dir     : directory containing per-image segmentation subfolders

    Returns
    -------
    list of result dicts, one per detected object instance
    """
    if objects is None:
        objects = OBJECTS
    if seg_name_map is None:
        seg_name_map = SEG_NAME_MAP

    estimator = PoseEstimator(
        configs        = objects,
        db_dir         = db_dir,
        n_render_views = n_views,
        camera_fx      = camera_fx,
        camera_fy      = camera_fy,
    )
    estimator.build_databases(force_rebuild=force_rebuild)

    if use_multi_view:
        #   image 1        → vertical/home pose
        #   images 2–21    → high arc (20 shots);
        #   images 22–41   → low  arc (20 shots);
        home, high_arc, low_arc = _load_poses(poses_path)
        arc_idx = np.linspace(0, 19, NUM_POSES_PER_LAYER, dtype=int)
        scenes = (
            [(f"{images_dir}/captured_image_1.jpg",
              f"{masks_dir}/captured_image_1",
              home)]
            + [(f"{images_dir}/captured_image_{i + 2}.jpg",
                f"{masks_dir}/captured_image_{i + 2}",
                high_arc[i]) for i in arc_idx]
            + [(f"{images_dir}/captured_image_{i + 22}.jpg",
                f"{masks_dir}/captured_image_{i + 22}",
                low_arc[i]) for i in arc_idx]
        )

        raw_images, mask_dirs, cam_poses = [], [], []
        for img_path, mask_dir, ee_T_world in scenes:
            bgr = cv2.imread(img_path)
            if bgr is None:
                raise FileNotFoundError(f"Could not load: {img_path}")
            raw_images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            mask_dirs.append(mask_dir)
            cam_poses.append(cam_T_world_from_ee(ee_T_world))

        results = estimator.process_multi_view(
            list(zip(raw_images, cam_poses)),
            camera_fx    = camera_fx,
            camera_fy    = camera_fy,
            top_k        = 5,
            refine       = True,
            mask_dirs    = mask_dirs,
            seg_name_map = seg_name_map,
        )
        image = raw_images[0]
        _add_base_link_poses(results, cam_poses[0])

    else:
        scene_image = f"{images_dir}/captured_image_{scene_idx}.jpg"
        mask_dir    = f"{masks_dir}/captured_image_{scene_idx}"
        bgr = cv2.imread(scene_image)
        if bgr is None:
            raise FileNotFoundError(f"Could not load: {scene_image}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        results = estimator.process_scene(
            image,
            camera_fx    = camera_fx,
            camera_fy    = camera_fy,
            top_k        = 5,
            refine       = True,
            mask_dir     = mask_dir,
            seg_name_map = seg_name_map,
        )

        if os.path.isfile(poses_path):
            raw_poses = np.load(poses_path, allow_pickle=True)
            if scene_idx == 1:
                ee = _pose7_to_ee_T_world([float(raw_poses[k]) for k in range(7)])
            else:
                ee = _pose7_to_ee_T_world(raw_poses[scene_idx + 5])
            _add_base_link_poses(results, cam_T_world_from_ee(ee))

    print_results_table(results)

    if save_figures:
        save_all_figures(
            image      = image,
            results    = results,
            refiners   = estimator.refiners,
            output_dir = output_dir,
            fx         = camera_fx,
            fy         = camera_fy,
        )
        print(f"\nDone! Check {output_dir}/ for:")
        print("  scene_poses.png   — axes drawn at each estimated pose")
        print("  comparisons.png   — real crop vs render per object")
        print("  masks_debug.png   — auto-generated colour masks")

    os.makedirs(output_dir, exist_ok=True)
    poses_out = [
        {
            "object_name":         r["object_name"],
            "instance_idx":        r.get("instance_idx"),
            "position_base_link_m": r.get("position_base_link_m"),
            "pose_base_link":      r.get("pose_base_link"),
            "confidence":          r.get("confidence"),
        }
        for r in results if "error" not in r
    ]
    out_path = os.path.join(output_dir, "object_poses.npy")
    np.save(out_path, poses_out, allow_pickle=True)
    print(f"Object poses saved → {out_path}")

    return results


if __name__ == "__main__":
    run_pose_estimation()
