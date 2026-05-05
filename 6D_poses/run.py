"""
run.py
──────
Configure objects and camera parameters below, then either:

    python run.py                          # standalone
    from run import run_pose_estimation    # as a library

Returns a list of result dicts — one per detected object instance.
"""

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from pose_estimator import PoseEstimator
from object_config import ObjectConfig
from visualisation import save_all_figures, print_results_table


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
# CONFIGURE YOUR OBJECTS
# ═══════════════════════════════════════════════════════════════════════════
#
# For each object you need:
#   name         short identifier (no spaces)
#   stl_path     path to the .stl file
#   color_rgb    the colour the object is printed in, as (R, G, B)
#   hsv_low/high HSV thresholds for auto-masking
#
# TIP: Run `python hsv_picker.py scene.jpg` to get HSV values by clicking
#      on your objects — it prints ready-to-paste config lines.
#
# Red is special: it wraps around hue=0 in HSV, so it needs two ranges.
# Use hsv_low2/hsv_high2 for the second range (see red objects below).
# ═══════════════════════════════════════════════════════════════════════════

OBJECTS = [

    ObjectConfig(
        name        = "apple",
        stl_path    = "models/Apple_STL.stl",
        color_rgb   = (200, 40, 40),
        hsv_low     = (0,   186, 105),
        hsv_high    = (10,  255, 198),
        hsv_low2    = (165, 186, 105),
        hsv_high2   = (179, 255, 198),
        diameter_m  = 0.045,
    ),

    ObjectConfig(
        name        = "blueberry",
        stl_path    = "models/Blueberry_STL.stl",
        color_rgb   = (60, 40, 100),
        hsv_low     = (120, 50, 20),
        hsv_high    = (150, 200, 120),
        diameter_m  = 0.015,
    ),

    ObjectConfig(
        name        = "cherry",
        stl_path    = "models/Cherry_STL.stl",
        color_rgb   = (180, 30, 30),
        hsv_low     = (0,   120, 60),
        hsv_high    = (10,  255, 200),
        hsv_low2    = (165, 120, 60),
        hsv_high2   = (179, 255, 200),
        diameter_m  = 0.02,
    ),

    ObjectConfig(
        name        = "cake",
        stl_path    = "models/Lava_Cake_STL.stl",
        color_rgb   = (80, 40, 15),
        hsv_low     = (82, 10, 0),
        hsv_high    = (118, 137, 62),
        diameter_m  = 0.075,
        is_symmetric = False,
    ),

    ObjectConfig(
        name        = "grape",
        stl_path    = "models/Grape_STL.stl",
        color_rgb   = (100, 50, 120),
        hsv_low     = (114, 45, 1),
        hsv_high    = (146, 207, 158),
        diameter_m  = 0.0236,
    ),

    ObjectConfig(
        name        = "half_grape",
        stl_path    = "models/Half_Grape_STL.stl",
        color_rgb   = (100, 50, 120),
        hsv_low     = (114, 45, 1),
        hsv_high    = (146, 207, 158),
        diameter_m  = 0.0236,
        is_symmetric = False,
    ),

    ObjectConfig(
        name        = "halved_strawberry",
        stl_path    = "models/Halved_Strawberry_STL.stl",
        color_rgb   = (210, 50, 60),
        hsv_low     = (0,   186, 105),
        hsv_high    = (10,  255, 198),
        hsv_low2    = (165, 186, 105),
        hsv_high2   = (179, 255, 198),
        diameter_m  = 0.0301,
        is_symmetric = False,
    ),

    ObjectConfig(
        name        = "mango_piece",
        stl_path    = "models/Mango_Piece_STL.stl",
        color_rgb   = (220, 150, 50),
        hsv_low     = (15,  100, 100),
        hsv_high    = (35,  255, 255),
        diameter_m  = 0.04,
        is_symmetric = False,
    ),

    ObjectConfig(
        name        = "small_tomato",
        stl_path    = "models/Small_Tomato_STL.stl",
        color_rgb   = (210, 50, 40),
        hsv_low     = (0,   120, 80),
        hsv_high    = (10,  255, 255),
        hsv_low2    = (165, 120, 80),
        hsv_high2   = (179, 255, 255),
        diameter_m  = 0.03,
    ),

    ObjectConfig(
        name        = "strawberry",
        stl_path    = "models/Strawberry_STL.stl",
        color_rgb   = (210, 50, 60),
        hsv_low     = (0,   120, 80),
        hsv_high    = (10,  255, 255),
        hsv_low2    = (165, 120, 80),
        hsv_high2   = (179, 255, 255),
        diameter_m  = 0.0301,
        is_symmetric = False,
    ),

]

# ═══════════════════════════════════════════════════════════════════════════
# CAMERA INTRINSICS
# ═══════════════════════════════════════════════════════════════════════════
# Get these from cv2.calibrateCamera(), or use this rough estimate:
#   fx = fy ≈ image_width * 1.2   (for a typical phone/webcam)

CAMERA_FX = 700.0
CAMERA_FY = 700.0

# ═══════════════════════════════════════════════════════════════════════════
# PATHS / SETTINGS
# ═══════════════════════════════════════════════════════════════════════════

SCENE_IDX    = 2                # which captured image to process (single-view)
OUTPUT_DIR   = "./results"       # figures saved here
DB_DIR       = "./pose_db"       # reference databases cached here
N_VIEWS      = 500               # render views per object (increase for accuracy)

# Maps the label names used in mask filenames to ObjectConfig names above.
# Keys are whatever the segmentation model wrote into the filename;
# values must match the `name` field in OBJECTS. Omit labels you want skipped.
SEG_NAME_MAP = {
    "blueberry":  "blueberry",
    "cake":       "cake",
    "grape":      "grape",
    "strawberry": "strawberry",
}

# ═══════════════════════════════════════════════════════════════════════════
# CAMERA MOUNT GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════
# Fixed transform from the end-effector frame to the camera frame.
# Camera is mounted 3 cm in front (+x) and 13.5 cm above (-z) the
# end-effector, with the same orientation.  Adjust axis signs to match
# your robot's convention, and add a rotation block if the camera is
# tilted relative to the end-effector.

T_CAM_FROM_EE = np.array([
    [1, 0, 0,  0.030],   #  3.0 cm forward
    [0, 1, 0,  0.000],
    [0, 0, 1, -0.135],   # 13.5 cm above
    [0, 0, 0,  1.000],
])


def cam_T_world_from_ee(ee_T_world: np.ndarray) -> np.ndarray:
    """Convert a world-to-end-effector pose to a world-to-camera pose."""
    return np.linalg.inv(T_CAM_FROM_EE) @ ee_T_world


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
        # 9-view subset:
        #   image 1        → vertical/home pose
        #   images 2–21    → high arc (20 shots); pick 4 evenly spaced
        #   images 22–41   → low  arc (20 shots); pick 4 evenly spaced
        home, high_arc, low_arc = _load_poses(poses_path)
        arc_idx = np.linspace(0, 19, 4, dtype=int)   # [0, 6, 13, 19]
        scenes = (
            [("data/captured_images/captured_image_1.jpg",
              "data/segmented/captured_image_1",
              home)]
            + [(f"data/captured_images/captured_image_{i + 2}.jpg",
                f"data/segmented/captured_image_{i + 2}",
                high_arc[i]) for i in arc_idx]
            + [(f"data/captured_images/captured_image_{i + 22}.jpg",
                f"data/segmented/captured_image_{i + 22}",
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

    else:
        scene_image = f"data/captured_images/captured_image_{scene_idx}.jpg"
        mask_dir    = f"data/segmented/captured_image_{scene_idx}"
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

    return results


if __name__ == "__main__":
    run_pose_estimation()
