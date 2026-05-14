"""
constants.py
────────────
Central place for all project-wide constants.
Edit this file to change objects, camera intrinsics, paths, or robot geometry.
"""

import numpy as np
from object_config import ObjectConfig

# ═══════════════════════════════════════════════════════════════════════════
# OBJECTS
# ═══════════════════════════════════════════════════════════════════════════
#
# For each object you need:
#   name       short identifier (no spaces)
#   stl_path   path to the .stl file
#   color_rgb  the colour the object is printed in, as (R, G, B)
#
# Masks are provided by YOLO — no HSV segmentation is used.
# ═══════════════════════════════════════════════════════════════════════════

OBJECTS = [

    ObjectConfig(
        name        = "apple",
        stl_path    = "models/Apple_STL.stl",
        color_rgb   = (200, 40, 40),
        diameter_m  = 0.045,
    ),

    ObjectConfig(
        name        = "blueberry",
        stl_path    = "models/Blueberry_STL.stl",
        color_rgb   = (60, 40, 100),
        diameter_m  = 0.015,
    ),

    ObjectConfig(
        name        = "cherry",
        stl_path    = "models/Cherry_STL.stl",
        color_rgb   = (180, 30, 30),
        diameter_m  = 0.02,
    ),

    ObjectConfig(
        name        = "cake",
        stl_path    = "models/Lava_Cake_STL.stl",
        color_rgb   = (80, 40, 15),
        diameter_m  = 0.075,
        is_symmetric = False,
    ),

    ObjectConfig(
        name        = "grape",
        stl_path    = "models/Grape_STL.stl",
        color_rgb   = (100, 50, 120),
        diameter_m  = 0.0236,
    ),

    ObjectConfig(
        name        = "half_grape",
        stl_path    = "models/Half_Grape_STL.stl",
        color_rgb   = (100, 50, 120),
        diameter_m  = 0.0236,
        is_symmetric = False,
    ),

    ObjectConfig(
        name        = "halved_strawberry",
        stl_path    = "models/Halved_Strawberry_STL.stl",
        color_rgb   = (210, 50, 60),
        diameter_m  = 0.0301,
        is_symmetric = False,
    ),

    ObjectConfig(
        name        = "mango_piece",
        stl_path    = "models/Mango_Piece_STL.stl",
        color_rgb   = (220, 150, 50),
        diameter_m  = 0.04,
        is_symmetric = False,
    ),

    ObjectConfig(
        name        = "small_tomato",
        stl_path    = "models/Small_Tomato_STL.stl",
        color_rgb   = (210, 50, 40),
        diameter_m  = 0.03,
    ),

    ObjectConfig(
        name        = "strawberry",
        stl_path    = "models/Strawberry_STL.stl",
        color_rgb   = (210, 50, 60),
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

SCENE_IDX  = 2               # which captured image to process (single-view)
OUTPUT_DIR = "./results"     # figures saved here
DB_DIR     = "./pose_db"     # reference databases cached here
N_VIEWS    = 500             # render views per object (increase for accuracy)

# Maps the label names used in mask filenames to ObjectConfig names above.
# Keys are whatever the segmentation model wrote into the filename;
# values must match the `name` field in OBJECTS. Omit labels you want skipped.
SEG_NAME_MAP = {
    "blueberry":  "blueberry",
    "cake":       "cake",
    "grape":      "grape",
    "strawberry": "strawberry",
    "tomato": "small_tomato",
    "apple": "apple",
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

NUM_POSES_PER_LAYER = 2