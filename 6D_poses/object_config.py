"""
object_config.py
────────────────
Defines the configuration for each object in the scene.
Edit OBJECTS in your run script — everything else is automatic.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class ObjectConfig:
    """
    Full configuration for one object.

    Required fields
    ───────────────
    name          Short unique identifier, e.g. "red_sphere"
    stl_path      Path to the .stl file
    color_rgb     Render colour as (R, G, B) ints 0-255
    hsv_low       HSV lower bound for colour segmentation (H 0-179, S 0-255, V 0-255)
    hsv_high      HSV upper bound

    Optional fields
    ───────────────
    diameter_m        Real physical diameter in metres (used to convert t to metres)
    is_symmetric      If True, only position is solved — rotation is skipped.
    hsv_low2/high2    Second HSV range (for colours that wrap around hue=0, e.g. red)
    """
    name: str
    stl_path: str
    color_rgb: tuple                        # (R, G, B)
    hsv_low: tuple                          # (H, S, V)
    hsv_high: tuple                         # (H, S, V)

    diameter_m: Optional[float] = None
    is_symmetric: Optional[bool] = None     # None = auto-detect from shape

    # Second HSV range for hue-wrapping colours (e.g. red spans 170-180 AND 0-10)
    hsv_low2: Optional[tuple] = None
    hsv_high2: Optional[tuple] = None


# ─── HSV RANGE QUICK REFERENCE ───────────────────────────────────────────────
# Colour        H low   H high   Notes
# ──────────────────────────────────────────────────────────────────────────────
# Red           0-10  + 165-179  Wraps around! Use hsv_low2/high2 for second range
# Orange        10      25
# Yellow        25      35
# Green         35      85
# Cyan          85      100
# Blue          100     130
# Purple/Violet 130     155
# Pink/Magenta  155     175
# Black         any     any      S<50, V<60
# White         any     any      S<30, V>200
# Grey          any     any      S<40, 60<V<200
# ─────────────────────────────────────────────────────────────────────────────
#
# TIP: Run `python hsv_picker.py scene.jpg` to click on your objects and
# get exact HSV ranges automatically.
