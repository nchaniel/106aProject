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

    Optional fields
    ───────────────
    diameter_m    Real physical diameter in metres (used to convert t to metres)
    is_symmetric  If True, only position is solved — rotation is skipped.
    """
    name: str
    stl_path: str
    color_rgb: tuple                        # (R, G, B)

    diameter_m: Optional[float] = None
    is_symmetric: Optional[bool] = None     # None = auto-detect from shape
