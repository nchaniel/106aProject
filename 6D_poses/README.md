# 6DoF Pose Estimator v2

RGB-only pose estimation for coloured 3D-printed objects.
No depth sensor. No manual masks. Runs on Mac (CPU / Apple Silicon MPS).

---

## Files

| File | Purpose |
|---|---|
| `run.py` | Configure objects + run. |
| `object_config.py` | `ObjectConfig` dataclass — defines each object |
| `pose_estimator.py` | Full pipeline (renderer, segmenter, DINOv2, refiner) |
| `visualisation.py` | Three output figures |
| `hsv_picker.py` | Interactive helper to get HSV ranges by clicking |
| `requirements.txt` | Dependencies |

---

## Quickstart

### 1. Install
```bash
pip install -r requirements.txt
```

macOS display fix (if you get EGL errors):
```bash
brew install mesa
export PYOPENGL_PLATFORM=osmesa
```

### 2. Get your HSV ranges
```bash
python hsv_picker.py scene.jpg
# Click on each object → prints copy-paste config lines
```

### 3. Configure objects in `run.py`
```python
OBJECTS = [
    ObjectConfig(
        name       = "red_sphere",
        stl_path   = "models/red_sphere.stl",
        color_rgb  = (200, 40, 40),
        hsv_low    = (0,   120, 80),
        hsv_high   = (10,  255, 255),
        hsv_low2   = (165, 120, 80),   # red wraps around hue=0
        hsv_high2  = (179, 255, 255),
        diameter_m = 0.04,             # real diameter in metres
    ),
    ...
]
```

### 4. Run
```bash
python run.py
```

**First run**: builds reference databases (~5–7 min per object).  
**After that**: loads from cache instantly.

---

## Outputs

### Saved figures

Three figures saved to `./results/` after every run:

| File | Shows |
|---|---|
| `scene_poses.png` | Scene with XYZ axes at each estimated pose |
| `comparisons.png` | Real crop \| render at pose \| edge overlay per object |
| `masks_debug.png` | Auto-generated colour masks — use to verify segmentation |

---

### `process_scene` return value

Returns `list[dict]` — one dict per detected object instance.

**Always present:**

| Key | Type | Description |
|---|---|---|
| `object_name` | `str` | Name of the detected object (matches `ObjectConfig.name`) |
| `instance_idx` | `int` | 0-based index when multiple instances of the same object are found |
| `mask` | `np.ndarray (H, W)` | Binary mask; 255 = object pixels, 0 = background |
| `inference_time_s` | `float` | Wall-clock seconds taken for this instance |
| `pose_w2c` | `np.ndarray (4, 4)` | Refined world-to-camera homogeneous transform |
| `coarse_pose_w2c` | `np.ndarray (4, 4)` | Initial DINOv2 retrieval pose before edge refinement |
| `rotation_matrix` | `np.ndarray (3, 3)` or `None` | Rotation block of `pose_w2c`; `None` for symmetric objects |
| `translation` | `np.ndarray (3,)` | Translation vector in normalised (unit-sphere) coordinates |
| `euler_angles_zyx_deg` | `np.ndarray (3,)` or `None` | ZYX Euler angles in degrees; `None` for symmetric objects |
| `confidence` | `float [0, 1]` | Mapped from edge score — higher is better |
| `score` | `float` | Raw Chamfer edge distance — lower is better |
| `is_symmetric` | `bool` | `True` if rotation estimation was skipped (sphere-like object) |
| `diameter_m` | `float` or `None` | Real-world object diameter from `ObjectConfig` |

**Present when `ObjectConfig.diameter_m` is set:**

| Key | Type | Description |
|---|---|---|
| `translation_m` | `np.ndarray (3,)` | `translation` scaled to metres using `diameter_m` |
| `position_m` | `np.ndarray (3,)` | Metric `[X, Y, Z]` in the camera frame estimated from apparent object size and focal length |

**On detection failure:**

| Key | Type | Description |
|---|---|---|
| `object_name` | `str` | Name of the object that failed |
| `instance_idx` | `int` | Instance index |
| `error` | `str` | Error message; all other keys are absent |

---

### `process_multi_view` return value

Returns `list[dict]` — same fields as `process_scene` (taken from the highest-confidence view, or view 0 when available), plus:

| Key | Type | Description |
|---|---|---|
| `triangulated_position_m` | `np.ndarray (3,)` | DLT-triangulated 3D world-frame position in metres; only present when the object was seen in ≥ 2 views |
| `n_views_triangulated` | `int` | Number of views used for triangulation; only present alongside `triangulated_position_m` |

The result list is ordered to match view 0's detections so that masks align when visualising against the first image.
