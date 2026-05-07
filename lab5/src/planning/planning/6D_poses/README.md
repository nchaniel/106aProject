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

Three figures saved to `./results/` after every run:

| File | Shows |
|---|---|
| `scene_poses.png` | Scene with XYZ axes at each estimated pose |
| `comparisons.png` | Real crop \| render at pose \| edge overlay per object |
| `masks_debug.png` | Auto-generated colour masks — use to verify segmentation |
