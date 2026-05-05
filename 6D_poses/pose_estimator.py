"""
pose_estimator.py
─────────────────
Full 6DoF pose estimation pipeline.
  - Colour-matched STL rendering
  - Automatic HSV colour segmentation (no manual masks needed)
  - Shape disambiguation for same-colour objects
  - Size-based disambiguation for same-colour same-shape objects
  - Rotation skipped for rotationally symmetric objects
  - DINOv2 ViT-S/14 for coarse retrieval
  - Iterative edge-based refinement
"""

import re
import time
import numpy as np
import cv2
import torch
import trimesh
import pyrender
from pathlib import Path
from typing import Optional
from scipy.spatial.transform import Rotation
from sklearn.neighbors import NearestNeighbors

from object_config import ObjectConfig


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("✓ Using Apple Metal (MPS) for DINOv2")
        return torch.device("mps")
    print("⚠  MPS not available — using CPU")
    return torch.device("cpu")


def _look_at(eye: np.ndarray,
             center: np.ndarray = None,
             up: np.ndarray = None) -> np.ndarray:
    """4×4 camera-to-world matrix (OpenGL convention)."""
    if center is None: center = np.zeros(3)
    if up is None:     up = np.array([0.0, 1.0, 0.0])
    f = center - eye
    norm = np.linalg.norm(f)
    if norm < 1e-8:
        return np.eye(4)
    f /= norm
    if abs(np.dot(f, up)) > 0.99:
        up = np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up);  r /= np.linalg.norm(r)
    u = np.cross(r, f)
    mat = np.eye(4)
    mat[:3, 0] = r;  mat[:3, 1] = u;  mat[:3, 2] = -f;  mat[:3, 3] = eye
    return mat


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Returns (n, 3) unit vectors evenly spread on a sphere."""
    golden = (1 + np.sqrt(5)) / 2
    i = np.arange(n)
    theta = 2 * np.pi * i / golden
    phi   = np.arccos(1 - 2 * (i + 0.5) / n)
    return np.stack([np.sin(phi)*np.cos(theta),
                     np.sin(phi)*np.sin(theta),
                     np.cos(phi)], axis=1)


def _detect_sphere(stl_path: str, threshold: float = 0.92) -> bool:
    """
    Heuristic: a mesh is sphere-like if its vertices lie close to a sphere.
    Returns True if the mesh is rotationally symmetric enough to skip rotation.
    """
    mesh = trimesh.load(stl_path)
    verts = np.array(mesh.vertices)
    verts -= verts.mean(axis=0)
    radii = np.linalg.norm(verts, axis=1)
    if radii.max() < 1e-8:
        return False
    radii_norm = radii / radii.max()
    # Coefficient of variation: low = very sphere-like
    cv = radii_norm.std() / radii_norm.mean()
    return cv < (1.0 - threshold)


def triangulate_dlt(
    centroids_uv: list[tuple[float, float]],
    poses_w2c: list[np.ndarray],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    DLT triangulation from ≥2 calibrated views.

    centroids_uv: [(u, v), ...] pixel centroid of the object per view
    poses_w2c:    [4×4, ...] world-to-camera transforms (same world frame)
    Returns:      (3,) point in that world frame, in the same units as the
                  translations in poses_w2c
    """
    A = []
    for (u, v), T in zip(centroids_uv, poses_w2c):
        P = np.array([[fx, 0, cx, 0],
                      [0, fy, cy, 0],
                      [0,  0,  1, 0]], dtype=np.float64) @ T.astype(np.float64)
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A, dtype=np.float64))
    X = Vt[-1]
    return (X[:3] / X[3]).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-computed mask loader
# ─────────────────────────────────────────────────────────────────────────────

def load_masks_from_dir(
    mask_dir: str | Path,
    name_map: dict[str, str],
) -> dict[str, list[np.ndarray]]:
    """
    Load pre-computed binary masks from a directory.

    Expects files named:  <anything>_<seg_label>_<idx>_mask.png
    name_map maps seg_label → ObjectConfig name (unknown labels are skipped).
    Returns {config_name: [mask0, mask1, ...]} sorted by instance index.
    """
    mask_dir = Path(mask_dir)
    labels   = sorted(name_map.keys(), key=len, reverse=True)  # longest first
    pattern  = re.compile(
        r'_(' + '|'.join(re.escape(l) for l in labels) + r')_(\d+)_mask$'
    )
    buckets: dict[str, list[tuple[int, np.ndarray]]] = {}
    for f in sorted(mask_dir.glob("*_mask.png")):
        m = pattern.search(f.stem)
        if not m:
            continue
        config_name = name_map[m.group(1)]
        inst_idx    = int(m.group(2))
        mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        buckets.setdefault(config_name, []).append((inst_idx, binary))
    return {name: [mask for _, mask in sorted(insts)]
            for name, insts in buckets.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 1. COLOUR SEGMENTER
# ─────────────────────────────────────────────────────────────────────────────

class ColourSegmenter:
    """
    Generates per-object binary masks from HSV colour thresholding.
    Handles same-colour objects by shape/size disambiguation.
    """

    def segment_scene(
        self,
        image: np.ndarray,          # (H, W, 3) RGB
        configs: list[ObjectConfig],
    ) -> dict[str, list[np.ndarray]]:
        """
        Returns {object_name: [mask, mask, ...]} — one mask per detected instance.
        Multiple instances of the same object are ordered largest→smallest.
        """
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Step 1: get raw colour masks per config
        colour_masks: dict[str, np.ndarray] = {}
        for cfg in configs:
            mask = self._colour_mask(hsv, cfg)
            colour_masks[cfg.name] = mask

        # Step 2: for same-colour configs, split shared blobs by shape/size
        results: dict[str, list[np.ndarray]] = {}
        grouped = self._group_by_colour(configs)

        for colour_key, group in grouped.items():
            if len(group) == 1:
                cfg = group[0]
                blobs = self._extract_blobs(colour_masks[cfg.name])
                results[cfg.name] = blobs if blobs else []
            else:
                # Multiple objects share a colour — disambiguate by shape/size
                shared_mask = colour_masks[group[0].name]
                blobs = self._extract_blobs(shared_mask)
                self._disambiguate(blobs, group, results)

        return results

    def _colour_mask(self, hsv: np.ndarray, cfg: ObjectConfig) -> np.ndarray:
        lo1 = np.array(cfg.hsv_low,  dtype=np.uint8)
        hi1 = np.array(cfg.hsv_high, dtype=np.uint8)
        mask = cv2.inRange(hsv, lo1, hi1)

        if cfg.hsv_low2 is not None and cfg.hsv_high2 is not None:
            lo2 = np.array(cfg.hsv_low2,  dtype=np.uint8)
            hi2 = np.array(cfg.hsv_high2, dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        return mask

    def _extract_blobs(self, mask: np.ndarray,
                       min_area: int = 200) -> list[np.ndarray]:
        """Returns list of single-blob masks, sorted largest→smallest."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        blobs = []
        for i in range(1, n):  # skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            blob_mask = (labels == i).astype(np.uint8) * 255
            blobs.append((area, blob_mask))
        blobs.sort(key=lambda x: x[0], reverse=True)
        return [b for _, b in blobs]

    def _group_by_colour(
        self, configs: list[ObjectConfig]
    ) -> dict[str, list[ObjectConfig]]:
        """Groups configs that have identical HSV ranges."""
        groups: dict[str, list[ObjectConfig]] = {}
        for cfg in configs:
            key = str(cfg.hsv_low) + str(cfg.hsv_high)
            groups.setdefault(key, []).append(cfg)
        return groups

    def _disambiguate(
        self,
        blobs: list[np.ndarray],
        group: list[ObjectConfig],
        results: dict[str, list[np.ndarray]],
    ):
        """
        Assigns blobs to same-colour objects using:
        1. Size — sort both blobs and configs by expected area
        2. Shape circularity — sphere vs non-sphere
        """
        if not blobs:
            for cfg in group:
                results[cfg.name] = []
            return

        # Score blobs: circularity = 4π·area / perimeter²  (1.0 = perfect circle)
        def circularity(mask):
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return 0.0
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perim = cv2.arcLength(c, True)
            if perim < 1e-3:
                return 0.0
            return 4 * np.pi * area / (perim ** 2)

        blob_info = [(cv2.countNonZero(b), circularity(b), b) for b in blobs]
        blob_info.sort(key=lambda x: x[0], reverse=True)  # largest first

        # Classify configs: sphere-like vs non-sphere by STL geometry
        def is_sphere_cfg(cfg):
            try:
                return _detect_sphere(cfg.stl_path)
            except Exception:
                return False

        # Sort configs: non-spheres first (they're typically bigger/flatter blobs),
        # spheres last (rounder, sometimes smaller)
        sphere_cfgs  = [c for c in group if is_sphere_cfg(c)]
        other_cfgs   = [c for c in group if not is_sphere_cfg(c)]

        ordered_cfgs = other_cfgs + sphere_cfgs

        for i, cfg in enumerate(ordered_cfgs):
            if i < len(blob_info):
                results.setdefault(cfg.name, []).append(blob_info[i][2])
            else:
                results.setdefault(cfg.name, [])


# ─────────────────────────────────────────────────────────────────────────────
# 2. REFERENCE RENDERER
# ─────────────────────────────────────────────────────────────────────────────

class ReferenceRenderer:
    """
    Renders an STL from many viewpoints with:
      - Object colour from ObjectConfig.color_rgb
    """

    def __init__(self, img_size: int = 224):
        self.img_size = img_size

    def load_mesh(self, cfg: ObjectConfig) -> tuple[trimesh.Trimesh, pyrender.Mesh]:
        """Loads, normalises, and colours the mesh."""
        mesh = trimesh.load(cfg.stl_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(
                [g for g in mesh.geometry.values()])

        # Normalise: centre + unit sphere
        mesh.vertices -= mesh.bounding_box.centroid
        scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
        if scale > 0:
            mesh.vertices /= scale

        r, g, b = [c / 255.0 for c in cfg.color_rgb]
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[r, g, b, 1.0],
            metallicFactor=0.05,
            roughnessFactor=0.85,
        )
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        return mesh, pr_mesh

    def render_views(
        self,
        cfg: ObjectConfig,
        n_views: int = 500,
        camera_distance: float = 2.5,
        fx: float = 525.0,
        fy: float = 525.0,
    ) -> list[dict]:
        W = H = self.img_size
        tm_mesh, pr_mesh = self.load_mesh(cfg)

        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
        scene.add(pr_mesh)

        camera = pyrender.IntrinsicsCamera(
            fx=fx, fy=fy, cx=W/2, cy=H/2, znear=0.1, zfar=10.0)
        light_key  = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        light_fill = pyrender.DirectionalLight(color=[0.7, 0.85, 1.0], intensity=1.2)

        renderer = pyrender.OffscreenRenderer(W, H)
        viewpoints = _fibonacci_sphere(n_views)
        results = []

        for i, vp in enumerate(viewpoints):
            eye = vp * camera_distance
            c2w = _look_at(eye)
            cam_pose = c2w @ np.diag([1, -1, -1, 1])

            cn = scene.add(camera, pose=cam_pose)
            ln1 = scene.add(light_key,  pose=cam_pose)
            ln2 = scene.add(light_fill, pose=_look_at(-vp * camera_distance))

            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color[:, :, :3]

            scene.remove_node(cn)
            scene.remove_node(ln1)
            scene.remove_node(ln2)

            w2c = np.linalg.inv(cam_pose)
            results.append({
                "image": color,
                "depth": depth,
                "pose_w2c": w2c,
                "pose_c2w": cam_pose,
                "view_idx": i,
            })

        renderer.delete()
        return results


# ─────────────────────────────────────────────────────────────────────────────
# 3. DINO FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class DinoFeatureExtractor:
    def __init__(self, device: torch.device):
        self.device = device
        print("Loading DINOv2 ViT-S/14...")
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14",
            pretrained=True, verbose=False,
        )
        self.model.eval().to(device)
        print("✓ DINOv2 loaded")

    @torch.no_grad()
    def extract(self, image: np.ndarray,
                mask: Optional[np.ndarray] = None) -> dict:
        """image: (H,W,3) uint8 RGB.  Returns {cls, patches}."""
        t = self._preprocess(image, mask).to(self.device)
        out = self.model.forward_features(t)
        return {
            "cls":     out["x_norm_clstoken"][0].cpu(),
            "patches": out["x_norm_patchtokens"][0].cpu(),
        }

    def _preprocess(self, image: np.ndarray,
                    mask: Optional[np.ndarray]) -> torch.Tensor:
        img = cv2.resize(image, (224, 224)).astype(np.float32) / 255.0
        if mask is not None:
            m = cv2.resize(mask, (224, 224),
                           interpolation=cv2.INTER_NEAREST)
            img *= (m > 0).astype(np.float32)[..., None]
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = (img - mean) / std
        return torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float()


# ─────────────────────────────────────────────────────────────────────────────
# 4. REFERENCE DATABASE
# ─────────────────────────────────────────────────────────────────────────────

class ReferenceDatabase:
    def __init__(self):
        self.cls_features: Optional[np.ndarray] = None
        self.poses_w2c: list[np.ndarray] = []
        self.images:    list[np.ndarray] = []

    def build(self, views: list[dict],
              extractor: DinoFeatureExtractor,
              object_name: str = ""):
        print(f"  Extracting features ({len(views)} views)...")
        cls_list = []
        t0 = time.time()
        for i, v in enumerate(views):
            feats = extractor.extract(v["image"])
            cls_list.append(feats["cls"].numpy())
            self.poses_w2c.append(v["pose_w2c"])
            self.images.append(v["image"])
            if (i+1) % 100 == 0:
                print(f"    {i+1}/{len(views)}  ({time.time()-t0:.0f}s)")

        self.cls_features = np.stack(cls_list)
        self._build_index()
        print(f"  ✓ {len(views)} views, {self.cls_features.shape[1]}D features")

    def _build_index(self):
        norms = np.linalg.norm(self.cls_features, axis=1, keepdims=True)
        self.cls_norm = self.cls_features / (norms + 1e-8)
        self.nn = NearestNeighbors(n_neighbors=5, metric="cosine",
                                   algorithm="brute")
        self.nn.fit(self.cls_norm)

    def query(self, cls_vec: np.ndarray, k: int = 5) -> list[dict]:
        q = cls_vec / (np.linalg.norm(cls_vec) + 1e-8)
        dists, idxs = self.nn.kneighbors([q], n_neighbors=k)
        return [{"view_idx": idx,
                 "pose_w2c": self.poses_w2c[idx],
                 "image":    self.images[idx],
                 "distance": float(dist)}
                for dist, idx in zip(dists[0], idxs[0])]

    def save(self, path: str):
        np.savez_compressed(path,
            cls_features=self.cls_features,
            poses_w2c=np.stack(self.poses_w2c),
            images=np.stack(self.images))
        print(f"  ✓ Saved → {path}.npz")

    def load(self, path: str):
        data = np.load(path + ".npz")
        self.cls_features = data["cls_features"]
        self.poses_w2c = list(data["poses_w2c"])
        self.images    = list(data["images"])
        self._build_index()
        print(f"  ✓ Loaded {len(self.poses_w2c)} views from {path}.npz")


# ─────────────────────────────────────────────────────────────────────────────
# 5. POSE REFINER
# ─────────────────────────────────────────────────────────────────────────────

class PoseRefiner:
    """
    Iterative render-and-compare refinement using Chamfer edge distance.
    Skips rotation perturbation for symmetric objects.
    """

    def __init__(self, cfg: ObjectConfig, img_size: int = 224):
        self.cfg = cfg
        self.img_size = img_size
        self.is_symmetric = self._resolve_symmetry(cfg)

        rr = ReferenceRenderer(img_size=img_size)
        _, self.pr_mesh = rr.load_mesh(cfg)
        self.tm_mesh = trimesh.load(cfg.stl_path)

        self.scene = pyrender.Scene(ambient_light=[0.4, 0.4, 0.4])
        self.scene.add(self.pr_mesh)
        self._renderer = pyrender.OffscreenRenderer(img_size, img_size)

    def _resolve_symmetry(self, cfg: ObjectConfig) -> bool:
        """Auto-detect sphere if is_symmetric is not explicitly set."""
        if cfg.is_symmetric is not None:
            return cfg.is_symmetric
        try:
            return _detect_sphere(cfg.stl_path)
        except Exception:
            return False

    def render_at_pose(self, pose_w2c: np.ndarray,
                       fx: float = 525.0, fy: float = 525.0) -> np.ndarray:
        W = H = self.img_size
        cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy,
                                        cx=W/2, cy=H/2,
                                        znear=0.1, zfar=10.0)
        cam_pose = np.linalg.inv(pose_w2c) @ np.diag([1, -1, -1, 1])
        light = pyrender.DirectionalLight(color=[1,1,1], intensity=3.0)
        cn = self.scene.add(cam,   pose=cam_pose)
        ln = self.scene.add(light, pose=cam_pose)
        color, _ = self._renderer.render(self.scene)
        self.scene.remove_node(cn)
        self.scene.remove_node(ln)
        return color[:, :, :3]

    def edge_score(self, render: np.ndarray,
                   real_crop: np.ndarray) -> float:
        """Chamfer-like distance on Canny edges. Lower = better."""
        def edges(img):
            g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return cv2.Canny(g, 50, 150).astype(np.float32) / 255.0

        re = edges(render);  qe = edges(real_crop)
        rd = cv2.distanceTransform((1-re).astype(np.uint8)*255, cv2.DIST_L2, 5)
        qd = cv2.distanceTransform((1-qe).astype(np.uint8)*255, cv2.DIST_L2, 5)
        score = (re * qd).sum() + (qe * rd).sum()
        return float(score / (re.sum() + qe.sum() + 1e-6))

    def refine(self, coarse_pose_w2c: np.ndarray,
               real_crop: np.ndarray,
               n_iterations: int = 6,
               fx: float = 525.0,
               fy: float = 525.0) -> tuple[np.ndarray, float]:

        best_pose  = coarse_pose_w2c.copy()
        best_score = float("inf")
        real = cv2.resize(real_crop, (self.img_size, self.img_size))

        angle_steps = np.linspace(np.radians(15), np.radians(2), n_iterations)
        trans_steps = np.linspace(0.15, 0.02, n_iterations)

        for it in range(n_iterations):
            render = self.render_at_pose(best_pose, fx=fx, fy=fy)
            score  = self.edge_score(render, real)
            if score < best_score:
                best_score = score

            candidates = self._perturb(best_pose, n=12,
                                       angle_std=angle_steps[it],
                                       trans_std=trans_steps[it])
            for p in candidates:
                r = self.render_at_pose(p, fx=fx, fy=fy)
                s = self.edge_score(r, real)
                if s < best_score:
                    best_score = s;  best_pose = p

        return best_pose, best_score

    def _perturb(self, pose, n, angle_std, trans_std):
        out = []
        for _ in range(n):
            p = pose.copy()
            if not self.is_symmetric:
                rvec = np.random.randn(3) * angle_std
                p[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(rvec).as_matrix()
            p[:3, 3] = pose[:3, 3] + np.random.randn(3) * trans_std
            out.append(p)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN ESTIMATOR
# ─────────────────────────────────────────────────────────────────────────────

class PoseEstimator:
    """
    Top-level API. Build once, run on many scenes.

    Typical usage
    ─────────────
        estimator = PoseEstimator(configs, db_dir="./pose_db")
        estimator.build_databases()                    # once
        results = estimator.process_scene(image)       # per scene
    """

    def __init__(
        self,
        configs: list[ObjectConfig],
        db_dir: str = "./pose_db",
        n_render_views: int = 500,
        img_size: int = 224,
        camera_fx: float = 525.0,
        camera_fy: float = 525.0,
    ):
        self.configs     = {c.name: c for c in configs}
        self.db_dir      = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.n_views     = n_render_views
        self.img_size    = img_size
        self.fx          = camera_fx
        self.fy          = camera_fy

        self.device      = get_device()
        self.extractor   = DinoFeatureExtractor(self.device)
        self.renderer    = ReferenceRenderer(img_size=img_size)
        self.segmenter   = ColourSegmenter()

        self.databases:  dict[str, ReferenceDatabase] = {}
        self.refiners:   dict[str, PoseRefiner]       = {}

    # ── Offline ───────────────────────────────────────────────────────────────

    def build_databases(self, force_rebuild: bool = False):
        print("\n══════════════════════════════════════════")
        print("  BUILDING REFERENCE DATABASES (offline)")
        print("══════════════════════════════════════════")
        t0 = time.time()

        for name, cfg in self.configs.items():
            db_path = str(self.db_dir / f"{name}_db")
            print(f"\n[{name}]")

            if not force_rebuild and Path(db_path + ".npz").exists():
                db = ReferenceDatabase()
                db.load(db_path)
            else:
                print(f"  Rendering {self.n_views} views...")
                views = self.renderer.render_views(
                    cfg, n_views=self.n_views,
                    fx=self.fx, fy=self.fy)
                db = ReferenceDatabase()
                db.build(views, self.extractor, object_name=name)
                db.save(db_path)

            self.databases[name] = db
            self.refiners[name]  = PoseRefiner(cfg, img_size=self.img_size)

        print(f"\n✓ All databases ready  ({(time.time()-t0)/60:.1f} min total)")

    # ── Online ────────────────────────────────────────────────────────────────

    def process_scene(
        self,
        image: np.ndarray,                  # (H, W, 3) RGB
        camera_fx: Optional[float] = None,
        camera_fy: Optional[float] = None,
        top_k: int = 5,
        refine: bool = True,
        mask_dir: Optional[str] = None,
        seg_name_map: Optional[dict[str, str]] = None,
    ) -> list[dict]:
        """
        Full pipeline: segment → estimate pose for every detected object.

        mask_dir:     path to a folder of pre-computed masks (skips HSV segmentation).
        seg_name_map: {seg_label: config_name} mapping for mask filenames.
                      Required when mask_dir is set.
        Returns list of result dicts — one per detected instance.
        """
        fx = camera_fx or self.fx
        fy = camera_fy or self.fy

        print("\n══════════════════════════════════════════")
        print("  ESTIMATING POSES")
        print("══════════════════════════════════════════")

        # 1. Get masks — from disk or HSV colour segmentation
        if mask_dir is not None:
            if not seg_name_map:
                raise ValueError("seg_name_map is required when mask_dir is set")
            print(f"  Using pre-computed masks from: {mask_dir}")
            all_masks = load_masks_from_dir(mask_dir, seg_name_map)
        else:
            all_masks = self.segmenter.segment_scene(
                image, list(self.configs.values()))

        results = []
        for name, masks in all_masks.items():
            if not masks:
                print(f"  [{name}] no instances detected")
                continue

            for inst_idx, mask in enumerate(masks):
                label = name if len(masks) == 1 else f"{name}[{inst_idx}]"
                print(f"  [{label}] estimating pose...")
                t0 = time.time()
                try:
                    result = self._estimate_one(
                        name, image, mask, fx, fy, top_k, refine)
                    result["object_name"]   = name
                    result["instance_idx"]  = inst_idx
                    result["mask"]          = mask
                    result["inference_time_s"] = time.time() - t0

                    # Metric position from apparent object size in image
                    cfg_obj = self.configs[name]
                    if cfg_obj.diameter_m is not None:
                        ys_m, xs_m = np.where(mask > 0)
                        if len(ys_m):
                            u_c   = float(xs_m.mean())
                            v_c   = float(ys_m.mean())
                            r_px  = max(1.0, float(np.sqrt(np.count_nonzero(mask) / np.pi)))
                            cx_i  = image.shape[1] / 2.0
                            cy_i  = image.shape[0] / 2.0
                            z_m   = fx * (cfg_obj.diameter_m / 2.0) / r_px
                            result["position_m"] = np.array([
                                (u_c - cx_i) / fx * z_m,
                                (v_c - cy_i) / fy * z_m,
                                z_m,
                            ])

                    results.append(result)
                    self._print_result(label, result)
                except Exception as e:
                    print(f"    ✗ {e}")
                    results.append({"object_name": name,
                                    "instance_idx": inst_idx,
                                    "error": str(e)})
        return results

    def process_multi_view(
        self,
        views: list[tuple[np.ndarray, np.ndarray]],
        camera_fx: Optional[float] = None,
        camera_fy: Optional[float] = None,
        top_k: int = 5,
        refine: bool = True,
        mask_dirs: Optional[list[str]] = None,
        seg_name_map: Optional[dict[str, str]] = None,
    ) -> list[dict]:
        """
        Multi-view pose estimation with triangulated 3D positions.

        views:     [(image_RGB, cam_T_world_4x4), ...]
                   cam_T_world is the world-to-camera transform (same convention
                   as pose_w2c elsewhere). All views must share the same world
                   frame — the first camera is typically the identity.
        mask_dirs: optional list of pre-computed mask directories, one per view.
                   Pass None entries to fall back to HSV segmentation for that view.

        Runs single-view pose estimation on each view, then triangulates the
        3D centroid of every detected object from all views that saw it.
        Results include 'triangulated_position_m' (world frame) when ≥2 views
        detect the object, in addition to the single-view 'position_m'.
        Rotation and mask are taken from the highest-confidence single view.
        The result list is ordered to match view 0's detections so that masks
        align when visualising against the first view's image.
        """
        fx = camera_fx or self.fx
        fy = camera_fy or self.fy
        H, W = views[0][0].shape[:2]
        cx, cy = W / 2.0, H / 2.0

        print("\n══════════════════════════════════════════")
        print(f"  MULTI-VIEW ESTIMATION  ({len(views)} views)")
        print("══════════════════════════════════════════")

        # Run the single-view pipeline on every view
        per_view: list[tuple[np.ndarray, list[dict]]] = []
        for i, (image, cam_T_world) in enumerate(views):
            print(f"\n── View {i + 1}/{len(views)} ──")
            md = (mask_dirs[i] if mask_dirs and i < len(mask_dirs)
                  else None)
            results = self.process_scene(
                image,
                camera_fx=fx,
                camera_fy=fy,
                top_k=top_k,
                refine=refine,
                mask_dir=md,
                seg_name_map=seg_name_map,
            )
            per_view.append((cam_T_world, results))

        # Group all detections by (object_name, instance_idx)
        by_key: dict[tuple, list[tuple]] = {}
        for view_idx, (cam_T_world, results) in enumerate(per_view):
            for r in results:
                if "error" in r:
                    continue
                key = (r["object_name"], r.get("instance_idx", 0))
                by_key.setdefault(key, []).append(
                    (view_idx, r, cam_T_world))

        # Build final results, preferring view-0 data for mask/visualisation
        final: list[dict] = []
        for (name, inst_idx), detections in by_key.items():
            view0 = [d for d in detections if d[0] == 0]
            base_detection = (view0[0] if view0
                              else max(detections,
                                       key=lambda x: x[1].get("confidence", 0.0)))
            result = dict(base_detection[1])

            # Collect per-view centroids for triangulation
            centroids, cam_poses = [], []
            for view_idx, r, cam_T_world in detections:
                mask = r.get("mask")
                if mask is None:
                    continue
                ys, xs = np.where(mask > 0)
                if not len(ys):
                    continue
                centroids.append((float(xs.mean()), float(ys.mean())))
                cam_poses.append(cam_T_world)

            if len(centroids) >= 2:
                X_world = triangulate_dlt(centroids, cam_poses, fx, fy, cx, cy)
                result["triangulated_position_m"] = X_world
                result["n_views_triangulated"] = len(centroids)
                print(f"  [{name}] triangulated from {len(centroids)} views: "
                      f"{X_world.round(4)} m")
            else:
                print(f"  [{name}] seen in only 1 view — "
                      f"triangulated_position_m not available")

            final.append(result)

        return final

    def _estimate_one(self, name, image, mask,
                      fx, fy, top_k, refine) -> dict:
        cfg = self.configs[name]
        db  = self.databases[name]
        refiner = self.refiners[name]

        # Crop to mask bounding box
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            raise ValueError("Empty mask")
        pad = 10
        y0 = max(0, ys.min()-pad);  y1 = min(image.shape[0], ys.max()+pad)
        x0 = max(0, xs.min()-pad);  x1 = min(image.shape[1], xs.max()+pad)
        crop      = image[y0:y1, x0:x1]
        mask_crop = mask[y0:y1, x0:x1]

        # Coarse retrieval
        feats      = self.extractor.extract(crop, mask=mask_crop)
        candidates = db.query(feats["cls"].numpy(), k=top_k)
        coarse     = candidates[0]["pose_w2c"]

        if not refine:
            return self._format(coarse, coarse,
                                float(candidates[0]["distance"]),
                                cfg, skip_rotation=refiner.is_symmetric)

        # Refinement
        best_pose, best_score = coarse, float("inf")
        for cand in candidates:
            p, s = refiner.refine(cand["pose_w2c"], crop, fx=fx, fy=fy)
            if s < best_score:
                best_score = s;  best_pose = p

        return self._format(best_pose, coarse, best_score,
                            cfg, skip_rotation=refiner.is_symmetric)

    def _format(self, pose_w2c, coarse, score, cfg,
                skip_rotation=False) -> dict:
        R = pose_w2c[:3, :3]
        t = pose_w2c[:3, 3]

        # Confidence: map edge score to 0-1 (lower score = higher confidence)
        confidence = float(np.clip(1.0 / (1.0 + score / 10.0), 0, 1))

        result = {
            "pose_w2c":           pose_w2c,
            "rotation_matrix":    R if not skip_rotation else None,
            "translation":        t,
            "euler_angles_zyx_deg": (Rotation.from_matrix(R).as_euler("zyx", degrees=True)
                                     if not skip_rotation else None),
            "translation_m":      (t * cfg.diameter_m
                                   if cfg.diameter_m else None),
            "diameter_m":         cfg.diameter_m,
            "confidence":         confidence,
            "score":              score,
            "is_symmetric":       skip_rotation,
            "coarse_pose_w2c":    coarse,
        }
        return result

    def _print_result(self, label, r):
        if r.get("is_symmetric"):
            t = r["translation"]
            print(f"    ✓ [{label}] position={t.round(3)}  "
                  f"(symmetric — rotation skipped)  "
                  f"conf={r['confidence']:.2f}  "
                  f"({r['inference_time_s']:.1f}s)")
        else:
            e = r["euler_angles_zyx_deg"]
            t = r["translation"]
            print(f"    ✓ [{label}] "
                  f"Z={e[0]:.1f}° Y={e[1]:.1f}° X={e[2]:.1f}°  "
                  f"t={t.round(3)}  "
                  f"conf={r['confidence']:.2f}  "
                  f"({r['inference_time_s']:.1f}s)")
