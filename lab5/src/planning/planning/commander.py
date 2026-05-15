"""
commander.py — orchestrates the full pipeline:
  1. Wait for arm_circler to finish orbit (/orbit_done)
  2. Run YOLO+SAM2 segmentation on captured images
  3. Run 6D pose estimation; display detected objects + positions
  4. Wait for user Enter → activate pick-and-place (/start_pick_place)
  5. Accept class-switch commands between pick cycles
"""

import json
import os
import sys
import subprocess
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PointStamped

# ── path constants ────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

def _find_project_root(start: str) -> str:
    """Walk up from start until finding a directory that contains 'armcircler/' as a child."""
    path = start
    for _ in range(12):
        if os.path.isdir(os.path.join(path, 'armcircler')):
            return path
        path = os.path.dirname(path)
    raise RuntimeError("Cannot locate project root (no parent with 'armcircler/' found)")

_PROJECT_ROOT = _find_project_root(_HERE)
_SIXD_DIR     = os.path.join(_PROJECT_ROOT, 'lab5', 'src', 'planning', 'planning', '6D_poses')
_SEG_SCRIPT   = os.path.join(_PROJECT_ROOT, 'lab5', 'src', 'perception', 'perception', 'segment_batch.py')
_YOLO_WEIGHTS = os.path.join(_PROJECT_ROOT, 'lab5', 'updated.pt')
_SAM2_PYTHON  = os.path.join(_PROJECT_ROOT, 'sam2', 'sam2_env', 'bin', 'python')


class Commander(Node):
    def __init__(self):
        super().__init__('commander')

        self._start_pub = self.create_publisher(Bool,   '/start_pick_place',   1)
        self._class_pub = self.create_publisher(String, '/set_target_class',   1)
        self._tasks_pub = self.create_publisher(String, '/planned_pick_tasks', 1)

        _latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._orbit_sub = self.create_subscription(
            Bool, '/orbit_done', self._on_orbit_done, _latched
        )
        self._plate_sub = self.create_subscription(
            PointStamped, '/detected_plate_point', self._on_plate_point, 10
        )

        self._pose_results = []   # filled after estimation completes
        self._plate_pos    = None # (x, y, z) of plate in base_link, from live detection

    def _on_plate_point(self, msg: PointStamped):
        if self._plate_pos is None:
            self._plate_pos = (msg.point.x, msg.point.y, msg.point.z)

    def _sorted_tasks(self):
        """Return pose results sorted by Z height ascending (lowest object first)."""
        return sorted(self._pose_results, key=lambda r: float(r['position_base_link_m'][2]))

    # ── orbit signal ──────────────────────────────────────────────────────────

    def _on_orbit_done(self, msg: Bool):
        if not msg.data:
            return
        self.get_logger().info("Orbit done — launching pose pipeline.")
        threading.Thread(target=self._run_pose_pipeline, daemon=True).start()

    # ── pose pipeline (runs in background thread) ─────────────────────────────

    def _run_pose_pipeline(self):
        cwd        = os.getcwd()
        images_dir = os.path.join(cwd, 'captured_images')
        masks_dir  = os.path.join(cwd, 'segmented')
        poses_path = os.path.join(cwd, 'poses.npy')
        output_dir = os.path.join(cwd, 'results')

        # ── Step 1: segmentation ──────────────────────────────────────────────
        print("\n[Commander] Running segmentation (YOLO + SAM2)...")
        seg_ok = self._run_subprocess(
            [_SAM2_PYTHON, _SEG_SCRIPT,
             '--input_dir',  images_dir,
             '--output_dir', masks_dir,
             '--yolo',       _YOLO_WEIGHTS,
             '--conf',       '0.75'],
            label='segmentation',
        )
        if not seg_ok:
            print("[Commander] Segmentation failed — skipping pose estimation.")
            self._prompt_user()
            return

        # ── Step 2: 6D pose estimation ────────────────────────────────────────
        print("[Commander] Running 6D pose estimation...")
        code = (
            "import sys, os; "
            f"sys.path.insert(0, r'{_SIXD_DIR}'); os.chdir(r'{_SIXD_DIR}'); "
            "from run import run_pose_estimation; "
            f"run_pose_estimation("
            f"  use_multi_view=True,"
            f"  poses_path=r'{poses_path}',"
            f"  images_dir=r'{images_dir}',"
            f"  masks_dir=r'{masks_dir}',"
            f"  output_dir=r'{output_dir}',"
            f")"
        )
        pose_ok = self._run_subprocess(
            [_SAM2_PYTHON, '-c', code],
            label='pose estimation',
        )
        if not pose_ok:
            print("[Commander] Pose estimation failed — continuing without 6D poses.")

        # ── Step 3: load + display results ───────────────────────────────────
        results_path = os.path.join(output_dir, 'object_poses.npy')
        if pose_ok and os.path.isfile(results_path):
            raw = list(np.load(results_path, allow_pickle=True))
            self._pose_results = self._filter_pose_results(raw)
            self._print_objects()
        else:
            self._pose_results = []
            print("[Commander] No pose results available.")

        self._prompt_user()

    def _run_subprocess(self, cmd, label):
        try:
            subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"[Commander] {label} subprocess error: {e}")
            return False

    # ── user interaction (blocks in background thread — stdin is safe here) ───

    def _filter_pose_results(self, results):
        """Remove bad triangulations; keep one best instance per class."""
        valid = []
        for r in results:
            pos  = r.get('position_base_link_m')
            conf = float(r.get('confidence', 0.0))
            if pos is None:
                continue
            z = float(pos[2])
            if z < -0.03 or z > 0.15:   # outside plausible table-surface range
                continue
            if conf <= 0.6:
                continue
            valid.append(r)

        # Per class, keep only the highest-confidence instance
        by_class = {}
        for r in valid:
            name = r.get('object_name', '?')
            conf = float(r.get('confidence', 0.0))
            if name not in by_class or conf > float(by_class[name].get('confidence', 0.0)):
                by_class[name] = r

        n_dropped = len(results) - len(by_class)
        if n_dropped:
            print(f"[Commander] Filtered {n_dropped} low-quality detections "
                  f"({len(by_class)} unique objects remain).")
        return list(by_class.values())

    def _print_objects(self):
        print("\n[Commander] ══ Detected Objects ══")
        if not self._pose_results:
            print("  (none)")
        for i, r in enumerate(self._pose_results):
            name = r.get('object_name', '?')
            pos  = r.get('position_base_link_m')
            conf = r.get('confidence', 0.0)
            if pos is not None:
                print(
                    f"  [{i}] {name:<20s}  "
                    f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})  "
                    f"conf={conf:.2f}"
                )
            else:
                print(f"  [{i}] {name:<20s}  (no position)")
        print()

    def _prompt_user(self):
        try:
            input("[Commander] Press Enter to activate pick-and-place: ")
        except EOFError:
            return

        # Publish sorted task list so main.py can execute picks in order
        # using the pre-computed 6D positions instead of live detection.
        ordered = self._sorted_tasks()
        if ordered:
            tasks = []
            for r in ordered:
                pos = r["position_base_link_m"]
                tasks.append({
                    "object_name": r["object_name"],
                    "position":    [float(v) for v in pos],
                })
            task_msg = String()
            task_msg.data = json.dumps(tasks)
            self._tasks_pub.publish(task_msg)
            print("[Commander] Pick order:")
            for i, t in enumerate(tasks):
                p = t["position"]
                print(f"  [{i+1}] {t['object_name']:<20s}  pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})")
            print()

        msg = Bool()
        msg.data = True
        self._start_pub.publish(msg)
        print("[Commander] Pick-and-place active — executing automatically.")


def main(args=None):
    rclpy.init(args=args)
    node = Commander()

    print("[Commander] Waiting for orbit to complete...")
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    spin_thread.join()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
