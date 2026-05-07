"""
commander.py — orchestrates the full pipeline:
  1. Wait for arm_circler to finish orbit (/orbit_done)
  2. Run YOLO+SAM2 segmentation on captured images
  3. Run 6D pose estimation; display detected objects + positions
  4. Wait for user Enter → activate pick-and-place (/start_pick_place)
  5. Accept class-switch commands between pick cycles
"""

import os
import sys
import subprocess
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import String, Bool

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
_SIXD_DIR     = os.path.join(_PROJECT_ROOT, '6D_poses')
_SEG_SCRIPT   = os.path.join(_PROJECT_ROOT, 'armcircler', 'segment_batch.py')
_YOLO_WEIGHTS = os.path.join(_PROJECT_ROOT, 'lab5', 'updated.pt')
_SAM2_PYTHON  = os.path.join(_PROJECT_ROOT, 'sam2', 'sam2_env', 'bin', 'python')


class Commander(Node):
    def __init__(self):
        super().__init__('commander')

        self._start_pub = self.create_publisher(Bool,   '/start_pick_place', 1)
        self._class_pub = self.create_publisher(String, '/set_target_class',  1)

        _latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._orbit_sub = self.create_subscription(
            Bool, '/orbit_done', self._on_orbit_done, _latched
        )

        self._pose_results = []   # filled after estimation completes

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
             '--yolo',       _YOLO_WEIGHTS],
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
            self._pose_results = list(np.load(results_path, allow_pickle=True))
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

        msg = Bool()
        msg.data = True
        self._start_pub.publish(msg)
        print("[Commander] Pick-and-place active!\n")

        while rclpy.ok():
            try:
                user_in = input(
                    "[Commander] Class to pick  (or 'list' / 'q' to quit): "
                ).strip()
            except EOFError:
                break

            if user_in == 'q':
                break
            elif user_in == 'list':
                self._print_objects()
            elif user_in:
                msg = String()
                msg.data = user_in
                self._class_pub.publish(msg)
                print(f"[Commander] Target → {user_in}")


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
