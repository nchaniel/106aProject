"""
generate_guide.py
-----------------
Generates a comprehensive PDF guide for the MEC106A Lab 5 codebase.
Run:  python generate_guide.py
Output: lab5_code_guide.pdf
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus.flowables import Flowable

# ─────────────────────────────────────────────────────────
# Styles
# ─────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def make_styles():
    s = {}
    s['title'] = ParagraphStyle('MyTitle',
        fontSize=26, fontName='Helvetica-Bold',
        spaceAfter=12, spaceBefore=0,
        alignment=TA_CENTER, textColor=colors.HexColor('#1a1a2e'))
    s['subtitle'] = ParagraphStyle('MySubtitle',
        fontSize=13, fontName='Helvetica',
        spaceAfter=6, alignment=TA_CENTER,
        textColor=colors.HexColor('#16213e'))
    s['h1'] = ParagraphStyle('H1',
        fontSize=18, fontName='Helvetica-Bold',
        spaceBefore=18, spaceAfter=6,
        textColor=colors.HexColor('#0f3460'),
        borderPad=2)
    s['h2'] = ParagraphStyle('H2',
        fontSize=14, fontName='Helvetica-Bold',
        spaceBefore=12, spaceAfter=4,
        textColor=colors.HexColor('#16213e'))
    s['h3'] = ParagraphStyle('H3',
        fontSize=11, fontName='Helvetica-Bold',
        spaceBefore=8, spaceAfter=3,
        textColor=colors.HexColor('#533483'))
    s['h4'] = ParagraphStyle('H4',
        fontSize=10, fontName='Helvetica-Bold',
        spaceBefore=6, spaceAfter=2,
        textColor=colors.HexColor('#0f3460'))
    s['body'] = ParagraphStyle('Body',
        fontSize=10, fontName='Helvetica',
        spaceAfter=4, leading=14,
        alignment=TA_JUSTIFY)
    s['bullet'] = ParagraphStyle('Bullet',
        fontSize=10, fontName='Helvetica',
        spaceAfter=2, leading=13,
        leftIndent=18, firstLineIndent=-10)
    s['bullet2'] = ParagraphStyle('Bullet2',
        fontSize=9.5, fontName='Helvetica',
        spaceAfter=2, leading=12,
        leftIndent=34, firstLineIndent=-10)
    s['code'] = ParagraphStyle('Code',
        fontSize=8.5, fontName='Courier',
        spaceAfter=2, leading=12,
        leftIndent=12,
        backColor=colors.HexColor('#f4f4f4'),
        textColor=colors.HexColor('#1a1a1a'))
    s['code_inline'] = ParagraphStyle('CodeInline',
        fontSize=9, fontName='Courier',
        spaceAfter=3, leading=12)
    s['note'] = ParagraphStyle('Note',
        fontSize=9.5, fontName='Helvetica-Oblique',
        spaceAfter=4, leading=13,
        leftIndent=12,
        textColor=colors.HexColor('#555555'))
    s['warning'] = ParagraphStyle('Warning',
        fontSize=9.5, fontName='Helvetica-Bold',
        spaceAfter=4, leading=13,
        leftIndent=12,
        textColor=colors.HexColor('#8B0000'))
    s['toc'] = ParagraphStyle('TOC',
        fontSize=10, fontName='Helvetica',
        spaceAfter=3, leading=14,
        leftIndent=0)
    s['toc2'] = ParagraphStyle('TOC2',
        fontSize=9.5, fontName='Helvetica',
        spaceAfter=2, leading=13,
        leftIndent=18)
    return s

ST = make_styles()

def H1(text): return Paragraph(text, ST['h1'])
def H2(text): return Paragraph(text, ST['h2'])
def H3(text): return Paragraph(text, ST['h3'])
def H4(text): return Paragraph(text, ST['h4'])
def P(text):  return Paragraph(text, ST['body'])
def B(text):  return Paragraph(f'• {text}', ST['bullet'])
def B2(text): return Paragraph(f'◦ {text}', ST['bullet2'])
def Code(text): return Paragraph(text, ST['code'])
def Note(text): return Paragraph(f'Note: {text}', ST['note'])
def Warn(text): return Paragraph(f'⚠  {text}', ST['warning'])
def SP(n=6):   return Spacer(1, n)
def HR():      return HRFlowable(width='100%', thickness=1, color=colors.HexColor('#cccccc'), spaceAfter=4, spaceBefore=4)

def code_block(lines):
    """Return a list of flowables for a monospace code block."""
    items = []
    bg = colors.HexColor('#f4f4f4')
    border = colors.HexColor('#cccccc')
    row_data = [[Paragraph('<br/>'.join(
        l.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        for l in lines
    ), ST['code'])]]
    t = Table(row_data, colWidths=[6.5*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), bg),
        ('BOX', (0,0), (-1,-1), 0.5, border),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    items.append(t)
    items.append(SP(4))
    return items

def topic_table(rows, col_widths=None):
    """Build a styled table. rows = list of lists (first row = headers)."""
    if col_widths is None:
        col_widths = [2*inch, 1.4*inch, 1.1*inch, 2.5*inch]
    header_style = ParagraphStyle('TH', fontSize=9, fontName='Helvetica-Bold', textColor=colors.white)
    cell_style   = ParagraphStyle('TD', fontSize=8.5, fontName='Helvetica', leading=11)
    code_style   = ParagraphStyle('TDCode', fontSize=8, fontName='Courier', leading=10)

    table_data = []
    for i, row in enumerate(rows):
        if i == 0:
            table_data.append([Paragraph(c, header_style) for c in row])
        else:
            styled = []
            for j, cell in enumerate(row):
                st = code_style if j == 0 else cell_style
                styled.append(Paragraph(str(cell), st))
            table_data.append(styled)

    t = Table(table_data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f3460')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f0f4ff')]),
        ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor('#cccccc')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    return t

def info_box(title, content_paras, color='#e8f4f8', border_color='#0f3460'):
    """Colored info box with a bold title."""
    items = [Paragraph(f'<b>{title}</b>', ParagraphStyle('BoxTitle', fontSize=10, fontName='Helvetica-Bold', textColor=colors.HexColor(border_color)))]
    items += content_paras
    row_data = [items]
    t = Table([[Paragraph('<br/>'.join([f'<b>{title}</b>'] + [c if isinstance(c, str) else '' for c in content_paras]), ParagraphStyle('Box', fontSize=9.5, fontName='Helvetica', leading=13))]], colWidths=[6.5*inch])
    # Use a simpler approach: just a styled table with all paragraphs as one cell
    all_p = [Paragraph(f'<b>{title}</b>', ParagraphStyle('BT', fontSize=10, fontName='Helvetica-Bold', textColor=colors.HexColor(border_color), spaceAfter=3))]
    all_p += content_paras
    inner = Table([[p] for p in all_p], colWidths=[6.3*inch])
    inner.setStyle(TableStyle([
        ('TOPPADDING', (0,0), (-1,-1), 1),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    outer = Table([[inner]], colWidths=[6.5*inch])
    outer.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(color)),
        ('BOX', (0,0), (-1,-1), 1.5, colors.HexColor(border_color)),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    return [outer, SP(6)]

# ─────────────────────────────────────────────────────────
# Document content builder
# ─────────────────────────────────────────────────────────

def build_content():
    doc = []

    # ── TITLE PAGE ─────────────────────────────────────────
    doc.append(SP(60))
    doc.append(Paragraph('MEC106A Lab 5', ST['title']))
    doc.append(SP(8))
    doc.append(Paragraph('Vision-Guided Pick and Place — Complete Code Guide', ST['subtitle']))
    doc.append(SP(6))
    doc.append(Paragraph('UR7e Robot · ROS2 · YOLO · SAM2 · 6D Pose Estimation', ST['subtitle']))
    doc.append(SP(30))
    doc.append(HR())
    doc.append(SP(10))
    doc.append(P('This document provides an in-depth reference for every node, function, topic, and call chain in the Lab 5 codebase. It covers the full pipeline from camera input to robot execution, including the 6D pose estimation offline stage, the real-time detection loop, the arm-orbiting inspection phase, the pick-and-place state machine, and the commander orchestration layer.'))
    doc.append(PageBreak())

    # ── TABLE OF CONTENTS ──────────────────────────────────
    doc.append(H1('Table of Contents'))
    toc_entries = [
        ('1', 'System Architecture Overview', 3),
        ('2', 'ROS2 Node Inventory', 4),
        ('3', 'Complete Topic & Service Map', 5),
        ('4', 'Node Deep-Dive: DetectionNode  (perception)', 6),
        ('4.1', 'YOLODetector helper class', 7),
        ('4.2', 'pixel_to_world helpers', 8),
        ('5', 'Node Deep-Dive: ConstantTransformPublisher  (static_tf_transform.py)', 9),
        ('6', 'Node Deep-Dive: IKPlanner  (ik.py)', 10),
        ('7', 'Node Deep-Dive: ArmCircler  (arm_circler.py)', 11),
        ('8', 'Node Deep-Dive: UR7e_CubeGrasp  (main.py)', 13),
        ('8.1', 'Subscriptions and publishers', 14),
        ('8.2', 'Job queue mechanism', 15),
        ('8.3', 'Two-phase centroid refinement', 16),
        ('8.4', 'Task-list auto-execution', 17),
        ('8.5', 'State machine: busy / refining / going_home', 18),
        ('8.6', 'Pick sequence — full call chain', 19),
        ('9', 'Node Deep-Dive: Commander  (commander.py)', 20),
        ('10', '6D Pose Estimation Pipeline', 22),
        ('10.1', 'run.py — entry point', 23),
        ('10.2', 'pose_estimator.py — five sub-components', 24),
        ('10.3', 'RANSAC + weighted DLT triangulation', 27),
        ('10.4', 'constants.py — configuration', 28),
        ('11', 'End-to-End Pipeline Call Chain', 29),
        ('12', 'Per-Class Grasp Offsets Reference', 32),
        ('13', 'Key Design Patterns', 33),
        ('14', 'Known Issues and Watch-outs', 35),
    ]
    for num, title, _ in toc_entries:
        indent = 18 if '.' in num else 0
        prefix = f'{num}.' if '.' not in num else f'  {num}'
        doc.append(Paragraph(f'{prefix}&nbsp;&nbsp;&nbsp;{title}', ST['toc2'] if '.' in num else ST['toc']))
    doc.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 1: ARCHITECTURE OVERVIEW
    # ══════════════════════════════════════════════════════════
    doc.append(H1('1. System Architecture Overview'))
    doc.append(P('The Lab 5 system is a fully autonomous pick-and-place pipeline for a UR7e robot arm. It integrates a RealSense RGBD camera, a YOLO object detector, SAM2 instance segmentation, and a custom 6D pose estimator to locate fruit objects on a table and place them onto a plate.'))
    doc.append(SP(4))
    doc.append(P('The system runs in two distinct phases:'))
    doc.append(B('<b>Inspection Phase (offline data collection):</b> The ArmCircler node drives the arm around the plate in a 2-row orbit (40 waypoints total), capturing images and recording end-effector poses. When the orbit completes, the Commander node triggers YOLO+SAM2 segmentation followed by multi-view RANSAC triangulation to produce 3D object positions in the robot\'s base_link frame.'))
    doc.append(B('<b>Execution Phase (pick and place):</b> The Commander publishes a JSON task list and signals main.py to start. UR7e_CubeGrasp reads the tasks, moves to pre-pregrasp above each 6D estimate, waits for live YOLO to refine the exact centroid, then executes a full pick-place trajectory through IK + MoveIt.'))
    doc.append(SP(6))

    doc.append(H2('High-Level Data Flow'))
    flow_lines = [
        'RealSense Camera',
        '  ├─ /camera/camera/color/image_raw      ──► DetectionNode.image_callback()',
        '  ├─ /camera/camera/depth/color/points   ──► DetectionNode.cloud_callback()',
        '  └─ /camera/camera/color/camera_info    ──► DetectionNode.camera_info_callback()',
        '                                               │',
        '                      /detected_class         │  publish on each frame',
        '                      /detected_pick_point ───┤  (best non-plate detection)',
        '                      /detected_plate_point ──┘  (plate always published)',
        '',
        'ArmCircler',
        '  ├─ subscribes: /detected_plate_point (triggers orbit once)',
        '  ├─ subscribes: /joint_states (seeds IK)',
        '  ├─ saves: captured_images/captured_image_{1..40}.jpg',
        '  ├─ saves: poses.npy  [x,y,z,qx,qy,qz,qw] per waypoint',
        '  └─ publishes: /orbit_done Bool(True)  ──► Commander',
        '',
        'Commander (separate terminal, stdin loop)',
        '  ├─ on /orbit_done: subprocess → segment_batch.py',
        '  ├─ on /orbit_done: subprocess → 6D_poses/run.py',
        '  ├─ loads results/object_poses.npy, prints to terminal',
        '  ├─ user types index/name → publishes /set_target_class + /target_drop_point',
        '  └─ user presses Enter → publishes /start_pick_place',
        '',
        'UR7e_CubeGrasp  (main.py)',
        '  ├─ on /start_pick_place: _go_home() then auto-run task list',
        '  ├─ on /target_drop_point: arm pick triggered (live detection path)',
        '  ├─ on /detected_pick_point: cube_callback() → two-phase IK',
        '  └─ IKPlanner → /compute_ik (MoveIt) → /plan_kinematic_path',
        '                → /scaled_joint_trajectory_controller (action)',
        '                → /toggle_gripper (service)',
    ]
    doc += code_block(flow_lines)

    # ══════════════════════════════════════════════════════════
    # SECTION 2: NODE INVENTORY
    # ══════════════════════════════════════════════════════════
    doc.append(H1('2. ROS2 Node Inventory'))
    doc.append(P('Six Python ROS2 nodes constitute the system, split across two packages: <b>perception</b> and <b>planning</b>.'))
    doc.append(SP(4))

    node_rows = [
        ['Node Name', 'Package / File', 'Launched By', 'Role'],
        ['detection_node', 'perception/detection_node.py', 'lab5_bringup.launch.py', 'Real-time YOLO detection; 3D centroid extraction; publishes pick + plate points'],
        ['ik_planner', 'planning/ik.py', 'lab5_bringup.launch.py', 'MoveIt IK wrapper — compute_ik() and plan_to_joints()'],
        ['constant_tf_publisher', 'planning/static_tf_transform.py', 'lab5_bringup.launch.py', 'Broadcasts static wrist_3_link → camera_link TF at startup'],
        ['arm_circler', 'planning/arm_circler.py', 'lab5_bringup.launch.py (optional)', 'Orbits arm around plate; captures images + poses; signals orbit_done'],
        ['cube_grasp', 'planning/main.py', 'lab5_bringup.launch.py', 'Top-level pick-and-place state machine; owns job queue and busy semaphore'],
        ['commander', 'planning/commander.py', 'ros2 run planning commander', 'CLI orchestrator; runs pose pipeline subprocesses; user interaction loop'],
    ]
    doc.append(topic_table(node_rows, [1.4*inch, 1.6*inch, 1.6*inch, 2.4*inch]))
    doc.append(SP(8))

    doc.append(H2('Launch Configuration'))
    doc.append(P('The launch file <b>src/planning/launch/lab5_bringup.launch.py</b> starts all nodes except the camera and commander. Key launch arguments:'))
    launch_rows = [
        ['Argument', 'Default', 'Effect'],
        ['robot_ip', '192.168.1.102', 'UR7e controller IP for MoveIt driver'],
        ['target_class', '""', 'Initial YOLO target class (empty = highest confidence)'],
        ['skip_circler', 'false', 'Skip orbit phase; UR7e_CubeGrasp publishes /orbit_done immediately (TRANSIENT_LOCAL QoS so Commander receives it late-join)'],
        ['launch_rviz', 'true', 'Launch RViz with MoveIt config'],
        ['shutdown_on_exit', 'true', 'Kill all nodes if any process exits'],
    ]
    doc.append(topic_table(launch_rows, [1.4*inch, 1.0*inch, 4.1*inch]))
    doc.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 3: TOPIC & SERVICE MAP
    # ══════════════════════════════════════════════════════════
    doc.append(H1('3. Complete Topic and Service Map'))
    doc.append(P('Every topic, service, and action used in the system is listed below. The <i>Direction</i> column shows which node produces (→) and which node consumes it.'))
    doc.append(SP(4))

    topic_rows = [
        ['Topic / Service / Action', 'Type', 'Direction', 'Purpose'],
        ['/camera/camera/color/image_raw', 'sensor_msgs/Image', 'Camera → DetectionNode, ArmCircler, main.py', 'RGB frames; YOLO runs on each frame'],
        ['/camera/camera/depth/color/points', 'sensor_msgs/PointCloud2', 'Camera → DetectionNode', 'Depth point cloud for 3D centroid extraction'],
        ['/camera/camera/color/camera_info', 'sensor_msgs/CameraInfo', 'Camera → DetectionNode', 'Intrinsics (fx, fy, cx, cy) for deprojection'],
        ['/detected_pick_point', 'geometry_msgs/PointStamped', 'DetectionNode → main.py', '3D centroid of best non-plate object in base_link'],
        ['/detected_class', 'std_msgs/String', 'DetectionNode → main.py', 'YOLO class label corresponding to the pick point'],
        ['/detected_plate_point', 'geometry_msgs/PointStamped', 'DetectionNode → main.py, ArmCircler, Commander', 'Plate centroid in base_link; always published regardless of target_class'],
        ['/set_target_class', 'std_msgs/String', 'Commander → DetectionNode', 'Runtime switch of which YOLO class to target'],
        ['/target_drop_point', 'geometry_msgs/PointStamped', 'Commander → main.py', 'Drop position + arms main.py for one live-detection pick (_pick_triggered=True)'],
        ['/start_pick_place', 'std_msgs/Bool', 'Commander / main.py → main.py', 'Activates pick-place (sets pick_place_enabled=True) and calls _go_home()'],
        ['/orbit_done', 'std_msgs/Bool', 'ArmCircler / main.py → Commander', 'Signals orbit complete; TRANSIENT_LOCAL QoS so late-joiners get it'],
        ['/pick_task_list', 'std_msgs/String (JSON)', 'Commander → main.py', 'Pre-computed ordered task list [{object_name, pick:[x,y,z], drop:[x,y,z]}, ...]'],
        ['/joint_states', 'sensor_msgs/JointState', 'Robot → main.py, ArmCircler', 'Live joint angles; seeds IK calls + triggers initial _go_home()'],
        ['/scaled_joint_trajectory_controller/follow_joint_trajectory', 'control_msgs/FollowJointTrajectory (Action)', 'main.py, ArmCircler → Controller', 'Executes planned joint trajectory on the physical robot'],
        ['/toggle_gripper', 'std_srvs/Trigger (Service)', 'main.py → Gripper driver', 'Toggle gripper open/closed'],
        ['/compute_ik', 'moveit_msgs/GetPositionIK (Service)', 'IKPlanner → MoveIt', 'Compute joint angles for a target Cartesian pose'],
        ['/plan_kinematic_path', 'moveit_msgs/GetMotionPlan (Service)', 'IKPlanner → MoveIt', 'Plan collision-free trajectory to target joint config (RRTConnect)'],
    ]
    doc.append(topic_table(topic_rows, [2.1*inch, 1.5*inch, 1.4*inch, 1.9*inch]))
    doc.append(SP(6))

    doc.append(H2('QoS Profiles'))
    doc.append(P('Most topics use default (KEEP_LAST, VOLATILE) QoS. The following use <b>TRANSIENT_LOCAL</b> (latched) so that late-joining subscribers receive the last published message:'))
    doc.append(B('<b>/orbit_done</b> — published by ArmCircler (or main.py when skip_circler=true); subscribed by Commander. Ensures Commander receives the signal even if it starts after the message is published.'))
    doc.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 4: DetectionNode
    # ══════════════════════════════════════════════════════════
    doc.append(H1('4. Node Deep-Dive: DetectionNode'))
    doc.append(P('<b>File:</b> src/perception/perception/detection_node.py &nbsp;|&nbsp; <b>Node name:</b> detection_node'))
    doc.append(P('DetectionNode is the primary perception node. It runs at camera framerate (30 Hz) and produces 3D object centroids in the robot\'s base_link frame. It is the only node that reads the live camera stream during normal operation.'))
    doc.append(SP(4))

    doc.append(H2('Subscriptions'))
    doc.append(B('<b>/camera/camera/color/image_raw</b> → <tt>image_callback(msg)</tt> — main processing callback; runs YOLO on every frame'))
    doc.append(B('<b>/camera/camera/depth/color/points</b> → <tt>cloud_callback(msg)</tt> — buffers latest PointCloud2'))
    doc.append(B('<b>/camera/camera/color/camera_info</b> → <tt>camera_info_callback(msg)</tt> — buffers intrinsics'))
    doc.append(B('<b>/set_target_class</b> → <tt>_set_target_class_cb(msg)</tt> — updates self.target_class at runtime'))
    doc.append(SP(4))

    doc.append(H2('Publishers'))
    doc.append(B('<b>/detected_pick_point</b> (PointStamped) — 3D centroid of highest-confidence non-plate object in base_link'))
    doc.append(B('<b>/detected_plate_point</b> (PointStamped) — 3D centroid of plate; always published regardless of target_class filter'))
    doc.append(B('<b>/detected_class</b> (String) — class label for the published pick point'))
    doc.append(SP(4))

    doc.append(H2('image_callback() — Core Processing Loop'))
    doc.append(P('Called on every RGB frame. Performs the following steps:'))
    cb_steps = [
        '1.  Convert ROS Image → OpenCV BGR via CvBridge',
        '2.  Call YOLODetector.detect(cv_image) → list of detections',
        '3.  For each detection:',
        '      a. Skip if target_class is set AND class_name != target_class AND class_name != "plate"',
        '         (plate is always processed regardless of target_class)',
        '      b. Call get_centroid_from_cloud_bbox(cloud, camera_info, bbox)',
        '         → reprojects PointCloud2 into image coords, averages XYZ inside bbox',
        '      c. Call transform_point(tf_buffer, pt_cam, "base_link")',
        '         → TF2 lookup to convert camera_frame → base_link',
        '      d. If class == "plate": publish on /detected_plate_point, store as self.plate_position, continue',
        '      e. Plate-radius filter: if 2D dist from plate_position < 0.14m → skip (object is on/near plate)',
        '      f. Keep track of best_pick_point by highest confidence score',
        '4.  After loop: publish best_pick_point on /detected_pick_point + /detected_class',
        '5.  If show_image=True: draw YOLO bounding boxes on frame, display via cv2.imshow()',
    ]
    doc += code_block(cb_steps)

    doc.append(H2('Plate-Radius Exclusion Filter'))
    doc.append(P('Once the plate centroid is known, any detection whose 2D distance from the plate (in base_link XY) is less than <b>self.plate_radius = 0.14 m</b> is silently dropped. This prevents the robot from trying to pick objects that are already on the plate. The threshold is configurable in code.'))
    doc.append(SP(4))

    doc.append(H2('Watchdog Timer'))
    doc.append(P('A 5-second repeating timer (<tt>_watchdog_cb</tt>) warns if no RGB frames or point cloud messages have been received. It cancels itself once both streams are flowing. This helps diagnose camera startup issues.'))
    doc.append(SP(4))

    doc.append(H2('Declare Parameters'))
    param_rows = [
        ['Parameter', 'Default', 'Description'],
        ['image_topic', '/camera/camera/color/image_raw', 'RGB image topic'],
        ['cloud_topic', '/camera/camera/depth/color/points', 'PointCloud2 topic'],
        ['camera_info_topic', '/camera/camera/color/camera_info', 'Camera intrinsics topic'],
        ['model_path', 'updated.pt', 'Path to YOLO weights file'],
        ['conf_threshold', '0.5', 'Minimum YOLO confidence to keep a detection'],
        ['show_image', 'True', 'Display annotated video window'],
        ['target_frame', 'base_link', 'TF2 target frame for 3D points'],
        ['target_class', '""', 'Empty = all classes; set to restrict picks to one class'],
    ]
    doc.append(topic_table(param_rows, [1.8*inch, 2.5*inch, 2.2*inch]))
    doc.append(SP(8))

    # 4.1 YOLODetector
    doc.append(H2('4.1  YOLODetector Helper Class'))
    doc.append(P('<b>File:</b> src/perception/perception/yolo_detector.py'))
    doc.append(P('A stateless wrapper around Ultralytics YOLO. It is instantiated once inside DetectionNode.__init__() and called on every frame.'))
    doc.append(SP(4))
    doc.append(H3('detect(image) → list[dict]'))
    doc.append(P('Runs YOLO inference on an OpenCV BGR image. Returns one dict per detection with fields:'))
    doc.append(B2('<tt>class_id</tt> — integer YOLO class index'))
    doc.append(B2('<tt>class_name</tt> — string label (e.g., "apple", "plate")'))
    doc.append(B2('<tt>confidence</tt> — float 0–1'))
    doc.append(B2('<tt>bbox</tt> — [x1, y1, x2, y2] pixel coordinates'))
    doc.append(B2('<tt>center</tt> — [cx, cy] bbox center in pixels'))
    doc.append(SP(4))
    doc.append(H3('draw_detections(image, detections) → np.ndarray'))
    doc.append(P('Draws green bounding boxes, red centroid dots, and confidence labels onto a copy of the image. Used by DetectionNode when show_image=True.'))
    doc.append(SP(4))

    # 4.2 pixel_to_world
    doc.append(H2('4.2  pixel_to_world Helpers'))
    doc.append(P('<b>File:</b> src/perception/perception/pixel_to_world.py'))
    doc.append(SP(4))
    doc.append(H3('get_centroid_from_cloud_bbox(cloud_msg, camera_info_msg, bbox) → PointStamped | None'))
    doc.append(P('Converts a YOLO bounding box into a 3D point in the camera frame by filtering the PointCloud2.'))
    ptwsteps = [
        'Steps:',
        '  1. Read camera intrinsics fx, fy, cx, cy from CameraInfo.K matrix',
        '  2. Parse PointCloud2 data as float32 array, shape (N, point_step//4)',
        '     Extract XYZ = first 3 floats per point',
        '  3. Drop NaN points and points with z <= 0',
        '  4. Project each valid 3D point to image coords:',
        '       u = fx * X/Z + cx',
        '       v = fy * Y/Z + cy',
        '  5. Keep only points (u, v) that fall inside the YOLO bbox',
        '  6. Return mean(X), mean(Y), mean(Z) of filtered points as PointStamped',
        '  7. Return None if no valid points project into the bbox',
    ]
    doc += code_block(ptwsteps)
    doc.append(H3('transform_point(tf_buffer, point_stamped, target_frame) → PointStamped'))
    doc.append(P('A thin wrapper: calls tf_buffer.lookup_transform(target_frame, source_frame, Time()) then do_transform_point(). Converts camera-frame points to base_link.'))
    doc.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 5: Static TF
    # ══════════════════════════════════════════════════════════
    doc.append(H1('5. Node Deep-Dive: ConstantTransformPublisher'))
    doc.append(P('<b>File:</b> src/planning/planning/static_tf_transform.py &nbsp;|&nbsp; <b>Node name:</b> constant_tf_publisher'))
    doc.append(P('Broadcasts a single static TF transform on startup: <b>wrist_3_link → camera_link</b>.'))
    doc.append(SP(4))
    doc.append(P('This transform encodes the physical camera mount geometry:'))
    doc.append(B('<b>Translation:</b> x = −0.025 m (slightly behind wrist), y = +0.130 m (to the side), z = 0.0 m'))
    doc.append(B('<b>Rotation:</b> encoded in the 3×3 matrix G[:3,:3] which reorders axes so the camera optical frame aligns correctly with the robot wrist frame'))
    doc.append(SP(4))
    doc.append(H2('Why This Matters'))
    doc.append(P('Without this transform, TF2 cannot convert PointCloud2 depth data from camera_optical_frame into base_link. The DetectionNode calls transform_point() which performs a TF2 lookup through the chain: camera_depth_optical_frame → camera_link → wrist_3_link → ... → base_link. The static broadcaster must start before any detection is attempted.'))
    doc.append(SP(4))
    doc.append(H2('Implementation Detail'))
    doc.append(P('The rotation matrix G[:3,:3] is converted to a quaternion using scipy Rotation.from_matrix().as_quat() and stored in a TransformStamped message published via StaticTransformBroadcaster. The transform is sent once in __init__() — there is no timer or callback.'))
    doc.append(SP(4))
    doc.append(Warn('The child_frame_id is camera_link (the ROOT of the RealSense TF sub-tree), NOT an optical frame. Publishing to an optical frame would create a parent conflict — two nodes would both claim to be the parent of that frame, breaking TF2.'))
    doc.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 6: IKPlanner
    # ══════════════════════════════════════════════════════════
    doc.append(H1('6. Node Deep-Dive: IKPlanner'))
    doc.append(P('<b>File:</b> src/planning/planning/ik.py &nbsp;|&nbsp; <b>Node name:</b> ik_planner'))
    doc.append(P('IKPlanner is a ROS2 node that provides two synchronous blocking calls used by both ArmCircler and UR7e_CubeGrasp. It communicates with MoveIt 2 via service calls.'))
    doc.append(SP(4))

    doc.append(H2('compute_ik(current_joint_state, x, y, z, qx=0, qy=1, qz=0, qw=0) → JointState | None'))
    doc.append(P('Requests MoveIt inverse kinematics for a Cartesian target. The default quaternion (qy=1, all others 0) represents a 180° rotation around Y — this is the gripper-pointing-straight-down orientation. It is the correct default for picking objects off a table.'))
    ik_steps = [
        'Internal steps:',
        '  1. Build PoseStamped in frame "base_link" with (x, y, z) + quaternion',
        '  2. Fill GetPositionIK.Request:',
        '       ik_link_name = "tool0"      (the gripper link)',
        '       pose_stamped = target pose',
        '       robot_state.joint_state = current_joint_state  (seed for IK solver)',
        '       avoid_collisions = True',
        '       timeout = Duration(sec=5)',
        '       group_name = "ur_manipulator"',
        '  3. Call /compute_ik service (synchronous — blocks until response)',
        '  4. If error_code == SUCCESS: return result.solution.joint_state',
        '     Otherwise: return None',
        '',
        'Critical: each IK call uses the PREVIOUS step\'s joint solution as the seed.',
        'This keeps the IK solver in a locally consistent joint-space region and',
        'prevents elbow flips between adjacent waypoints.',
    ]
    doc += code_block(ik_steps)

    doc.append(H2('plan_to_joints(target_joint_state, start_joint_state=None) → RobotTrajectory | None'))
    doc.append(P('Calls MoveIt motion planning (RRTConnect) to generate a collision-free trajectory from the current state to a target joint configuration.'))
    plan_steps = [
        'Internal steps:',
        '  1. Build GetMotionPlan.Request:',
        '       group_name = "ur_manipulator"',
        '       allowed_planning_time = 10.0 s',
        '       planner_id = "RRTConnectkConfigDefault"',
        '       start_state.joint_state = start_joint_state (if provided)',
        '  2. For each joint in target_joint_state: add JointConstraint',
        '       tolerance_above = tolerance_below = 0.005 rad',
        '       weight = 1.0',
        '  3. Call /plan_kinematic_path service (synchronous)',
        '  4. Return motion_plan_response.trajectory on SUCCESS, else None',
    ]
    doc += code_block(plan_steps)

    doc.append(H2('Startup Wait'))
    doc.append(P('In __init__(), IKPlanner blocks in a loop calling wait_for_service() on both /compute_ik and /plan_kinematic_path until they are available. This ensures MoveIt is fully started before any IK is attempted.'))
    doc.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 7: ArmCircler
    # ══════════════════════════════════════════════════════════
    doc.append(H1('7. Node Deep-Dive: ArmCircler'))
    doc.append(P('<b>File:</b> src/planning/planning/arm_circler.py &nbsp;|&nbsp; <b>Node name:</b> arm_circler'))
    doc.append(P('ArmCircler drives the arm in a 2-row orbital path around the plate to capture images from multiple viewpoints. It is enabled by default and disabled with skip_circler:=true.'))
    doc.append(SP(4))

    doc.append(H2('Subscriptions'))
    doc.append(B('<b>/detected_plate_point</b> → <tt>_plate_callback(msg)</tt> — triggers orbit construction (once only, guarded by _orbit_triggered flag)'))
    doc.append(B('<b>/joint_states</b> → <tt>_joint_state_callback(msg)</tt> — stores latest joint state for IK seeding'))
    doc.append(B('<b>/camera/camera/color/image_raw</b> → <tt>_photo_callback(msg)</tt> — buffers latest camera frame'))
    doc.append(SP(4))

    doc.append(H2('Publishers'))
    doc.append(B('<b>/orbit_done</b> (Bool, TRANSIENT_LOCAL) — published once when all waypoints complete'))
    doc.append(B('<b>/start_pick_place</b> (Bool) — also available as publisher (unused in current flow; commander handles this)'))
    doc.append(SP(4))

    doc.append(H2('_plate_callback() — Orbit Trigger'))
    doc.append(P('This is the main initialization callback. It runs exactly once (guarded by <tt>self._orbit_triggered</tt>). On first call:'))
    orbit_steps = [
        '  1. Set _orbit_triggered = True (prevents re-triggering)',
        '  2. Clean and recreate captured_images/ directory',
        '  3. Extract plate centroid (cx, cy, cz) from message',
        '  4. Build orbit geometry:',
        '       radius = 0.15 m    (horizontal distance from plate centroid)',
        '       height = 0.30 m    (row 1) / 0.20 m (row 2)',
        '       num_points = 20 per row  (40 total)',
        '       tilt_distance = 0.12 m  (row 1) / 0.17 m (row 2)',
        '',
        '  5. For each of 2 rows:',
        '       angles = linspace(-π/4, π + π/8, 20)',
        '       Row 2 angles are REVERSED so the arm sweeps back',
        '       For each angle θ:',
        '         tx = cx + radius * cos(θ)',
        '         ty = cy + radius * sin(θ)',
        '         tz = cz + height',
        '',
        '         Orientation: look-at rotation pointing back toward plate',
        '           rot_z = R.from_euler("z", θ - π/2)     # face inward',
        '           rot_y = R.from_euler("y", π)             # flip to point down',
        '           tilt_angle = arctan2(radius + tilt_distance, height)',
        '           rot_x = R.from_euler("x", tilt_angle)   # tilt toward plate',
        '           combined = rot_z * rot_y * rot_x',
        '           q = combined.as_quat()  # [x, y, z, w]',
        '',
        '         Call compute_ik(joint_state, tx, ty, tz, qx, qy, qz, qw)',
        '         If IK succeeds: append JointState to job_queue',
        '  6. After both rows: add a final "home-like" pose above the plate',
        '  7. Save pose_data as poses.npy  (shape: (41, 7) including plate centroid)',
        '  8. Call _execute_jobs() to start sequential execution',
    ]
    doc += code_block(orbit_steps)

    doc.append(H2('_execute_jobs() — Sequential Waypoint Execution'))
    doc.append(P('Pops one JointState from job_queue, calls plan_to_joints(), then calls _execute_joint_trajectory(). After each trajectory completes:'))
    doc.append(B('_on_exec_done() fires → calls <tt>_take_photo()</tt> to save the current frame to captured_images/'))
    doc.append(B('Then calls <tt>_execute_jobs()</tt> recursively for the next waypoint'))
    doc.append(B('When queue is empty: publishes Bool(True) on /orbit_done'))
    doc.append(SP(4))
    doc.append(Note('_take_photo() silently skips if self._frame is None (no camera frame buffered yet). Images are saved as captured_image_1.jpg through captured_image_N.jpg.'))
    doc.append(SP(4))

    doc.append(H2('poses.npy Format'))
    doc.append(P('The first entry in poses.npy is the plate centroid [cx, cy, cz, 0, 0, 0, 1]. Entries 1..N are the arm end-effector poses at each waypoint: [tx, ty, tz, qx, qy, qz, qw]. Pose i corresponds to captured_image_{i}.jpg (0-indexed in poses.npy, 1-indexed in filenames).'))
    doc.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 8: UR7e_CubeGrasp (main.py)
    # ══════════════════════════════════════════════════════════
    doc.append(H1('8. Node Deep-Dive: UR7e_CubeGrasp  (main.py)'))
    doc.append(P('<b>File:</b> src/planning/planning/main.py &nbsp;|&nbsp; <b>Node name:</b> cube_grasp'))
    doc.append(P('This is the central execution node. It owns the <b>busy</b> semaphore, the <b>job_queue</b>, and all state machine flags. It interfaces with IKPlanner, the trajectory controller, and the gripper service.'))
    doc.append(SP(4))

    doc.append(H2('8.1  Subscriptions and Publishers'))
    doc.append(H3('Subscriptions'))
    doc.append(B('<b>/detected_pick_point</b> (PointStamped) → <tt>cube_callback()</tt> — main pick trigger'))
    doc.append(B('<b>/detected_class</b> (String) → <tt>class_callback()</tt> — stores self.detected_class for offset lookup'))
    doc.append(B('<b>/joint_states</b> (JointState) → <tt>joint_state_callback()</tt> — stores latest joints; triggers initial _go_home() on first message'))
    doc.append(B('<b>/detected_plate_point</b> (PointStamped) → <tt>plate_callback()</tt> — stores self.plate_pose as default drop target'))
    doc.append(B('<b>/pick_task_list</b> (String/JSON) → <tt>_on_task_list()</tt> — loads pre-computed task list'))
    doc.append(B('<b>/start_pick_place</b> (Bool) → <tt>_on_start_pick_place()</tt> — activates pick-and-place mode'))
    doc.append(B('<b>/target_drop_point</b> (PointStamped) → <tt>_on_target_drop()</tt> — sets drop override + arms _pick_triggered'))
    doc.append(SP(4))
    doc.append(H3('Publishers'))
    doc.append(B('<b>/orbit_done</b> (Bool, TRANSIENT_LOCAL) — published in __init__() when skip_circler=True'))
    doc.append(SP(4))
    doc.append(H3('Action Clients'))
    doc.append(B('<b>/scaled_joint_trajectory_controller/follow_joint_trajectory</b> (FollowJointTrajectory) — executes planned trajectories on the robot'))
    doc.append(SP(4))
    doc.append(H3('Service Clients'))
    doc.append(B('<b>/toggle_gripper</b> (Trigger) — opens/closes gripper (toggle; state tracked by self.gripper_open)'))
    doc.append(SP(4))

    doc.append(H2('8.2  Job Queue Mechanism'))
    doc.append(P('The job queue (<tt>self.job_queue</tt>) is a flat list where each entry is one of:'))
    doc.append(B('<b>JointState</b> — a target joint configuration; calls plan_to_joints() then _execute_joint_trajectory()'))
    doc.append(B('<b>\'toggle_grip\'</b> (string literal) — calls _toggle_gripper() synchronously'))
    doc.append(SP(4))
    doc.append(P('<tt>execute_jobs()</tt> pops the front of the queue and dispatches the job. Each async completion callback calls execute_jobs() again. This creates a chain: the ROS event loop is never blocked — each job fires the next one via callback.'))
    doc.append(SP(4))
    jq_example = [
        'Example job queue for a pick-place cycle (after _on_refined_detection):',
        '',
        '  [0] JointState  → pre_grasp_joints   (hover above object)',
        '  [1] JointState  → grasp_joints        (descend to grasp height)',
        '  [2] "toggle_grip"                      (close gripper)',
        '  [3] JointState  → lift_joints          (lift object)',
        '  [4] JointState  → drop_pre_joints      (move above drop zone)',
        '  [5] JointState  → drop_joints          (descend to release height)',
        '  [6] "toggle_grip"                      (open gripper)',
        '',
        'After queue drains:  execute_jobs() detects empty queue, going_home=False',
        '  → calls _go_home()  → appends home JointState  → clears busy when home reached',
    ]
    doc += code_block(jq_example)

    doc.append(H2('8.3  Two-Phase Centroid Refinement'))
    doc.append(P('Detections at the camera\'s field-of-view edge are distorted. The two-phase system corrects this before committing to the full grasp trajectory:'))
    phase_steps = [
        'PHASE 1  (cube_callback — initial detection):',
        '  - Compute IK for (cx, cy, cz + 0.5)  -- 0.5 m directly above rough centroid',
        '  - Append only this one waypoint to job_queue',
        '  - Set _refining = True',
        '  - execute_jobs() runs, arm moves to pre-pregrasp',
        '',
        'execute_jobs() sees empty queue while _refining == True:',
        '  - Sets _at_pre_pregrasp = True',
        '  - Prints "Pre-pregrasp reached. Waiting for refined centroid..."',
        '  - Returns (waits for next detection)',
        '',
        'PHASE 2  (cube_callback — refined detection):',
        '  - cube_callback checks: _refining AND _at_pre_pregrasp → calls _on_refined_detection()',
        '  - Uses fresh, well-centered centroid with full grasp offsets',
        '  - Appends 7 jobs: pre_grasp, grasp, toggle, lift, drop_pre, drop, toggle',
        '  - execute_jobs() continues the sequence',
    ]
    doc += code_block(phase_steps)

    doc.append(H2('8.4  Task-List Auto-Execution'))
    doc.append(P('When a pre-computed task list is loaded via /pick_task_list, the system executes picks automatically without human input. The flow after /start_pick_place is received:'))
    tl_flow = [
        '_on_start_pick_place() called:',
        '  → sets pick_place_enabled = True',
        '  → calls _go_home()  (sets busy=True)',
        '',
        'Home reached → execute_jobs() sees empty queue, _going_home=True:',
        '  → _going_home = False, busy = False',
        '  → checks: self._task_list and self._task_idx < len(_task_list)',
        '  → calls _start_next_task()',
        '',
        '_start_next_task():',
        '  → pops task[_task_idx]  (object_name, pick:[x,y,z], drop:[x,y,z])',
        '  → sets self.detected_class = object_name',
        '  → sets self._current_drop_override = [drop_x, drop_y, drop_z]',
        '  → computes IK for (cx, cy, cz + 0.5)  (pre-pregrasp above 6D estimate)',
        '  → sets _refining = True, appends pre-pregrasp JointState',
        '  → execute_jobs() → arm moves to pre-pregrasp',
        '',
        'PHASE 2 fires when live detection sees the object below the arm:',
        '  → _on_refined_detection() uses _current_drop_override as drop position',
        '  → executes full pick-place sequence',
        '',
        'After drop, job queue drains → _go_home() → home reached:',
        '  → _start_next_task() for next item, or "All tasks complete."',
    ]
    doc += code_block(tl_flow)

    doc.append(H2('8.5  State Machine: busy / _refining / _going_home'))
    sm_table = [
        ['Flag', 'Type', 'True Means', 'Cleared By'],
        ['busy', 'bool', 'Arm is in motion or a pick cycle is active; cube_callback() returns immediately', 'execute_jobs() when home is reached; also cleared on trajectory failure'],
        ['_refining', 'bool', 'Arm is moving to or waiting at pre-pregrasp; subsequent cube_callback() detections are suppressed', 'Set False in _on_refined_detection() when Phase 2 fires'],
        ['_at_pre_pregrasp', 'bool', 'Arm has physically arrived at pre-pregrasp; next cube_callback() should call _on_refined_detection()', 'Set False in _on_refined_detection()'],
        ['_going_home', 'bool', 'Home waypoint is in the job queue (or arm is moving home)', 'Set False when execute_jobs() sees empty queue and _going_home=True'],
        ['pick_place_enabled', 'bool', 'System will respond to detections; False until /start_pick_place or skip_circler=true', 'Never reset; permanently True once activated'],
        ['_pick_triggered', 'bool', 'One live-detection pick has been armed by /target_drop_point', 'Cleared in cube_callback() when pick is accepted'],
    ]
    doc.append(topic_table(sm_table, [1.2*inch, 0.6*inch, 2.8*inch, 1.8*inch]))
    doc.append(SP(6))

    doc.append(H2('8.6  Pick Sequence — Full Call Chain'))
    doc.append(P('A complete pick cycle from first detection to home, showing every function call:'))
    pick_chain = [
        'image_callback()  [DetectionNode, ~30 Hz]',
        '  └─ detect() → transform_point() → publish /detected_class, /detected_pick_point',
        '',
        'cube_callback(pick_pose)  [UR7e_CubeGrasp]',
        '  ├─ check: pick_place_enabled, _pick_triggered, not busy, joint_state, plate_pose',
        '  ├─ _pick_triggered = False  (consume one-shot)',
        '  ├─ busy = True',
        '  ├─ compute_ik(joint_state, cx, cy, cz+0.5)',
        '  ├─ _refining = True',
        '  ├─ job_queue.append(pre_pre_grasp_joints)',
        '  └─ execute_jobs()',
        '       └─ plan_to_joints() → _execute_joint_trajectory()',
        '            └─ exec_ac.send_goal_async() → _on_goal_sent()',
        '                 └─ goal_handle.get_result_async() → _on_exec_done()',
        '                      └─ execute_jobs()  [queue now empty, _refining=True]',
        '                           └─ _at_pre_pregrasp = True  (WAIT)',
        '',
        '... arm physically at pre-pregrasp, next detection fires ...',
        '',
        'cube_callback(refined_pose)  [_refining AND _at_pre_pregrasp]',
        '  └─ _on_refined_detection(refined_pose)',
        '       ├─ _refining = False, _at_pre_pregrasp = False',
        '       ├─ drop from _current_drop_override (task-list) or plate_pose',
        '       ├─ lookup offsets in PICK_OFFSETS[detected_class]',
        '       ├─ compute_ik(joint_state, cx+x_off, cy+y_off, cz+pre_grasp_z)',
        '       ├─ compute_ik(pre_grasp, cx+x_off, cy+y_off, cz+grasp_z)',
        '       ├─ compute_ik(grasp, cx+x_off, cy+y_off, cz+lift_z)',
        '       ├─ compute_ik(lift, drop_x, drop_y, drop_z+0.20)',
        '       ├─ compute_ik(drop_pre, drop_x, drop_y, drop_z+0.15)',
        '       └─ job_queue = [pre_grasp, grasp, "toggle_grip",',
        '                        lift, drop_pre, drop, "toggle_grip"]',
        '       └─ execute_jobs()  → runs 7 jobs sequentially via callbacks',
        '',
        'After job 7: execute_jobs() → _going_home=False → _go_home()',
        '  └─ busy=True, _going_home=True, job_queue=[home_joints]',
        '  └─ execute_jobs() → plan → execute → _on_exec_done()',
        '       └─ execute_jobs(): queue empty, _going_home=True',
        '            → _going_home=False, busy=False',
        '            → if task list: _start_next_task()',
        '            → else: "Home reached. Ready for next pick command."',
    ]
    doc += code_block(pick_chain)
    doc.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 9: Commander
    # ══════════════════════════════════════════════════════════
    doc.append(H1('9. Node Deep-Dive: Commander'))
    doc.append(P('<b>File:</b> src/planning/planning/commander.py &nbsp;|&nbsp; <b>Node name:</b> commander'))
    doc.append(P('Commander is the high-level orchestrator. It runs in a separate terminal (stdin is required for user interaction). It receives the orbit-done signal, runs the two-stage pose pipeline as subprocesses, presents results to the user, and arms main.py for individual picks.'))
    doc.append(SP(4))

    doc.append(H2('Subscriptions'))
    doc.append(B('<b>/orbit_done</b> (Bool, TRANSIENT_LOCAL) → <tt>_on_orbit_done()</tt> — triggers pose pipeline in a background thread'))
    doc.append(B('<b>/detected_plate_point</b> (PointStamped) → <tt>plate_callback()</tt> — stores plate pose for drop position fallback'))
    doc.append(SP(4))

    doc.append(H2('Publishers'))
    doc.append(B('<b>/start_pick_place</b> (Bool) — published when user presses Enter'))
    doc.append(B('<b>/set_target_class</b> (String) — published with each pick command to switch YOLO target'))
    doc.append(B('<b>/target_drop_point</b> (PointStamped) — published with each pick command; receiving this also arms main.py\'s _pick_triggered'))
    doc.append(SP(4))

    doc.append(H2('_on_orbit_done() → _run_pose_pipeline() [background thread]'))
    doc.append(P('When /orbit_done fires, Commander spawns a daemon thread to run the full pose pipeline. This keeps the ROS spin loop responsive:'))
    pipeline_steps = [
        'Thread: _run_pose_pipeline()',
        '  cwd = os.getcwd()  (must be lab5/ for path resolution)',
        '',
        '  Step 1: YOLO + SAM2 segmentation subprocess',
        '    cmd = [SAM2_PYTHON, segment_batch.py,',
        '           --input_dir  captured_images/',
        '           --output_dir segmented/,',
        '           --yolo       updated.pt]',
        '    subprocess.run(cmd, check=True)',
        '    Output: segmented/captured_image_N/<stem>_<class>_<i>_mask.png',
        '',
        '  Step 2: 6D pose estimation subprocess',
        '    cmd = [SAM2_PYTHON, -c, "',
        '      import sys, os; sys.path.insert(0, SIXD_DIR); os.chdir(SIXD_DIR);',
        '      from run import run_pose_estimation;',
        '      run_pose_estimation(use_multi_view=True, position_only=True, ...)"]',
        '    Output: results/object_poses.npy',
        '',
        '  Step 3: Load and display results',
        '    _pose_results = np.load("results/object_poses.npy", allow_pickle=True)',
        '    _print_objects(): prints [idx] name  pos=(x,y,z)  conf=0.xx',
        '',
        '  Step 4: _prompt_user() — blocks on stdin',
    ]
    doc += code_block(pipeline_steps)

    doc.append(H2('Path Resolution'))
    doc.append(P('Commander uses <tt>_find_project_root()</tt> to locate the project by walking up the directory tree until it finds a parent containing an <b>armcircler/</b> subdirectory. This works whether running from source or from the installed package (which can be 7+ levels deep inside install/).'))
    doc.append(SP(4))
    path_lines = [
        '_PROJECT_ROOT = <parent containing armcircler/>',
        '_SIXD_DIR     = _PROJECT_ROOT/lab5/src/planning/planning/6D_poses/',
        '_SEG_SCRIPT   = _PROJECT_ROOT/lab5/segment_batch.py',
        '_YOLO_WEIGHTS = _PROJECT_ROOT/lab5/src/planning/planning/updated.pt',
        '_SAM2_PYTHON  = _PROJECT_ROOT/sam2/sam2_env/bin/python  (venv with sam2+torch)',
    ]
    doc += code_block(path_lines)

    doc.append(H2('_prompt_user() — Interactive CLI Loop'))
    prompt_steps = [
        '1. Blocks on input("Press Enter to activate pick-and-place: ")',
        '   → publishes Bool(True) on /start_pick_place',
        '   → prints detected objects table',
        '',
        '2. Enters while rclpy.ok() loop:',
        '   Prompts: "Type class name or index to pick (or list / q): "',
        '',
        '   Input is a digit:',
        '     → look up _pose_results[idx]',
        '     → _arm_for_pick(object_name, position_base_link_m)',
        '',
        '   Input is a string (class name):',
        '     → search _pose_results for matching object_name',
        '     → _arm_for_pick(user_in, drop_pos_from_results)',
        '',
        '   "list" → re-print detected objects',
        '   "q"    → break loop',
        '',
        '_arm_for_pick(class_name, drop_pos_arr):',
        '   1. Resolve drop position (pose results or plate centroid)',
        '   2. Publish String(class_name) on /set_target_class',
        '   3. Publish PointStamped(drop_x, drop_y, drop_z) on /target_drop_point',
        '      (receiving /target_drop_point in main.py sets _pick_triggered=True)',
    ]
    doc += code_block(prompt_steps)
    doc.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 10: 6D Pose Pipeline
    # ══════════════════════════════════════════════════════════
    doc.append(H1('10. 6D Pose Estimation Pipeline'))
    doc.append(P('The 6D pose pipeline runs entirely offline between the orbit phase and the pick phase. It is invoked as a subprocess by Commander and outputs <b>results/object_poses.npy</b>. All files live in <b>src/planning/planning/6D_poses/</b>.'))
    doc.append(SP(4))

    doc.append(H2('10.1  run.py — Entry Point'))
    doc.append(P('<tt>run_pose_estimation()</tt> is the top-level function called by Commander. It orchestrates the full pipeline:'))
    run_steps = [
        'run_pose_estimation(use_multi_view=True, position_only=True, ...)',
        '',
        '  1. Build PoseEstimator(configs=OBJECTS, position_only=True, ...)',
        '     → position_only=True skips DINOv2 + STL rendering (faster)',
        '',
        '  2. estimator.build_databases(force_rebuild=False)',
        '     → position_only: prints "skipping reference database build"',
        '',
        '  3. Load arm poses from poses.npy:',
        '     _load_all_poses(poses_path) → list of 4×4 ee_T_world matrices',
        '     MAX_IMAGES = 20  (YOLO reliable only on first 20 views)',
        '',
        '  4. For each of indices [0..19]:',
        '     Load captured_image_{i+1}.jpg → raw_images[i]',
        '     Build cam_T_world[i] = cam_T_world_from_ee(ee_T_world[i])',
        '       = inv(T_CAM_FROM_EE) @ ee_T_world',
        '',
        '  5. estimator.process_multi_view(',
        '       list(zip(raw_images, cam_poses)),',
        '       mask_dirs = [segmented/captured_image_{i+1} for i in indices]',
        '     ) → results list',
        '',
        '  6. _add_base_link_poses(results, cam_poses[0]):',
        '     → for each result: position_base_link_m = triangulated_position_m',
        '        (already in world/base_link frame from triangulation)',
        '',
        '  7. Save results/object_poses.npy:',
        '     [{object_name, instance_idx, position_base_link_m, confidence}, ...]',
    ]
    doc += code_block(run_steps)

    doc.append(H3('Pose Transform Chain'))
    doc.append(P('Converting arm EE poses into camera frame poses:'))
    transform_steps = [
        '_pose7_to_ee_T_world(p):',
        '  p = [x, y, z, qx, qy, qz, qw]  (arm EE pose in world/base_link frame)',
        '  world_T_ee = 4×4 matrix from quaternion + translation',
        '  ee_T_world = inv(world_T_ee)',
        '',
        'cam_T_world_from_ee(ee_T_world):',
        '  T_CAM_FROM_EE = fixed 4×4 offset (camera mount: +3cm forward, -13.5cm above)',
        '  cam_T_world = inv(T_CAM_FROM_EE) @ ee_T_world',
        '  This converts world coordinates into the camera frame at each orbit position',
    ]
    doc += code_block(transform_steps)

    doc.append(H2('10.2  pose_estimator.py — Five Sub-Components'))

    doc.append(H3('Component 1: ReferenceRenderer'))
    doc.append(P('Renders an object\'s STL mesh from N viewpoints using pyrender + trimesh. Used only in full 6DoF mode (position_only=False).'))
    doc.append(B('<b>load_mesh(cfg)</b> — loads STL, normalises to unit sphere, applies ObjectConfig.color_rgb as material'))
    doc.append(B('<b>render_views(cfg, n_views=500)</b> — renders from 500 Fibonacci-sphere viewpoints; returns list of {image, depth, pose_w2c, pose_c2w} dicts'))
    doc.append(B('<b>_fibonacci_sphere(n)</b> — generates n evenly-spaced unit vectors on a sphere using the golden-ratio spiral method'))
    doc.append(SP(4))

    doc.append(H3('Component 2: DinoFeatureExtractor'))
    doc.append(P('Wraps DINOv2 ViT-S/14 (from Facebook Research). Used only in full 6DoF mode.'))
    doc.append(B('<b>extract(image, mask=None)</b> — preprocesses to 224×224, normalises, runs DINOv2 forward pass, returns {cls: CLS token, patches: patch tokens}'))
    doc.append(B('Mask zeroes out background pixels before feature extraction, improving discriminability'))
    doc.append(SP(4))

    doc.append(H3('Component 3: ReferenceDatabase'))
    doc.append(P('Stores 500 rendered views + their DINOv2 CLS features for KNN retrieval. Used only in full 6DoF mode.'))
    doc.append(B('<b>build(views, extractor)</b> — extracts features for all views, builds KNN index (cosine similarity, brute-force)'))
    doc.append(B('<b>query(cls_vec, k=5)</b> — L2-normalises query, returns k nearest reference views with their w2c poses'))
    doc.append(B('<b>save/load(path)</b> — caches to .npz to avoid recomputing'))
    doc.append(SP(4))

    doc.append(H3('Component 4: PoseRefiner'))
    doc.append(P('Iterative render-and-compare edge-based refinement. Used only in full 6DoF mode.'))
    doc.append(B('<b>refine(coarse_pose_w2c, real_crop, n_iterations=6)</b> — starts from coarse DINO pose; iteratively perturbs and re-renders; keeps best by edge_score'))
    doc.append(B('<b>edge_score(render, real_crop)</b> — Chamfer-like distance on Canny edges: (render_edges * dist_transform(real_edges)).sum() + symmetric'))
    doc.append(B('<b>_perturb(pose, n=12)</b> — random rotation perturbations (skipped for symmetric objects) + random translation perturbations'))
    doc.append(B('<b>_resolve_symmetry(cfg)</b> — if cfg.is_symmetric is None, calls _detect_sphere() on the STL to auto-detect rotationally symmetric objects'))
    doc.append(SP(4))

    doc.append(H3('Component 5: PoseEstimator (top-level API)'))
    doc.append(P('The main entry point for pose estimation. Owns databases and refiners.'))
    doc.append(SP(4))

    doc.append(H4('process_multi_view() — used in current pipeline (position_only=True)'))
    mv_steps = [
        'process_multi_view(views, mask_dirs, seg_name_map, fx, fy, ...)',
        '  position_only=True → delegates to _multi_view_position_only()',
        '',
        '_multi_view_position_only(views, mask_dirs, seg_name_map, fx, fy, cx, cy):',
        '  For each view i:',
        '    all_masks = load_masks_from_dir(mask_dirs[i], seg_name_map)',
        '      → parses filenames: <stem>_<seg_label>_<inst_idx>_mask.png',
        '      → returns {config_name: [binary_mask0, mask1, ...]}',
        '    _scene_centroids_only(image, all_masks, fx, fy):',
        '      For each object mask:',
        '        u_c = median(x pixels in mask)  (robust to edge clipping)',
        '        v_c = median(y pixels in mask)',
        '        area = mask pixel count',
        '        depth estimation (if diameter_m known):',
        '          r_px = sqrt(area / π)           (effective radius in pixels)',
        '          z_m  = fx * (diameter_m/2) / r_px',
        '          x_m  = (u_c - cx) / fx * z_m',
        '          y_m  = (v_c - cy) / fy * z_m',
        '          position_m = [x_m, y_m, z_m]   (camera frame)',
        '        centroid_uv = (u_c, v_c)',
        '        mask_area = area',
        '      Returns list of {object_name, instance_idx, centroid_uv, mask_area, position_m}',
        '',
        '  Group observations by (object_name, instance_idx)',
        '',
        '  For each group:',
        '    triangulate_ransac(uvs, cam_poses, areas, fx, fy, cx, cy)',
        '    → triangulated_position_m  (in world/base_link frame)',
        '    confidence = min(1.0, n_views_seen / n_total_views)',
    ]
    doc += code_block(mv_steps)

    doc.append(H2('10.3  RANSAC + Weighted DLT Triangulation'))
    doc.append(P('The triangulation functions in pose_estimator.py recover the 3D world position of an object from 2D centroid observations across multiple camera views.'))
    doc.append(SP(4))

    doc.append(H3('triangulate_dlt(centroids_uv, poses_w2c, fx, fy, cx, cy, weights)'))
    dlt_steps = [
        'Direct Linear Transform (DLT) triangulation:',
        '',
        '  For each view (u, v) with camera matrix P = K @ T_w2c:',
        '    Add 2 rows to matrix A (weighted by sqrt(mask_area)):',
        '      (u*P[2] - P[0]) * w',
        '      (v*P[2] - P[1]) * w',
        '',
        '  Solve: SVD(A) → take last right singular vector V[-1]',
        '  Return X = V[-1][:3] / V[-1][3]  (homogeneous to 3D)',
        '',
        '  Weights = mask pixel counts → larger, clearer views have more influence',
    ]
    doc += code_block(dlt_steps)

    doc.append(H3('triangulate_ransac(centroids_uv, poses_w2c, mask_areas, fx, fy, cx, cy, n_iter=100)'))
    ransac_steps = [
        'RANSAC wrapper around DLT:',
        '  1. For n_iter iterations:',
        '       Pick random pair (i, j) of views',
        '       DLT triangulate from just those 2 views → X_candidate',
        '       For each view k: reproject X_candidate, measure pixel error',
        '       Count inliers (reprojection error < 8 px)',
        '       Keep best inlier set',
        '',
        '  2. If best_inliers < 2: use all views (fallback)',
        '',
        '  3. Re-run DLT on inlier views only, weighted by mask_area',
        '     → final triangulated_position_m  (world frame)',
        '',
        '  Returns (X_world, inlier_indices)',
        '',
        '  Effect: rejects views where the object is partially occluded,',
        '  at glancing angle, or where the mask centroid is unreliable.',
    ]
    doc += code_block(ransac_steps)

    doc.append(H2('10.4  constants.py — Configuration'))
    doc.append(P('Central configuration file for the 6D pose pipeline. Key entries:'))
    doc.append(SP(4))
    obj_rows = [
        ['Object Name', 'diameter_m', 'color_rgb', 'is_symmetric'],
        ['apple', '0.045 m', '(200, 40, 40)', 'auto (sphere)'],
        ['blueberry', '0.015 m', '(60, 40, 100)', 'auto (sphere)'],
        ['cherry', '0.020 m', '(180, 30, 30)', 'auto (sphere)'],
        ['cake', '0.075 m', '(80, 40, 15)', 'False (asymmetric)'],
        ['grape', '0.0236 m', '(100, 50, 120)', 'auto'],
        ['half_grape', '0.0236 m', '(100, 50, 120)', 'False'],
        ['halved_strawberry', '0.0301 m', '(210, 50, 60)', 'False'],
        ['mango_piece', '0.040 m', '(220, 150, 50)', 'False'],
        ['small_tomato', '0.030 m', '(210, 50, 40)', 'auto'],
        ['strawberry', '0.0301 m', '(210, 50, 60)', 'False'],
    ]
    doc.append(topic_table(obj_rows, [1.4*inch, 0.9*inch, 1.5*inch, 1.2*inch]))
    doc.append(SP(4))
    doc.append(P('<b>diameter_m</b> directly affects depth accuracy: z = fx × (diameter/2) / r_px. Wrong diameters lead to wrong z estimates and missed picks. Tune per-object against real measurements.'))
    doc.append(SP(4))
    seg_lines = [
        'SEG_NAME_MAP: maps YOLO/SAM2 label names in mask filenames → ObjectConfig names',
        '  "apple"      → "apple"',
        '  "blueberry"  → "blueberry"',
        '  "cake"       → "cake"',
        '  "grape"      → "grape"',
        '  "strawberry" → "strawberry"',
        '  "tomato"     → "small_tomato"',
        'Labels absent from this map are skipped by pose estimation.',
    ]
    doc += code_block(seg_lines)
    doc.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 11: END-TO-END PIPELINE
    # ══════════════════════════════════════════════════════════
    doc.append(H1('11. End-to-End Pipeline Call Chain'))
    doc.append(P('This section traces the complete system from startup to completing all picks, showing which node calls what.'))
    doc.append(SP(4))

    doc.append(H2('Stage 1: Startup'))
    startup = [
        'Terminal 1:',
        '  ros2 launch realsense2_camera rs_launch.py pointcloud.enable:=true ...',
        '  → Starts /camera/... topics',
        '',
        'Terminal 2:',
        '  ros2 launch planning lab5_bringup.launch.py robot_ip:=... target_class:=apple',
        '  → Starts: ConstantTransformPublisher (broadcasts wrist_3_link→camera_link TF)',
        '  → Starts: IKPlanner (waits for /compute_ik, /plan_kinematic_path)',
        '  → Starts: DetectionNode (starts YOLO inference loop)',
        '  → Starts: ArmCircler (subscribes to /detected_plate_point)',
        '  → Starts: UR7e_CubeGrasp',
        '       joint_state_callback() fires on first /joint_states message',
        '       → _go_home() called (arm moves to observation pose)',
        '',
        'Terminal 3:',
        '  cd lab5/',
        '  ros2 run planning commander',
        '  → Commander subscribes to /orbit_done (TRANSIENT_LOCAL — will get it late)',
        '  → Commander subscribes to /detected_plate_point',
        '  → Prints "Waiting for orbit to complete (/orbit_done)..."',
    ]
    doc += code_block(startup)

    doc.append(H2('Stage 2: Plate Detection Triggers Orbit'))
    stage2 = [
        'DetectionNode.image_callback():',
        '  → YOLO detects "plate" in frame',
        '  → get_centroid_from_cloud_bbox() → 3D centroid',
        '  → transform_point() → base_link frame',
        '  → publish /detected_plate_point (PointStamped)',
        '',
        'ArmCircler._plate_callback(msg):',
        '  → First call only (_orbit_triggered guard)',
        '  → Delete + recreate captured_images/ directory',
        '  → Build 40 waypoints (2 rows × 20 angles) around plate centroid',
        '  → For each waypoint: compute_ik() → append to job_queue',
        '  → Save poses.npy',
        '  → Call _execute_jobs()',
        '       Loop: plan_to_joints() → execute → photo → next',
        '  → When queue empty: publish Bool(True) on /orbit_done',
    ]
    doc += code_block(stage2)

    doc.append(H2('Stage 3: Offline Pose Pipeline (Commander)'))
    stage3 = [
        'Commander._on_orbit_done(msg):',
        '  → Spawn daemon thread: _run_pose_pipeline()',
        '',
        'Thread _run_pose_pipeline():',
        '  → subprocess: SAM2_PYTHON segment_batch.py',
        '       Reads:  captured_images/captured_image_{1..N}.jpg',
        '       Runs:   YOLO detection + SAM2 instance segmentation',
        '       Writes: segmented/captured_image_N/<stem>_<class>_<i>_mask.png',
        '',
        '  → subprocess: SAM2_PYTHON -c "from run import run_pose_estimation; ..."',
        '       Reads:  poses.npy, captured_images/, segmented/',
        '       Runs:   _load_all_poses() → cam_T_world per view',
        '               process_multi_view() → _multi_view_position_only()',
        '                 → per view: load_masks_from_dir()',
        '                            _scene_centroids_only() → centroid_uv, depth',
        '               triangulate_ransac() → position_base_link_m per object',
        '       Writes: results/object_poses.npy',
        '',
        '  → Load results, print table',
        '  → Block on: input("Press Enter to activate pick-and-place: ")',
    ]
    doc += code_block(stage3)

    doc.append(H2('Stage 4: User Activates + Pick Loop'))
    stage4 = [
        'User presses Enter:',
        '  Commander._prompt_user():',
        '  → publish Bool(True) on /start_pick_place',
        '',
        'UR7e_CubeGrasp._on_start_pick_place():',
        '  → pick_place_enabled = True',
        '  → _go_home()  [if task list: "N tasks queued. Going home first."]',
        '',
        'Home reached → execute_jobs() → _start_next_task()',
        '  OR',
        'User types "2" in Commander:',
        '  Commander._arm_for_pick("cake", pose_results[2]["position_base_link_m"])',
        '  → publish String("cake") on /set_target_class',
        '  → publish PointStamped(drop_x, drop_y, drop_z) on /target_drop_point',
        '',
        'DetectionNode._set_target_class_cb(): target_class = "cake"',
        '',
        'UR7e_CubeGrasp._on_target_drop(): _current_drop_override = [x,y,z], _pick_triggered = True',
        '',
        'Next DetectionNode frame with "cake" detected:',
        '  → publish /detected_class("cake"), /detected_pick_point(pt)',
        '',
        'UR7e_CubeGrasp.cube_callback(pt):',
        '  [Phase 1 → Phase 2 → pick-place cycle as described in Section 8.6]',
        '',
        'After pick: arm returns home, ready for next command.',
    ]
    doc += code_block(stage4)
    doc.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 12: PICK OFFSETS
    # ══════════════════════════════════════════════════════════
    doc.append(H1('12. Per-Class Grasp Offsets Reference'))
    doc.append(P('These offsets are applied to the <b>refined centroid</b> (Phase 2) in <tt>_on_refined_detection()</tt>. All values are in metres. The x/y offsets correct for systematic camera-to-gripper misalignment. The z offsets define the grasp trajectory heights.'))
    doc.append(SP(4))
    offset_rows = [
        ['Class', 'x_offset', 'y_offset', 'pre_grasp_z', 'grasp_z', 'lift_z'],
        ['apple', '0.010', '0.020', '0.160', '0.130', '0.185'],
        ['tomato', '0.015', '0.025', '0.160', '0.145', '0.185'],
        ['cake', '0.015', '0.025', '0.200', '0.140', '0.200'],
        ['strawberry', '0.020', '0.020', '0.200', '0.145', '0.200'],
        ['cherry', '0.020', '0.020', '0.200', '0.140', '0.200'],
        ['grape', '0.020', '0.020', '0.200', '0.144', '0.200'],
        ['blueberry', '0.020', '0.020', '0.200', '0.145', '0.200'],
        ['chocolate', '0.020', '0.005', '0.200', '0.143', '0.200'],
        ['DEFAULT', '0.020', '0.005', '0.160', '0.140', '0.185'],
    ]
    doc.append(topic_table(offset_rows, [1.0*inch, 0.8*inch, 0.8*inch, 0.9*inch, 0.8*inch, 0.8*inch]))
    doc.append(SP(6))
    offset_meaning = [
        'How these are applied (in _on_refined_detection):',
        '',
        '  cx, cy, cz = refined centroid in base_link',
        '',
        '  pre_grasp = (cx + x_offset,  cy + y_offset,  cz + pre_grasp_z)  ← hover',
        '  grasp     = (cx + x_offset,  cy + y_offset,  cz + grasp_z)      ← close gripper',
        '  lift      = (cx + x_offset,  cy + y_offset,  cz + lift_z)       ← lift',
        '  drop_pre  = (drop_x,  drop_y,  drop_z + 0.20)                   ← above plate',
        '  drop      = (drop_x,  drop_y,  drop_z + 0.15)                   ← release',
        '',
        '  Tuning guide:',
        '    grasp_z too high → gripper closes on air above object',
        '    grasp_z too low  → gripper hits table surface',
        '    x/y_offset wrong → gripper center misses object',
    ]
    doc += code_block(offset_meaning)
    doc.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 13: DESIGN PATTERNS
    # ══════════════════════════════════════════════════════════
    doc.append(H1('13. Key Design Patterns'))

    doc.append(H2('Pattern 1: busy Semaphore'))
    doc.append(P('The <tt>self.busy</tt> boolean in UR7e_CubeGrasp prevents concurrent pick cycles. Once a pick is accepted in cube_callback(), busy is set True and all subsequent cube_callback() calls return immediately. It is only cleared after the arm returns home (or on trajectory failure). This guarantees exactly one pick cycle executes at a time.'))

    doc.append(H2('Pattern 2: IK Chaining (Seed Propagation)'))
    doc.append(P('Each <tt>compute_ik()</tt> call in the pick sequence uses the <i>previous step\'s joint solution</i> as the seed, not the live /joint_states. This is critical: it keeps the IK solver exploring solutions in the same joint-space neighborhood, preventing the arm from making large, unexpected elbow flips between adjacent waypoints.'))
    doc.append(SP(4))
    chain_ex = [
        'seed = self.joint_state           (current arm position)',
        'pre_grasp   = compute_ik(seed, ...)',
        'grasp       = compute_ik(pre_grasp, ...)',
        'lift        = compute_ik(grasp, ...)',
        'drop_pre    = compute_ik(lift, ...)',
        'drop        = compute_ik(drop_pre, ...)',
    ]
    doc += code_block(chain_ex)

    doc.append(H2('Pattern 3: Async Trajectory Execution via Callbacks'))
    doc.append(P('Trajectories are executed without blocking the ROS event loop:'))
    async_chain = [
        'execute_jobs()',
        '  └─ _execute_joint_trajectory(joint_traj)',
        '       └─ exec_ac.send_goal_async(goal)',
        '            └─ callback: _on_goal_sent(future)',
        '                 └─ goal_handle.get_result_async()',
        '                      └─ callback: _on_exec_done(future)',
        '                           └─ execute_jobs()  [next job]',
        '',
        'The ROS spin loop is never stalled. The arm executes while ROS',
        'keeps processing subscriptions (so detection topics stay live).',
    ]
    doc += code_block(async_chain)

    doc.append(H2('Pattern 4: TRANSIENT_LOCAL QoS for One-Time Signals'))
    doc.append(P('/orbit_done is published once and must be received by Commander even if Commander starts late. TRANSIENT_LOCAL (latched) QoS ensures any subscriber that joins after the message is published will still receive the last value. Both publisher and subscriber must use TRANSIENT_LOCAL — if they mismatch, the subscriber gets nothing.'))

    doc.append(H2('Pattern 5: Two-Phase Centroid Refinement'))
    doc.append(P('The initial YOLO centroid may be distorted if the object is near the camera\'s field-of-view edge. By first moving directly above the rough estimate (pre-pregrasp), the camera gets a near-overhead, undistorted view. The refined centroid from this position is used for the actual grasp trajectory. This corrects systematic radial distortion without any calibration data.'))

    doc.append(H2('Pattern 6: Topic Ordering Race Condition'))
    doc.append(P('/detected_class MUST arrive before /detected_pick_point in UR7e_CubeGrasp. The class is used to select per-class offsets in PICK_OFFSETS. If /detected_pick_point arrives first, self.detected_class will be stale or empty and DEFAULT_OFFSETS will be used silently. DetectionNode always publishes class then pick_point in the same callback, so this is only a risk if network latency separates them.'))

    doc.append(H2('Pattern 7: Background Subprocess Management in Commander'))
    doc.append(P('The YOLO+SAM2 segmentation and 6D pose estimation are run as external Python subprocesses using the sam2 virtual environment\'s Python interpreter. This is because these tools require packages (torch, sam2, trimesh, pyrender) that are not available in the ROS Python environment. Commander runs them via subprocess.run() in a daemon thread to keep the ROS node responsive.'))

    doc.append(H2('Pattern 8: Graceful Failure Recovery'))
    doc.append(P('In _on_exec_done(), if the trajectory action returns an exception (e.g., controller rejected goal), the handler clears busy, job_queue, and all sentinel flags, then logs the error. This prevents the system from hanging indefinitely in a broken state. The next detection event can start a fresh pick cycle.'))
    doc.append(PageBreak())

    # ══════════════════════════════════════════════════════════
    # SECTION 14: KNOWN ISSUES
    # ══════════════════════════════════════════════════════════
    doc.append(H1('14. Known Issues and Watch-outs'))

    doc.append(H2('Camera Launch'))
    doc.append(Warn('The RealSense camera is commented out in lab5_bringup.launch.py. It must be started in a separate terminal before bringup. Forgetting this will cause DetectionNode to never receive data (watchdog timer will warn after 5s).'))
    doc.append(SP(4))

    doc.append(H2('IK Failure on Drop Waypoints'))
    doc.append(P('drop_pre_joints and drop_joints IK failures are not checked before enqueueing in _on_refined_detection(). If IK returns None for a drop waypoint, None will be passed to plan_to_joints(), which will attempt to iterate over None.name and raise an AttributeError. MoveIt IK failure on the drop zone usually means the plate is in an unreachable area — move the plate closer to the robot\'s workspace.'))
    doc.append(SP(4))

    doc.append(H2('Gripper State Assumption'))
    doc.append(P('self.gripper_open is initialised to True in UR7e_CubeGrasp. If the gripper is physically closed at startup, the first toggle_grip will open it instead of close. The fix is to manually open the gripper before launching.'))
    doc.append(SP(4))

    doc.append(H2('Object Moves After Pre-Pregrasp'))
    doc.append(P('After the arm reaches pre-pregrasp, the system waits indefinitely for a detection. If the object is removed or moves out of YOLO\'s field of view, the arm will remain at pre-pregrasp forever. busy stays True. A timeout mechanism could be added to execute_jobs() in the _at_pre_pregrasp wait state.'))
    doc.append(SP(4))

    doc.append(H2('Commander Must Run in Separate Terminal'))
    doc.append(P('Commander blocks on stdin (input() calls). stdin is not forwarded to nodes launched via ros2 launch. Commander must always be started manually with ros2 run planning commander in its own terminal.'))
    doc.append(SP(4))

    doc.append(H2('Source vs Install Sync'))
    doc.append(P('colcon build copies Python files into install/planning/lib/python3.10/site-packages/planning/. It does NOT create symlinks. Every change to src/planning/planning/*.py must be manually mirrored to the install tree until the next colcon build:'))
    sync_cmd = [
        'cp src/planning/planning/main.py      install/planning/lib/python3.10/site-packages/planning/',
        'cp src/planning/planning/commander.py install/planning/lib/python3.10/site-packages/planning/',
        'cp src/planning/planning/arm_circler.py install/planning/lib/python3.10/site-packages/planning/',
    ]
    doc += code_block(sync_cmd)

    doc.append(H2('6D Position Accuracy'))
    doc.append(P('The depth estimate z = fx × (diameter_m/2) / r_px is sensitive to the diameter_m values in constants.py. A 10% error in diameter produces a roughly 10% error in z, which directly affects grasp_z_offset accuracy. Measure actual object diameters and update constants.py for best results.'))
    doc.append(SP(4))

    doc.append(H2('busy=True During Go-Home After /start_pick_place'))
    doc.append(P('When the user presses Enter and pick-and-place activates, _go_home() is called immediately (sets busy=True). Any YOLO detection that fires during the home trajectory is ignored. Wait for "Home reached. Ready for next pick command." in the logs before expecting a pick to trigger.'))
    doc.append(SP(4))

    doc.append(H2('Task List Consumes Picks Automatically'))
    doc.append(P('If a task list was published before /start_pick_place, main.py will automatically execute all tasks after going home — without any further Commander input. Typing class names in Commander after this only affects the live-detection fallback (used only when the task list is exhausted or empty).'))
    doc.append(SP(20))
    doc.append(HR())
    doc.append(SP(8))
    doc.append(Paragraph('End of Document — MEC106A Lab 5 Code Guide', ST['subtitle']))

    return doc


# ─────────────────────────────────────────────────────────
# Build the PDF
# ─────────────────────────────────────────────────────────

def main():
    out_path = 'lab5_code_guide.pdf'
    doc = SimpleDocTemplate(
        out_path,
        pagesize=letter,
        leftMargin=1*inch,
        rightMargin=1*inch,
        topMargin=0.9*inch,
        bottomMargin=0.9*inch,
        title='MEC106A Lab 5 — Code Guide',
        author='Auto-generated',
    )

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#888888'))
        canvas.drawString(1*inch, 0.5*inch, 'MEC106A Lab 5 — Vision-Guided Pick and Place')
        canvas.drawRightString(7.5*inch, 0.5*inch, f'Page {doc.page}')
        canvas.restoreState()

    content = build_content()
    doc.build(content, onFirstPage=on_page, onLaterPages=on_page)
    print(f'PDF generated: {out_path}')


if __name__ == '__main__':
    main()
