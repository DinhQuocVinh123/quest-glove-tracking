#!/usr/bin/env python3
"""
Dual-Hand MMPose Real-Time System:
- Hand 1 (User's Right / Bare Hand): GROUND TRUTH Reference Hand (Bare skin)
- Hand 2 (User's Left / Gloved Hand): INPUT Hand for Fine-Tuning (White Haptic Glove)

Features:
1. Native 30+ FPS Top-Down Batch Inference on Apple Silicon Metal (MPS).
2. Real-Time Pose Alignment & Error Telemetry:
   - Displays side-by-side flexion meters for Ground Truth vs Gloved Hand.
   - Real-time finger-by-finger delta error bars (Thumb, Index, Mid, Ring, Pinky).
3. Ground-Truth Data Collection for Fine-Tuning ('t' key):
   - Pairs bare hand ground-truth keypoints (mirrored) with glove crops.
   - Saves dataset to /Users/taminhtri/VR/dataset/ for immediate model training.
4. Gesture classification (Fist, Open Palm, Pinch, Pointing, Victory).
"""

import argparse
import json
import os
import sys
import time

# This script's own directory (D:\...\VR\) contains a sibling folder also
# named "mmpose" (the cloned repo). Python auto-prepends the script's
# directory to sys.path, which makes that sibling folder shadow the real,
# editable-installed "mmpose" package and breaks its internal path lookups.
# Remove it before importing mmpose so the real installed package is found.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

import cv2
import numpy as np
import torch
from mmpose.apis import inference_topdown, init_model

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_BASE_DIR, "glove_config.json")
SNAPSHOT_PATH = os.path.join(_BASE_DIR, "live_snapshot.jpg")
DATASET_DIR = os.path.join(_BASE_DIR, "dataset")
DATASET_IMG_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_FILE = os.path.join(DATASET_DIR, "annotations.json")

CONFIG_FILE = os.path.join(_BASE_DIR, "mmpose", "configs", "hand_2d_keypoint", "rtmpose", "hand5", "rtmpose-m_8xb256-210e_hand5-256x256.py")
CHECKPOINT_FILE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"

# 21 Hand Keypoints Connections
HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCPS = [1, 5, 9, 13, 17]

# Colors for Glove Hand (matches physical marker bands):
# Thumb: Orange, Index: Sky Blue, Middle: Pink, Ring: Purple, Pinky: Green
GLOVE_COLORS = [
    (0, 145, 255),  # Thumb: Orange
    (240, 200, 0),  # Index: Sky Blue (matches index mount)
    (210, 80, 255), # Middle: Pink (matches pink band)
    (220, 60, 160), # Ring: Purple (matches purple band)
    (50, 230, 80),  # Pinky: Neon Green (matches green band)
]

# Colors for Ground Truth Bare Hand (Emerald & Gold)
GT_COLORS = [
    (0, 215, 255),  # Gold
    (80, 220, 100), # Emerald
    (60, 200, 80),  # Green
    (40, 180, 60),  # Darker Green
    (20, 160, 40),  # Forest Green
]


class OneEuroFilter:
    """OneEuro Filter for smooth, low-latency tracking."""
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.008, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = np.array(x0, dtype=float)
        self.dx_prev = np.zeros_like(self.x_prev)
        self.t_prev = float(t0)

    def __call__(self, t, x, min_cutoff=None):
        if min_cutoff is not None:
            self.min_cutoff = float(min_cutoff)
        t_e = max(t - self.t_prev, 1e-4)
        a_d = self._alpha(t_e, self.d_cutoff)
        dx = (np.array(x, dtype=float) - self.x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(t_e, cutoff)
        x_hat = a * np.array(x, dtype=float) + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

    @staticmethod
    def _alpha(t_e, cutoff):
        r = 2 * np.pi * cutoff * t_e
        return r / (r + 1.0)


def calculate_finger_flex(kpts):
    """Calculate 0-100% flexion for each finger."""
    wrist = kpts[0]
    mcp_mid = kpts[9]
    palm_len = max(np.linalg.norm(mcp_mid - wrist), 1e-4)

    flex_values = []
    for tip_idx, mcp_idx in zip(FINGER_TIPS, FINGER_MCPS):
        dist_tip_wrist = np.linalg.norm(kpts[tip_idx] - wrist)
        ratio = dist_tip_wrist / palm_len
        bend = int(np.clip((1.90 - ratio) / 1.05 * 100, 0, 100))
        flex_values.append(bend)
    return flex_values


def classify_gesture(flex_values, kpts):
    """Classify gesture into common haptic postures."""
    thumb, index, mid, ring, pinky = flex_values
    wrist = kpts[0]
    mcp_mid = kpts[9]
    palm_len = max(np.linalg.norm(mcp_mid - wrist), 1e-4)

    if all(b >= 65 for b in [index, mid, ring, pinky]):
        return "CLENCHED FIST"
    if all(b <= 30 for b in flex_values):
        return "OPEN PALM"
    if index <= 35 and all(b >= 55 for b in [mid, ring, pinky]):
        return "POINTING"
    if index <= 35 and mid <= 35 and ring >= 55 and pinky >= 55:
        return "PEACE / V-SIGN"
    dist_thumb_index = np.linalg.norm(kpts[4] - kpts[8])
    if dist_thumb_index < 0.40 * palm_len:
        return "PINCH GESTURE"

    return "ACTIVE"


def draw_hand_skeleton(img, kpts, scores, colors, score_thr=0.20, thickness=3):
    """Draw anti-aliased hand skeleton."""
    for bone_idx, (start, end) in enumerate(HAND_SKELETON):
        if scores[start] > score_thr and scores[end] > score_thr:
            pt1 = (int(kpts[start][0]), int(kpts[start][1]))
            pt2 = (int(kpts[end][0]), int(kpts[end][1]))
            color = colors[min(bone_idx // 4, 4)]
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

    for i, (x, y) in enumerate(kpts):
        if scores[i] > score_thr:
            pt = (int(x), int(y))
            color = (255, 255, 255) if i == 0 else colors[min((i - 1) // 4, 4)]
            cv2.circle(img, pt, max(3, thickness + 2), color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, max(4, thickness + 3), (20, 20, 20), 1, cv2.LINE_AA)


def draw_dual_hud(img, gt_flex, glove_flex, gt_gesture, glove_gesture, fps, inf_time, rec_count, is_recording):
    """Draw side-by-side comparison HUD for Ground Truth vs Gloved Hand."""
    h, w = img.shape[:2]
    panel_w = 340
    panel_h = 240
    x_start = w - panel_w - 15
    y_start = 18

    overlay = img.copy()
    cv2.rectangle(overlay, (x_start - 10, y_start - 6), (w - 10, y_start + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)

    # Header
    cv2.putText(img, "DUAL-HAND CALIBRATION & FINETUNING", (x_start, y_start + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # Sub-headers: GT (Left) vs Glove (Right)
    cv2.putText(img, "GT BARE HAND", (x_start, y_start + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(img, "WHITE GLOVE", (x_start + 180, y_start + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1, cv2.LINE_AA)

    # State badges
    cv2.putText(img, f"{gt_gesture[:11]}", (x_start, y_start + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 255, 200), 1, cv2.LINE_AA)
    cv2.putText(img, f"{glove_gesture[:11]}", (x_start + 180, y_start + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 220, 100), 1, cv2.LINE_AA)

    # Rows for 5 fingers
    total_delta = 0
    for i, name in enumerate(FINGER_NAMES):
        y = y_start + 78 + i * 24
        g_val = gt_flex[i] if gt_flex is not None else 0
        v_val = glove_flex[i] if glove_flex is not None else 0
        delta = abs(g_val - v_val)
        total_delta += delta

        cv2.putText(img, f"{name[:3]}", (x_start, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        # GT Bar
        gt_bar_w = 45
        cv2.rectangle(img, (x_start + 35, y - 8), (x_start + 35 + gt_bar_w, y + 2), (50, 50, 50), -1)
        fill_gt = int(gt_bar_w * (g_val / 100.0))
        cv2.rectangle(img, (x_start + 35, y - 8), (x_start + 35 + fill_gt, y + 2), (80, 220, 100), -1)
        cv2.putText(img, f"{g_val}%", (x_start + 84, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        # Delta in Center
        delta_color = (80, 255, 120) if delta <= 15 else ((0, 220, 255) if delta <= 30 else (0, 80, 255))
        cv2.putText(img, f"d{delta}", (x_start + 122, y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, delta_color, 1)

        # Glove Bar
        glove_bar_w = 45
        cv2.rectangle(img, (x_start + 155, y - 8), (x_start + 155 + glove_bar_w, y + 2), (50, 50, 50), -1)
        fill_gl = int(glove_bar_w * (v_val / 100.0))
        cv2.rectangle(img, (x_start + 155, y - 8), (x_start + 155 + fill_gl, y + 2), GLOVE_COLORS[i], -1)
        cv2.putText(img, f"{v_val}%", (x_start + 204, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

    align_score = max(0, 100 - total_delta // 5)
    cv2.putText(img, f"Pose Alignment: {align_score}%", (x_start, y_start + 204),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    # Recording Status Indicator
    if is_recording:
        rec_color = (0, 0, 255) if (int(time.time() * 2) % 2 == 0) else (255, 255, 255)
        cv2.putText(img, f"[REC ACTIVE] Pairs: {rec_count}", (x_start, y_start + 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, rec_color, 2, cv2.LINE_AA)
    else:
        cv2.putText(img, f"Rec Idle ('t': Toggle) | Pairs: {rec_count}", (x_start, y_start + 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)


def init_dataset_storage():
    """Ensure dataset directories exist."""
    os.makedirs(DATASET_IMG_DIR, exist_ok=True)
    if not os.path.exists(ANNOTATIONS_FILE):
        with open(ANNOTATIONS_FILE, "w") as f:
            json.dump([], f)


def record_training_pair(frame, glove_box, glove_kpts, gt_kpts, count):
    """
    Save cropped glove frame and ground-truth paired keypoints for fine-tuning.
    """
    h, w = frame.shape[:2]
    bx1, by1, bx2, by2 = [int(v) for v in glove_box]
    bx1 = max(0, min(w - 20, bx1))
    by1 = max(0, min(h - 20, by1))
    bx2 = max(bx1 + 20, min(w, bx2))
    by2 = max(by1 + 20, min(h, by2))

    glove_crop = frame[by1:by2, bx1:bx2].copy()
    if glove_crop.shape[0] < 40 or glove_crop.shape[1] < 40:
        return count

    img_filename = f"glove_{count:05d}.jpg"
    img_filepath = os.path.join(DATASET_IMG_DIR, img_filename)
    cv2.imwrite(img_filepath, glove_crop)

    # Normalize GT keypoints to glove crop coordinate frame
    # Wrist centered, scaled by palm length
    gt_wrist = gt_kpts[0]
    gt_palm = max(np.linalg.norm(gt_kpts[9] - gt_wrist), 1e-4)

    gl_wrist = glove_kpts[0]
    gl_palm = max(np.linalg.norm(glove_kpts[9] - gl_wrist), 1e-4)

    # Mirror GT across vertical axis (since Left hand vs Right hand)
    mirrored_gt = gt_kpts.copy()
    mirrored_gt[:, 0] = gt_wrist[0] - (mirrored_gt[:, 0] - gt_wrist[0])

    # Transform normalized mirrored GT to glove box pixel space
    mapped_gt = gl_wrist + (mirrored_gt - gt_wrist) * (gl_palm / gt_palm)

    record = {
        "id": count,
        "image_file": img_filename,
        "crop_box": [bx1, by1, bx2, by2],
        "glove_kpts": glove_kpts.tolist(),
        "gt_kpts_raw": gt_kpts.tolist(),
        "gt_kpts_aligned": mapped_gt.tolist(),
        "timestamp": time.time(),
    }

    try:
        with open(ANNOTATIONS_FILE, "r") as f:
            data = json.load(f)
        data.append(record)
        with open(ANNOTATIONS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save annotation: {e}")

    return count + 1


def main():
    parser = argparse.ArgumentParser(description="Dual-Hand MMPose Ground-Truth & Fine-Tuning System")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    if torch.cuda.is_available():
        _default_device = "cuda"
    elif torch.backends.mps.is_available():
        _default_device = "mps"
    else:
        _default_device = "cpu"
    parser.add_argument(
        "--device",
        type=str,
        default=_default_device,
        help="Device: 'cuda', 'mps' or 'cpu'",
    )
    args = parser.parse_args()

    init_dataset_storage()

    # Load existing pairs count
    rec_count = 0
    if os.path.exists(ANNOTATIONS_FILE):
        try:
            with open(ANNOTATIONS_FILE, "r") as f:
                rec_count = len(json.load(f))
        except Exception:
            rec_count = 0

    print(f"Initializing camera at 720p (30 FPS)...")
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"Loading RTMPose-m on {args.device.upper()}...")
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=args.device)

    # Warmup MPS
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    dummy_boxes = np.array([
        [150, 200, 500, 650],
        [780, 200, 1130, 650]
    ])
    _ = inference_topdown(model, dummy_img, bboxes=dummy_boxes)
    if args.device == "mps":
        torch.mps.synchronize()

    win_name = "Dual-Hand Ground-Truth & Fine-Tuning Tracker"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Filters for both hands
    filter_gt = None
    filter_glove = None

    # Tracking ROIs
    box_gt = None
    box_glove = None

    is_recording = False
    prev_time = time.time()
    last_snap_time = 0.0
    fps = 0.0
    inf_latency = 0.0

    print("\n[READY] Dual-Hand Ground-Truth Tracker active!")
    print("Hand Roles:")
    print(" - Hand 1 (Left Screen): GROUND TRUTH (Bare Skin Hand)")
    print(" - Hand 2 (Right Screen): INPUT FOR FINETUNING (White Haptic Glove)")
    print("Controls:")
    print(f"  't' : Toggle recording fine-tuning pairs to {DATASET_DIR}")
    print("  'r' : Reset tracking locks")
    print("  's' : Save snapshot")
    print("  'q' : Quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            curr_time = time.time()
            fps = 0.90 * fps + 0.10 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            # Define default regions for 2 hands:
            # Bare Hand (GT): Left half of screen [0.08*w to 0.48*w]
            # Glove Hand: Right half of screen [0.52*w to 0.92*w]
            default_gt_box = np.array([w * 0.08, h * 0.20, w * 0.48, h * 0.85])
            default_glove_box = np.array([w * 0.52, h * 0.20, w * 0.92, h * 0.85])

            eval_gt = box_gt if box_gt is not None else default_gt_box
            eval_glove = box_glove if box_glove is not None else default_glove_box

            batch_boxes = np.array([eval_gt, eval_glove])

            t_start = time.time()
            results = inference_topdown(model, frame, bboxes=batch_boxes)
            if args.device == "mps":
                torch.mps.synchronize()
            inf_latency = (time.time() - t_start) * 1000

            gt_kpts, gt_scores = None, None
            glove_kpts, glove_scores = None, None

            # Process Hand 0: Ground Truth Bare Hand
            if len(results) > 0 and hasattr(results[0], "pred_instances"):
                inst0 = results[0].pred_instances
                k0 = inst0.keypoints[0]
                s0 = inst0.keypoint_scores[0]
                core_conf0 = np.mean(s0[[0, 1, 5, 9, 13, 17]])
                palm_len0 = np.linalg.norm(k0[9] - k0[0])

                if core_conf0 > 0.22 and palm_len0 > 30.0:
                    gt_kpts = k0
                    gt_scores = s0

                    if filter_gt is None:
                        filter_gt = OneEuroFilter(curr_time, gt_kpts)
                        smoothed_gt = gt_kpts
                    else:
                        smoothed_gt = filter_gt(curr_time, gt_kpts)

                    # Update GT Box
                    cx, cy = (smoothed_gt[0] + smoothed_gt[9]) / 2.0
                    box_dim = max(palm_len0 * 2.8, 140.0)
                    half = box_dim / 2.0
                    new_gt = np.array([max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)])
                    box_gt = 0.70 * new_gt + 0.30 * (box_gt if box_gt is not None else new_gt)

                    # Draw GT skeleton (Gold/Emerald)
                    draw_hand_skeleton(frame, smoothed_gt, gt_scores, GT_COLORS, score_thr=0.20, thickness=3)

                    # Draw GT Box
                    bx1, by1, bx2, by2 = [int(v) for v in box_gt]
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (80, 255, 120), 2)
                    cv2.putText(frame, f"GROUND TRUTH ({core_conf0*100:.0f}%)", (bx1, max(20, by1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 255, 120), 2)
                else:
                    box_gt = None
                    filter_gt = None

            # Process Hand 1: White Haptic Glove (Input to Fine-Tune)
            if len(results) > 1 and hasattr(results[1], "pred_instances"):
                inst1 = results[1].pred_instances
                k1 = inst1.keypoints[0]
                s1 = inst1.keypoint_scores[0]
                core_conf1 = np.mean(s1[[0, 1, 5, 9, 13, 17]])
                palm_len1 = np.linalg.norm(k1[9] - k1[0])

                if core_conf1 > 0.22 and palm_len1 > 30.0:
                    glove_kpts = k1
                    glove_scores = s1

                    if filter_glove is None:
                        filter_glove = OneEuroFilter(curr_time, glove_kpts)
                        smoothed_glove = glove_kpts
                    else:
                        smoothed_glove = filter_glove(curr_time, glove_kpts)

                    # Update Glove Box
                    cx, cy = (smoothed_glove[0] + smoothed_glove[9]) / 2.0
                    box_dim = max(palm_len1 * 2.8, 140.0)
                    half = box_dim / 2.0
                    new_gl = np.array([max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)])
                    box_glove = 0.70 * new_gl + 0.30 * (box_glove if box_glove is not None else new_gl)

                    # Draw Glove skeleton (Physical marker colors)
                    draw_hand_skeleton(frame, smoothed_glove, glove_scores, GLOVE_COLORS, score_thr=0.20, thickness=3)

                    # Draw Glove Box
                    bx1, by1, bx2, by2 = [int(v) for v in box_glove]
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 200, 255), 2)
                    cv2.putText(frame, f"INPUT GLOVE ({core_conf1*100:.0f}%)", (bx1, max(20, by1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 2)
                else:
                    box_glove = None
                    filter_glove = None

            # Calculate Flexion & Gestures
            gt_flex = calculate_finger_flex(smoothed_gt) if gt_kpts is not None else None
            gl_flex = calculate_finger_flex(smoothed_glove) if glove_kpts is not None else None

            gt_gesture = classify_gesture(gt_flex, smoothed_gt) if gt_flex is not None else "SEARCHING"
            gl_gesture = classify_gesture(gl_flex, smoothed_glove) if gl_flex is not None else "SEARCHING"

            # Draw Comparison HUD
            draw_dual_hud(frame, gt_flex, gl_flex, gt_gesture, gl_gesture, fps, inf_latency, rec_count, is_recording)

            # Record Pair if Recording Active and Both Hands Tracked
            if is_recording and gt_kpts is not None and glove_kpts is not None and box_glove is not None:
                rec_count = record_training_pair(frame, box_glove, smoothed_glove, smoothed_gt, rec_count)

            # Top-left info header
            cv2.putText(frame, f"FPS: {fps:.1f} ({args.device.upper()}) | 2-Hand Infer: {inf_latency:.1f}ms",
                        (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "'t': Toggle Recording Pairs | 'r': Reset | 's': Snapshot | 'q': Quit",
                        (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(win_name, frame)

            # Auto-save snapshot every 1.5s
            if curr_time - last_snap_time > 1.5:
                last_snap_time = curr_time
                cv2.imwrite(SNAPSHOT_PATH, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break
            elif key == ord("t"):
                is_recording = not is_recording
                print(f"[RECORDING] {'ENABLED' if is_recording else 'PAUSED'} (Pairs: {rec_count})")
            elif key == ord("r"):
                box_gt = None
                box_glove = None
                filter_gt = None
                filter_glove = None
            elif key == ord("s"):
                cv2.imwrite(SNAPSHOT_PATH, frame)
                print(f"Snapshot saved to {SNAPSHOT_PATH}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
