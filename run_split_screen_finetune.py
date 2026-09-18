#!/usr/bin/env python3
"""
MMPose Split-Screen Dual-Hand Tracker & Fine-Tuning Data Engine.

Layout:
- LEFT HALF: GROUND TRUTH (Bare Hand - High Fidelity Human Skin Pose)
- RIGHT HALF: INPUT TO TRAIN (White Haptic Glove - Tuned for White Fabric & Sensors)
- CENTER DIVIDER: Prevents torso / t-shirt text ("LEVENTS") cross-talk.

Features:
1. Complete separation of Ground Truth and Training Input across split screen.
2. Real-time Finger Flexion & Alignment Delta Telemetry.
3. One-Key Dataset Recording ('t' key) to capture paired ground-truth samples.
4. Color-coded skeletons matching physical haptic glove markers:
   - Index: Sky Blue (sensor clip)
   - Middle: Pink
   - Ring: Purple
   - Pinky: Neon Green
   - Thumb: Orange
5. Anti-collapse bounding boxes with OneEuro jitter filter.
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
from mmpose.apis.inferencers import MMPoseInferencer

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_BASE_DIR, "glove_config.json")
SNAPSHOT_PATH = os.path.join(_BASE_DIR, "live_snapshot.jpg")
DATASET_DIR = os.path.join(_BASE_DIR, "dataset")
DATASET_IMG_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_FILE = os.path.join(DATASET_DIR, "annotations.json")

# Hand Skeleton (21 keypoints)
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

# Ground Truth Colors (Emerald / Gold)
GT_COLORS = [
    (0, 215, 255),  # Thumb: Gold
    (80, 230, 100), # Index: Emerald
    (60, 210, 80),  # Middle
    (40, 190, 60),  # Ring
    (20, 170, 40),  # Pinky
]

# Glove Colors (Physical Marker Bands)
# Thumb: Orange, Index: Sky Blue, Middle: Pink, Ring: Purple, Pinky: Green
GLOVE_COLORS = [
    (0, 145, 255),  # Thumb: Orange
    (240, 200, 0),  # Index: Sky Blue (matches 3D sensor clip)
    (210, 80, 255), # Middle: Pink (matches pink band)
    (220, 60, 160), # Ring: Purple (matches purple band)
    (50, 230, 80),  # Pinky: Neon Green (matches green band)
]


class OneEuroFilter:
    """OneEuro Jitter Filter."""
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
    """Classify hand gesture."""
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
        return "PINCH"

    return "ACTIVE"


def draw_skeleton(img, kpts, scores, colors, score_thr=0.12, thickness=3):
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


def draw_side_hud(img, x_pos, y_pos, title, title_color, flex_values, gesture, colors):
    """Draw HUD panel for one hand side."""
    panel_w = 160
    overlay = img.copy()
    cv2.rectangle(overlay, (x_pos - 8, y_pos - 6), (x_pos + panel_w, y_pos + 195), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    cv2.putText(img, title, (x_pos, y_pos + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, title_color, 1, cv2.LINE_AA)
    cv2.putText(img, f"State: {gesture}", (x_pos, y_pos + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    if flex_values is not None:
        for i, (name, bend) in enumerate(zip(FINGER_NAMES, flex_values)):
            y = y_pos + 60 + i * 24
            cv2.putText(img, f"{name[:3]}:", (x_pos, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)

            bar_x = x_pos + 42
            bar_w = 68
            bar_h = 10
            cv2.rectangle(img, (bar_x, y - 8), (bar_x + bar_w, y - 8 + bar_h), (50, 50, 50), -1)
            fill_w = int(bar_w * (bend / 100.0))
            cv2.rectangle(img, (bar_x, y - 8), (bar_x + fill_w, y - 8 + bar_h), colors[i], -1)
            cv2.putText(img, f"{bend}%", (bar_x + bar_w + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)
    else:
        cv2.putText(img, "WAITING HAND...", (x_pos, y_pos + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 150, 150), 1)


def init_dataset_storage():
    os.makedirs(DATASET_IMG_DIR, exist_ok=True)
    if not os.path.exists(ANNOTATIONS_FILE):
        with open(ANNOTATIONS_FILE, "w") as f:
            json.dump([], f)


def align_gt_to_glove(gt_kpts, glove_kpts):
    """Map GT (bare-hand) keypoints into the glove hand's frame.

    This needs to correct for BOTH a size difference AND an orientation
    (rotation) difference between the two hands -- the bare hand and the
    gloved hand are almost never held at exactly the same tilt/angle, so
    a scale-only mapping leaves the whole label rotated off the real
    fingers. We solve this with a similarity transform (rotation + scale
    + translation) built from 2 anchor points per hand (wrist -> middle
    MCP), computed via complex-number division so rotation and scale are
    handled together in one step. Only the wrist and middle-MCP keypoints
    feed this transform -- these are the 2 most reliable landmarks even on
    the glove (large, high-contrast), unlike individual fingertips.
    """
    gt_wrist = gt_kpts[0]
    gl_wrist = glove_kpts[0]

    # Mirror GT across vertical axis for Left vs Right hand alignment first
    # (wrist, index 0, stays fixed under this mirror), THEN derive the
    # "forward" (wrist -> middle MCP) vector from the mirrored data so the
    # rotation/scale computed below matches what we're actually transforming.
    mirrored_gt = gt_kpts.copy()
    mirrored_gt[:, 0] = gt_wrist[0] - (mirrored_gt[:, 0] - gt_wrist[0])

    gt_forward = complex(*(mirrored_gt[9] - gt_wrist))
    gl_forward = complex(*(glove_kpts[9] - gl_wrist))
    if abs(gt_forward) < 1e-4:
        gt_forward = complex(1e-4, 0)
    similarity = gl_forward / gt_forward  # encodes rotation + scale together

    offsets = mirrored_gt - gt_wrist
    offsets_c = offsets[:, 0] + 1j * offsets[:, 1]
    transformed_c = offsets_c * similarity
    aligned_gt = np.stack(
        [gl_wrist[0] + transformed_c.real, gl_wrist[1] + transformed_c.imag],
        axis=1,
    )
    return aligned_gt


def save_paired_sample(frame, glove_box, glove_kpts, gt_kpts, count):
    """Save paired training sample for glove fine-tuning."""
    h, w = frame.shape[:2]

    # Compute the aligned GT label FIRST (before cropping), so the crop box
    # itself can be sized from it.
    aligned_gt = align_gt_to_glove(gt_kpts, glove_kpts)

    # Size the crop box from the ALIGNED (teacher-derived) keypoints, not
    # from the glove model's own raw 21-point guess. The glove model is
    # often wildly wrong on individual fingertips (esp. the 3 unmarked
    # fingers), which would otherwise stretch/misplace the saved crop even
    # though the label itself is now correct. aligned_gt reflects the TRUE
    # hand shape (mapped from the reliable bare-hand side), so using it to
    # frame the crop keeps the saved image properly centered on the hand.
    min_x = min(aligned_gt[:, 0].min(), glove_box[0])
    max_x = max(aligned_gt[:, 0].max(), glove_box[2])
    min_y = min(aligned_gt[:, 1].min(), glove_box[1])
    max_y = max(aligned_gt[:, 1].max(), glove_box[3])
    pad = int(max(max_x - min_x, max_y - min_y) * 0.15)

    bx1 = max(0, min(w - 20, int(min_x - pad)))
    by1 = max(0, min(h - 20, int(min_y - pad)))
    bx2 = max(bx1 + 20, min(w, int(max_x + pad)))
    by2 = max(by1 + 20, min(h, int(max_y + pad)))

    glove_crop = frame[by1:by2, bx1:bx2].copy()
    if glove_crop.shape[0] < 40 or glove_crop.shape[1] < 40:
        return count

    img_name = f"glove_{count:05d}.jpg"
    img_path = os.path.join(DATASET_IMG_DIR, img_name)
    cv2.imwrite(img_path, glove_crop)

    # Box-relative coordinates for training
    crop_kpts_glove = glove_kpts - np.array([bx1, by1])
    crop_kpts_gt = aligned_gt - np.array([bx1, by1])

    record = {
        "id": count,
        "image_file": img_name,
        "crop_box": [bx1, by1, bx2, by2],
        "glove_kpts_crop": crop_kpts_glove.tolist(),
        "gt_kpts_crop": crop_kpts_gt.tolist(),
        "timestamp": time.time(),
    }

    try:
        with open(ANNOTATIONS_FILE, "r") as f:
            data = json.load(f)
        data.append(record)
        with open(ANNOTATIONS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Saving annotation: {e}")

    return count + 1


def nothing(x):
    pass


def main():
    parser = argparse.ArgumentParser(description="Split-Screen Dual-Hand Ground Truth & Training Tracker")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--swap-sides",
        action="store_true",
        help="Swap which screen half is Ground Truth (bare hand) vs Glove input, "
             "without changing any detection/tracking logic.",
    )
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

    # Resume numbering from the highest glove_NNNNN.jpg index that already
    # exists on disk, NOT from len(records). Using len(records) collides
    # with existing image files once any records have been deleted (e.g.
    # via review_dataset.py), because a fresh session would then start
    # counting from a lower number than files that already exist, silently
    # overwriting them and leaving old annotations.json entries pointing at
    # a since-overwritten (mismatched) image.
    rec_count = 0
    if os.path.exists(DATASET_IMG_DIR):
        existing_indices = []
        for fname in os.listdir(DATASET_IMG_DIR):
            if fname.startswith("glove_") and fname.endswith(".jpg"):
                try:
                    existing_indices.append(int(fname[len("glove_"):-len(".jpg")]))
                except ValueError:
                    pass
        if existing_indices:
            rec_count = max(existing_indices) + 1

    # Separate from rec_count (the collision-free filename index): tracks
    # how many pairs are actually in the dataset right now, for the on-
    # screen HUD. These can differ once records have been deleted.
    total_pairs = 0
    if os.path.exists(ANNOTATIONS_FILE):
        try:
            with open(ANNOTATIONS_FILE, "r") as f:
                total_pairs = len(json.load(f))
        except Exception:
            total_pairs = 0

    print("Initializing camera at 720p (30 FPS)...")
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print(f"Loading RTMPose on {args.device.upper()}...")
    inferencer = MMPoseInferencer(pose2d="hand", device=args.device)

    win_name = "Split-Screen Dual-Hand Ground-Truth & Training"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Sliders
    cv2.createTrackbar("Glove Sens %", win_name, 12, 40, nothing)
    cv2.createTrackbar("GT Sens %", win_name, 20, 50, nothing)

    filter_gt = None
    filter_glove = None

    box_gt = None
    box_glove = None

    is_recording = False
    prev_time = time.time()
    last_snap_time = 0.0
    fps = 0.0

    print("\n[READY] Split-Screen Dual-Hand Tracker running!")
    print("Screen Division:")
    print("  LEFT HALF  : GROUND TRUTH (Bare Hand)")
    print("  RIGHT HALF : INPUT TO TRAIN (White Haptic Glove)")
    print("Controls:")
    print("  't' : Toggle paired training data recording")
    print("  'r' : Reset tracking locks")
    print("  's' : Save snapshot")
    print("  'q' : Quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if not args.swap_sides:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            mid_x = w // 2

            curr_time = time.time()
            fps = 0.90 * fps + 0.10 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            # Read sliders
            glove_sens = max(8, cv2.getTrackbarPos("Glove Sens %", win_name)) / 100.0
            gt_sens = max(10, cv2.getTrackbarPos("GT Sens %", win_name)) / 100.0

            # Center exclusion margin to avoid torso / t-shirt text ("LEVENTS")
            margin = 55
            gt_default = [40, 100, mid_x - margin, h - 40]
            glove_default = [mid_x + margin, 100, w - 40, h - 40]

            # 1. GROUND TRUTH (Left Half - Bare Hand)
            roi_gt = box_gt if box_gt is not None else gt_default
            gx1, gy1, gx2, gy2 = [int(v) for v in roi_gt]
            gx1 = max(0, min(mid_x - 40, gx1))
            gy1 = max(0, min(h - 40, gy1))
            gx2 = max(gx1 + 40, min(mid_x - 10, gx2))
            gy2 = max(gy1 + 40, min(h, gy2))

            crop_gt = frame[gy1:gy2, gx1:gx2]
            gt_found = False
            gt_kpts = None
            gt_scores = None

            if crop_gt.shape[0] > 50 and crop_gt.shape[1] > 50:
                for res in inferencer(crop_gt, return_vis=False):
                    preds = res.get("predictions", [])
                    if preds and len(preds[0]) > 0:
                        pred = preds[0][0]
                        kpts = np.array(pred["keypoints"])
                        scores = np.array(pred["keypoint_scores"])
                        core_conf = np.mean(scores[[0, 1, 5, 9, 13, 17]])
                        palm_len = np.linalg.norm(kpts[9] - kpts[0])

                        if core_conf >= gt_sens and palm_len > 20.0:
                            gt_found = True
                            global_k = kpts.copy()
                            global_k[:, 0] += gx1
                            global_k[:, 1] += gy1

                            if filter_gt is None:
                                filter_gt = OneEuroFilter(curr_time, global_k)
                                smoothed_gt = global_k
                            else:
                                smoothed_gt = filter_gt(curr_time, global_k)

                            gt_kpts = smoothed_gt
                            gt_scores = scores

                            # Dynamic box
                            min_x = np.min(smoothed_gt[:, 0])
                            max_x = np.max(smoothed_gt[:, 0])
                            min_y = np.min(smoothed_gt[:, 1])
                            max_y = np.max(smoothed_gt[:, 1])
                            pad = int(max(max_x - min_x, max_y - min_y) * 0.35)
                            box_gt = [
                                max(10, min_x - pad),
                                max(10, min_y - pad),
                                min(mid_x - 15, max_x + pad),
                                min(h - 10, max_y + pad),
                            ]

            if not gt_found:
                box_gt = None
                filter_gt = None

            # 2. INPUT GLOVE (Right Half - White Haptic Glove)
            # If not tracked, try candidate zones on the right side
            glove_candidates = []
            if box_glove is not None:
                glove_candidates.append(box_glove)
            else:
                # Upper workspace & Middle workspace in right half
                glove_candidates.append([mid_x + 80, 140, w - 60, 520])
                glove_candidates.append([mid_x + 60, 240, w - 60, 680])

            gl_found = False
            gl_kpts = None
            gl_scores = None

            for c_roi in glove_candidates:
                vx1, vy1, vx2, vy2 = [int(v) for v in c_roi]
                vx1 = max(mid_x + 10, min(w - 40, vx1))
                vy1 = max(0, min(h - 40, vy1))
                vx2 = max(vx1 + 40, min(w, vx2))
                vy2 = max(vy1 + 40, min(h, vy2))

                crop_gl = frame[vy1:vy2, vx1:vx2]
                if crop_gl.shape[0] < 50 or crop_gl.shape[1] < 50:
                    continue

                for res in inferencer(crop_gl, return_vis=False):
                    preds = res.get("predictions", [])
                    if preds and len(preds[0]) > 0:
                        pred = preds[0][0]
                        kpts = np.array(pred["keypoints"])
                        scores = np.array(pred["keypoint_scores"])
                        core_conf = np.mean(scores[[0, 1, 5, 9, 13, 17]])
                        palm_len = np.linalg.norm(kpts[9] - kpts[0])

                        # Glove sensitivity check
                        if core_conf >= glove_sens and palm_len > 15.0:
                            gl_found = True
                            global_k = kpts.copy()
                            global_k[:, 0] += vx1
                            global_k[:, 1] += vy1

                            if filter_glove is None:
                                filter_glove = OneEuroFilter(curr_time, global_k)
                                smoothed_gl = global_k
                            else:
                                smoothed_gl = filter_glove(curr_time, global_k)

                            gl_kpts = smoothed_gl
                            gl_scores = scores

                            # Dynamic box with anti-collapse
                            min_x = np.min(smoothed_gl[:, 0])
                            max_x = np.max(smoothed_gl[:, 0])
                            min_y = np.min(smoothed_gl[:, 1])
                            max_y = np.max(smoothed_gl[:, 1])
                            pad = int(max(max_x - min_x, max_y - min_y) * 0.40)
                            box_glove = [
                                max(mid_x + 15, min_x - pad),
                                max(10, min_y - pad),
                                min(w - 10, max_x + pad),
                                min(h - 10, max_y + pad),
                            ]
                            break
                if gl_found:
                    break

            if not gl_found:
                box_glove = None
                filter_glove = None

            # 3. DRAW VISUAL DIVIDER & LABELS
            cv2.line(frame, (mid_x, 0), (mid_x, h), (100, 100, 100), 2)
            cv2.line(frame, (mid_x, 0), (mid_x, h), (0, 220, 255), 1)

            cv2.putText(frame, "GROUND TRUTH: BARE HAND", (40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 255, 120), 2, cv2.LINE_AA)
            cv2.putText(frame, "INPUT: WHITE HAPTIC GLOVE", (mid_x + 40, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2, cv2.LINE_AA)

            # Draw Ground Truth
            if gt_kpts is not None:
                draw_skeleton(frame, gt_kpts, gt_scores, GT_COLORS, score_thr=gt_sens, thickness=3)
                bx1, by1, bx2, by2 = [int(v) for v in box_gt]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (80, 255, 120), 2)
                cv2.putText(frame, f"GT LOCKED ({np.mean(gt_scores)*100:.0f}%)", (bx1, by1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 255, 120), 2)
            else:
                cv2.rectangle(frame, (gt_default[0], gt_default[1]), (gt_default[2], gt_default[3]), (50, 80, 50), 1, cv2.LINE_AA)
                cv2.putText(frame, "HOLD BARE HAND HERE", (gt_default[0] + 20, gt_default[1] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 100), 1)

            # Draw Glove
            if gl_kpts is not None:
                display_gl_kpts = gl_kpts
                display_gl_scores = gl_scores
                if gt_kpts is not None:
                    # Middle/ring/pinky have no distinct visual marker on the
                    # glove (plain white fabric), so the model's own guess
                    # for them is unreliable even when thumb/index track
                    # fine. For live display only (not detection, box
                    # tracking, or the confidence gate), borrow their
                    # position from the bare hand instead, mapped into the
                    # glove's frame -- so what you see on screen mirrors the
                    # already-accurate bare-hand pose for those 3 fingers.
                    aligned_gt_live = align_gt_to_glove(gt_kpts, gl_kpts)
                    display_gl_kpts = gl_kpts.copy()
                    display_gl_kpts[9:21] = aligned_gt_live[9:21]
                    display_gl_scores = gl_scores.copy()
                    display_gl_scores[9:21] = 1.0
                draw_skeleton(frame, display_gl_kpts, display_gl_scores, GLOVE_COLORS, score_thr=glove_sens, thickness=3)
                bx1, by1, bx2, by2 = [int(v) for v in box_glove]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 200, 255), 2)
                cv2.putText(frame, f"GLOVE LOCKED ({np.mean(gl_scores)*100:.0f}%)", (bx1, by1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 2)
            else:
                cv2.rectangle(frame, (glove_default[0], glove_default[1]), (glove_default[2], glove_default[3]), (60, 60, 100), 1, cv2.LINE_AA)
                cv2.putText(frame, "HOLD WHITE GLOVE HERE", (glove_default[0] + 20, glove_default[1] + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 180, 255), 1)

            # Flexion & Gestures
            gt_flex = calculate_finger_flex(gt_kpts) if gt_kpts is not None else None
            gl_flex = calculate_finger_flex(gl_kpts) if gl_kpts is not None else None

            gt_gest = classify_gesture(gt_flex, gt_kpts) if gt_flex is not None else "WAITING"
            gl_gest = classify_gesture(gl_flex, gl_kpts) if gl_flex is not None else "WAITING"

            draw_side_hud(frame, 20, 65, "GT BARE HAND", (80, 255, 120), gt_flex, gt_gest, GT_COLORS)
            draw_side_hud(frame, w - 180, 65, "WHITE GLOVE", (0, 200, 255), gl_flex, gl_gest, GLOVE_COLORS)

            # Real-Time Alignment Telemetry & Dataset Pair Saving
            if gt_flex is not None and gl_flex is not None:
                deltas = [abs(g - v) for g, v in zip(gt_flex, gl_flex)]
                match_pct = max(0, 100 - sum(deltas) // 5)
                match_col = (80, 255, 120) if match_pct >= 80 else ((0, 220, 255) if match_pct >= 65 else (0, 100, 255))
                cv2.putText(frame, f"ALIGNMENT: {match_pct}%", (mid_x - 85, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, match_col, 2, cv2.LINE_AA)

                # Save paired sample if recording
                if is_recording and box_glove is not None:
                    prev_rec_count = rec_count
                    rec_count = save_paired_sample(frame, box_glove, gl_kpts, gt_kpts, rec_count)
                    if rec_count != prev_rec_count:
                        total_pairs += 1

            # Recording Status Banner
            if is_recording:
                rec_pulse = (0, 0, 255) if (int(time.time() * 2) % 2 == 0) else (255, 255, 255)
                cv2.putText(frame, f"[REC ACTIVE] Pairs: {total_pairs}", (mid_x - 110, h - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, rec_pulse, 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, f"Rec Paused ('t' to toggle) | Pairs: {total_pairs}", (mid_x - 150, h - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

            cv2.putText(frame, f"FPS: {fps:.1f} ({args.device.upper()})", (mid_x - 60, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(win_name, frame)

            if curr_time - last_snap_time > 1.5:
                last_snap_time = curr_time
                cv2.imwrite(SNAPSHOT_PATH, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break
            elif key == ord("t"):
                is_recording = not is_recording
                print(f"[RECORDING] {'ACTIVE' if is_recording else 'PAUSED'} (Pairs: {total_pairs})")
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
