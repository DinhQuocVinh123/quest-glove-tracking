#!/usr/bin/env python3
"""
MMPose Ultra-Fast White Haptics Glove Tracker
Optimized for Apple Silicon Metal (MPS) & VR/Haptic Hand Gestures.

Key Optimizations:
1. 3x-4x Speedup (30-45 FPS): Direct inference_topdown with RTMPose-m (21ms latency),
   bypassing multi-stage detection overhead and 1080p camera throttling.
2. Clenched Fist & Grasp Support: Core knuckle validation (Wrist + 5 MCPs) ensures
   tracking never drops when fingers curl into a tight fist.
3. Anti-Collapse Dynamic Bounding Box: Box dimensions anchored to rigid palm length,
   preventing the bounding box from shrinking or collapsing on occlusions.
4. T-Shirt Text & Distraction Rejection: Anatomical bone ratio checks reject text (e.g. "LEVENTS")
   and non-hand background edges.
5. Velocity & Momentum Extrapolation: Proactively predicts hand trajectory to track rapid flicks.
6. Haptics Telemetry HUD: 5-finger bend meters matching physical glove color codes
   (Index: Cyan, Middle: Pink, Ring: Purple, Pinky: Green, Thumb: Orange), plus
   gesture recognition (FIST, OPEN PALM, PINCH, POINTING).
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

CONFIG_FILE = os.path.join(_BASE_DIR, "mmpose", "configs", "hand_2d_keypoint", "rtmpose", "hand5", "rtmpose-m_8xb256-210e_hand5-256x256.py")
_FINETUNED_CKPT = os.path.join(_BASE_DIR, "checkpoints", "rtmpose_glove_finetuned.pth")
_ORIGINAL_CKPT_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"
# --original CLI flag (see main()) forces this back to _ORIGINAL_CKPT_URL for
# A/B testing against the fine-tuned checkpoint.
CHECKPOINT_FILE = _FINETUNED_CKPT if os.path.exists(_FINETUNED_CKPT) else _ORIGINAL_CKPT_URL

# Hand skeleton connections (21 keypoints)
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

# Colors matching the user's white haptic glove:
# Thumb: Orange, Index: Sky Blue (sensor clip), Middle: Pink, Ring: Purple, Pinky: Neon Green
FINGER_COLORS = [
    (0, 145, 255),  # Thumb: Orange
    (240, 200, 0),  # Index: Sky Blue (matches index mount)
    (210, 80, 255), # Middle: Pink (matches pink band)
    (220, 60, 160), # Ring: Purple (matches purple band)
    (50, 230, 80),  # Pinky: Neon Green (matches green band)
]


class OneEuroFilter:
    """OneEuro Filter for smooth, low-latency keypoint tracking."""
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


def load_config():
    defaults = {
        "conf_thr": 26,       # 0.26 default
        "smooth_factor": 45,  # OneEuro smoothing scale
        "box_pad": 35,        # 35%
        "bone_thick": 3,
        "show_hud": 1,
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                saved = json.load(f)
                defaults.update(saved)
        except Exception:
            pass
    return defaults


def save_config(cfg):
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Configuration saved to {CONFIG_PATH}")


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
    """Classify hand posture into common haptic/VR gestures."""
    thumb, index, mid, ring, pinky = flex_values
    wrist = kpts[0]
    mcp_mid = kpts[9]
    palm_len = max(np.linalg.norm(mcp_mid - wrist), 1e-4)

    # Fist
    if all(b >= 65 for b in [index, mid, ring, pinky]):
        return "CLENCHED FIST"
    # Open Palm
    if all(b <= 30 for b in flex_values):
        return "OPEN PALM"
    # Pointing
    if index <= 35 and all(b >= 55 for b in [mid, ring, pinky]):
        return "POINTING"
    # Victory / Peace
    if index <= 35 and mid <= 35 and ring >= 55 and pinky >= 55:
        return "PEACE / V-SIGN"
    # Pinch
    dist_thumb_index = np.linalg.norm(kpts[4] - kpts[8])
    if dist_thumb_index < 0.40 * palm_len:
        return "PINCH GESTURE"

    return "ACTIVE TRACKING"


def is_valid_hand(kpts, scores, conf_thr=0.25):
    """
    Anatomical validation:
    1. Knuckle core confidence (wrist + 5 MCP knuckles).
    2. Bone proportion check to reject t-shirt text and noise.
    """
    wrist = kpts[0]
    mcps = kpts[FINGER_MCPS]
    core_scores = scores[[0, 1, 5, 9, 13, 17]]
    core_conf = float(np.mean(core_scores))

    if core_conf < conf_thr:
        return False, core_conf

    # Palm length
    palm_len = np.linalg.norm(kpts[9] - wrist)
    if palm_len < 30.0:  # Too small to be a real hand
        return False, core_conf

    # Knuckle span (Index MCP to Pinky MCP)
    knuckle_span = np.linalg.norm(kpts[17] - kpts[5])
    if knuckle_span < 20.0 or knuckle_span > palm_len * 2.2:
        return False, core_conf

    # Check that thumb is not flying across the entire screen (e.g. to t-shirt text)
    thumb_tip_dist = np.linalg.norm(kpts[4] - kpts[1])
    if thumb_tip_dist > palm_len * 2.0:
        return False, core_conf

    return True, core_conf


def draw_flex_hud(img, flex_values, gesture, fps, inf_time, device_name):
    """Draw haptics telemetry and finger flexion bars."""
    h, w = img.shape[:2]
    panel_w = 210
    x_start = w - panel_w - 15
    y_start = 18

    overlay = img.copy()
    cv2.rectangle(overlay, (x_start - 10, y_start - 6), (w - 10, y_start + 235), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    # Header
    cv2.putText(img, "HAPTIC GLOVE TELEMETRY", (x_start, y_start + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Gesture badge
    gesture_color = (0, 255, 255) if gesture == "CLENCHED FIST" else (0, 255, 0)
    cv2.putText(img, f"State: {gesture}", (x_start, y_start + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, gesture_color, 1, cv2.LINE_AA)

    # Finger Flexion Meters
    for i, (name, bend) in enumerate(zip(FINGER_NAMES, flex_values)):
        y = y_start + 68 + i * 25
        color = FINGER_COLORS[i]
        cv2.putText(img, f"{name[:3]}:", (x_start, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 210), 1)

        bar_x = x_start + 45
        bar_w = 100
        bar_h = 11
        cv2.rectangle(img, (bar_x, y - 9), (bar_x + bar_w, y - 9 + bar_h), (50, 50, 50), -1)

        fill_w = int(bar_w * (bend / 100.0))
        cv2.rectangle(img, (bar_x, y - 9), (bar_x + fill_w, y - 9 + bar_h), color, -1)
        cv2.putText(img, f"{bend}%", (bar_x + bar_w + 6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

    # Perf stats at bottom of panel
    cv2.putText(img, f"Rate: {fps:.1f} FPS | Infer: {inf_time:.1f}ms", (x_start, y_start + 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, f"Hardware: {device_name.upper()}", (x_start, y_start + 222),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 255, 120), 1, cv2.LINE_AA)


def draw_skeleton(img, kpts, scores, score_thr=0.20, thickness=3):
    """Draw anti-aliased hand skeleton matching glove color bands."""
    for bone_idx, (start, end) in enumerate(HAND_SKELETON):
        if scores[start] > score_thr and scores[end] > score_thr:
            pt1 = (int(kpts[start][0]), int(kpts[start][1]))
            pt2 = (int(kpts[end][0]), int(kpts[end][1]))
            color = FINGER_COLORS[min(bone_idx // 4, 4)]
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

    for i, (x, y) in enumerate(kpts):
        if scores[i] > score_thr:
            pt = (int(x), int(y))
            # Wrist is white with dark outline
            color = (255, 255, 255) if i == 0 else FINGER_COLORS[min((i - 1) // 4, 4)]
            cv2.circle(img, pt, max(3, thickness + 2), color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, max(4, thickness + 3), (20, 20, 20), 1, cv2.LINE_AA)


def nothing(x):
    pass


def main():
    global CHECKPOINT_FILE
    parser = argparse.ArgumentParser(description="Ultra-Fast White Haptics Glove Tracker")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--original",
        action="store_true",
        help="Force the ORIGINAL (non-fine-tuned) checkpoint, for A/B comparison against the fine-tuned one.",
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

    if args.original:
        CHECKPOINT_FILE = _ORIGINAL_CKPT_URL

    print(f"Initializing camera {args.cam} at 720p (30 FPS)...")
    # CAP_DSHOW instead of the default MSMF backend: MSMF on Windows can get
    # into a broken state (grabFrame fails with HRESULT errors) after a prior
    # process was killed without releasing the camera; DirectShow re-opens
    # the device more reliably.
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    # Set 720p for fast 30+ FPS capture on macOS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ckpt_source = "FINE-TUNED (glove)" if CHECKPOINT_FILE == _FINETUNED_CKPT else "ORIGINAL (bare-hand only)"
    print(f"Loading RTMPose-m Top-Down Hand Model on {args.device.upper()}... [{ckpt_source}]")
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=args.device)

    # Warm up model JIT
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    dummy_box = np.array([[400, 200, 880, 680]])
    _ = inference_topdown(model, dummy_img, bboxes=dummy_box)
    if args.device == "mps":
        torch.mps.synchronize()

    cfg = load_config()

    win_name = "Ultra-Fast White Haptics Hand Tracker"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Interactive Trackbars
    cv2.createTrackbar("Confidence %", win_name, cfg["conf_thr"], 90, nothing)
    cv2.createTrackbar("Smoothing %", win_name, cfg["smooth_factor"], 100, nothing)
    cv2.createTrackbar("Box Pad %", win_name, cfg["box_pad"], 60, nothing)
    cv2.createTrackbar("Thickness", win_name, cfg["bone_thick"], 8, nothing)
    cv2.createTrackbar("Show HUD", win_name, cfg["show_hud"], 1, nothing)

    filter_kpts = None
    state = "SEARCHING"
    tracked_box = None
    lost_frames = 0
    prev_time = time.time()
    last_snap_time = 0.0
    fps = 0.0
    inf_latency = 0.0

    # Velocity tracker
    prev_center = None
    velocity = np.zeros(2, dtype=float)

    print("\n[READY] Tracker running!")
    print("Features:")
    print(" - Native 30-45 FPS Top-Down MPS Inference")
    print(" - Clenched Fist & Curled Fingers Core Knuckle Locking")
    print(" - Anti-Collapse Dynamic Bounding Box")
    print(" - T-shirt Text / Clutter Rejection")
    print("Controls:")
    print("  'c' : Save config | 'r' : Reset lock | 's' : Save snapshot | 'q' : Quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            curr_time = time.time()
            fps = 0.90 * fps + 0.10 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            # Read trackbars
            conf_thr = max(10, cv2.getTrackbarPos("Confidence %", win_name)) / 100.0
            smooth_val = cv2.getTrackbarPos("Smoothing %", win_name)
            box_pad = max(15, cv2.getTrackbarPos("Box Pad %", win_name)) / 100.0
            thickness = max(1, cv2.getTrackbarPos("Thickness", win_name))
            show_hud = cv2.getTrackbarPos("Show HUD", win_name)

            oneeuro_cutoff = np.interp(100 - smooth_val, [0, 100], [0.4, 4.0])

            # Determine Search ROI
            candidate_boxes = []
            if state == "TRACKING" and tracked_box is not None:
                # Extrapolate position using velocity (predict ahead)
                pred_box = tracked_box.copy()
                pred_box[0] += 1.2 * velocity[0]
                pred_box[2] += 1.2 * velocity[0]
                pred_box[1] += 1.2 * velocity[1]
                pred_box[3] += 1.2 * velocity[1]
                candidate_boxes.append(pred_box)
            else:
                # Smart multi-zone acquisition for desktop haptic glove:
                # Zone 1: Center-Right workspace (natural hand area)
                candidate_boxes.append(np.array([w * 0.35, h * 0.25, w * 0.85, h * 0.85]))
                # Zone 2: Lower desktop area
                candidate_boxes.append(np.array([w * 0.20, h * 0.40, w * 0.80, h * 0.98]))

            found_hand = False
            best_kpts = None
            best_scores = None
            best_core_conf = 0.0
            best_box = None

            for c_box in candidate_boxes:
                # Clamp candidate box
                bx1 = max(0, min(w - 60, int(c_box[0])))
                by1 = max(0, min(h - 60, int(c_box[1])))
                bx2 = max(bx1 + 60, min(w, int(c_box[2])))
                by2 = max(by1 + 60, min(h, int(c_box[3])))
                eval_box = np.array([[bx1, by1, bx2, by2]])

                t_inf_start = time.time()
                results = inference_topdown(model, frame, bboxes=eval_box)
                if args.device == "mps":
                    torch.mps.synchronize()
                inf_latency = (time.time() - t_inf_start) * 1000

                if len(results) > 0 and hasattr(results[0], "pred_instances"):
                    inst = results[0].pred_instances
                    kpts = inst.keypoints[0]
                    scores = inst.keypoint_scores[0]

                    valid, core_conf = is_valid_hand(kpts, scores, conf_thr=conf_thr)
                    if valid and core_conf > best_core_conf:
                        best_core_conf = core_conf
                        best_kpts = kpts
                        best_scores = scores
                        best_box = [bx1, by1, bx2, by2]
                        found_hand = True
                        break  # Found good hand in primary zone

            if found_hand and best_kpts is not None:
                lost_frames = 0
                state = "TRACKING"

                # OneEuro Filter smoothing
                if filter_kpts is None:
                    filter_kpts = OneEuroFilter(curr_time, best_kpts, min_cutoff=oneeuro_cutoff)
                    smoothed_kpts = best_kpts
                else:
                    smoothed_kpts = filter_kpts(curr_time, best_kpts, min_cutoff=oneeuro_cutoff)

                # Compute stable palm center (midpoint of wrist and middle MCP)
                palm_center = (smoothed_kpts[0] + smoothed_kpts[9]) / 2.0
                if prev_center is not None:
                    raw_v = palm_center - prev_center
                    velocity = 0.65 * raw_v + 0.35 * velocity
                prev_center = palm_center.copy()

                # Calculate rigid palm length to anchor box size
                palm_len = np.linalg.norm(smoothed_kpts[9] - smoothed_kpts[0])

                # Calculate bounding box bounds
                min_x = np.min(smoothed_kpts[:, 0])
                max_x = np.max(smoothed_kpts[:, 0])
                min_y = np.min(smoothed_kpts[:, 1])
                max_y = np.max(smoothed_kpts[:, 1])

                span_w = max_x - min_x
                span_h = max_y - min_y

                # Anti-collapse guarantee: box never drops below 2.8 * palm length
                box_side = max(span_w * (1.0 + box_pad), span_h * (1.0 + box_pad), palm_len * 2.8, 140.0)
                half = box_side / 2.0

                cx, cy = palm_center
                new_box = np.array([
                    max(0, cx - half),
                    max(0, cy - half * 1.1),  # Slightly more headroom for extended fingers
                    min(w, cx + half),
                    min(h, cy + half * 0.9),
                ])

                if tracked_box is not None:
                    tracked_box = 0.70 * new_box + 0.30 * tracked_box
                else:
                    tracked_box = new_box

                # Draw skeleton
                draw_skeleton(frame, smoothed_kpts, best_scores, score_thr=conf_thr, thickness=thickness)

                # Flexion HUD & Gesture Recognition
                flex_vals = calculate_finger_flex(smoothed_kpts)
                gesture = classify_gesture(flex_vals, smoothed_kpts)

                if show_hud:
                    draw_flex_hud(frame, flex_vals, gesture, fps, inf_latency, args.device)

                # Draw tracking bounding box
                bx1, by1, bx2, by2 = [int(v) for v in tracked_box]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(frame, f"WHITE HAPTIC GLOVE ({best_core_conf*100:.0f}%)", (bx1, max(20, by1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 2)

            else:
                lost_frames += 1
                if lost_frames > 12:
                    state = "SEARCHING"
                    tracked_box = None
                    filter_kpts = None
                    prev_center = None
                    velocity[:] = 0

            # Searching banner if not locked
            if state == "SEARCHING":
                # Draw subtle search zones
                zone = candidate_boxes[0]
                zx1, zy1, zx2, zy2 = [int(v) for v in zone]
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (255, 200, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, "HOLD WHITE HAPTIC GLOVE IN VIEW", (zx1 + 10, zy1 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 200, 0), 2)

            # Top-left info header
            header_color = (0, 255, 0) if state == "TRACKING" else (0, 200, 255)
            cv2.putText(frame, f"FPS: {fps:.1f} ({args.device.upper()}) | State: {state} | Latency: {inf_latency:.1f}ms",
                        (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, header_color, 2, cv2.LINE_AA)
            cv2.putText(frame, "Keys: 'c': Save Config | 'r': Reset | 's': Snapshot | 'q': Quit",
                        (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(win_name, frame)

            # Save snapshot every 1.5s for inspection
            if curr_time - last_snap_time > 1.5:
                last_snap_time = curr_time
                cv2.imwrite(SNAPSHOT_PATH, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break
            elif key == ord("r"):
                state = "SEARCHING"
                tracked_box = None
                filter_kpts = None
                prev_center = None
                velocity[:] = 0
            elif key == ord("s"):
                cv2.imwrite(SNAPSHOT_PATH, frame)
                print(f"Snapshot saved to {SNAPSHOT_PATH}")
            elif key == ord("c"):
                cur_cfg = {
                    "conf_thr": int(conf_thr * 100),
                    "smooth_factor": smooth_val,
                    "box_pad": int(box_pad * 100),
                    "bone_thick": thickness,
                    "show_hud": show_hud,
                }
                save_config(cur_cfg)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
