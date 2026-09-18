#!/usr/bin/env python3
"""
MMPose Real-Time Tuned Glove Tracker with Interactive Sliders & OneEuro Filter.

Features:
1. False-Positive Filtering: Rejects small textures (t-shirt text, background objects)
   by enforcing anatomical hand size & span requirements.
2. Live Interactive Sliders (Trackbars):
   - Confidence % (Default 35%)
   - Smoothing % (OneEuro Jitter Filter)
   - Box Padding %
   - Skeleton Thickness
   - Show HUD Toggle
3. OneEuro Jitter Filter: Eliminates hand tremor & jitter with 0 latency.
4. Real-time Finger Flexion HUD: Displays live 0-100% bend meters for all 5 fingers.
5. Persistent settings saving ('c' key saves to glove_config.json).

Usage:
    python run_tuned_glove.py
"""

import argparse
import json
import os
import sys
import time
import cv2
import numpy as np
import torch
from mmpose.apis.inferencers import MMPoseInferencer

CONFIG_PATH = "/Users/taminhtri/VR/glove_config.json"

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

FINGER_COLORS = [
    (0, 140, 255),  # Thumb - Orange
    (255, 0, 255),  # Index - Magenta
    (255, 255, 0),  # Middle - Cyan
    (0, 0, 255),    # Ring - Red
    (0, 255, 0),    # Pinky - Green
]


class OneEuroFilter:
    """OneEuro Filter for smooth, low-latency keypoint tracking."""
    def __init__(self, t0, x0, min_cutoff=1.5, beta=0.007, d_cutoff=1.0):
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
        "conf_thr": 32,       # 0.32 default
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
        bend = int(np.clip((1.85 - ratio) / 1.05 * 100, 0, 100))
        flex_values.append(bend)
    return flex_values


def draw_flex_hud(img, flex_values):
    """Draw real-time finger flexion meters on the right side of the screen."""
    h, w = img.shape[:2]
    panel_w = 175
    x_start = w - panel_w - 15
    y_start = 20

    overlay = img.copy()
    cv2.rectangle(overlay, (x_start - 10, y_start - 5), (w - 10, y_start + 180), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)

    cv2.putText(img, "FINGER BEND", (x_start, y_start + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    for i, (name, bend) in enumerate(zip(FINGER_NAMES, flex_values)):
        y = y_start + 45 + i * 26
        color = FINGER_COLORS[i]
        cv2.putText(img, f"{name[:3]}:", (x_start, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        bar_x = x_start + 45
        bar_w = 80
        bar_h = 12
        cv2.rectangle(img, (bar_x, y - 10), (bar_x + bar_w, y - 10 + bar_h), (60, 60, 60), -1)

        fill_w = int(bar_w * (bend / 100.0))
        cv2.rectangle(img, (bar_x, y - 10), (bar_x + fill_w, y - 10 + bar_h), color, -1)
        cv2.putText(img, f"{bend}%", (bar_x + bar_w + 5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1)


def draw_skeleton(img, kpts, scores, score_thr=0.15, thickness=3):
    """Draw anti-aliased hand skeleton."""
    for bone_idx, (start, end) in enumerate(HAND_SKELETON):
        if scores[start] > score_thr and scores[end] > score_thr:
            pt1 = (int(kpts[start][0]), int(kpts[start][1]))
            pt2 = (int(kpts[end][0]), int(kpts[end][1]))
            color = FINGER_COLORS[min(bone_idx // 4, 4)]
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

    for i, (x, y) in enumerate(kpts):
        if scores[i] > score_thr:
            pt = (int(x), int(y))
            color = (255, 255, 255) if i == 0 else FINGER_COLORS[min((i - 1) // 4, 4)]
            cv2.circle(img, pt, max(3, thickness + 2), color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, max(4, thickness + 3), (0, 0, 0), 1, cv2.LINE_AA)


def nothing(x):
    pass


def main():
    parser = argparse.ArgumentParser(description="Tuned Real-Time Glove Tracker")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device: 'mps' or 'cpu'",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(1)

    print("Loading RTMPose on Apple Silicon MPS...")
    inferencer = MMPoseInferencer(pose2d="hand", device=args.device)

    cfg = load_config()

    win_name = "Tuned Glove Tracker (Live Tuning)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Live Trackbars
    cv2.createTrackbar("Confidence %", win_name, cfg["conf_thr"], 90, nothing)
    cv2.createTrackbar("Smoothing %", win_name, cfg["smooth_factor"], 100, nothing)
    cv2.createTrackbar("Box Pad %", win_name, cfg["box_pad"], 60, nothing)
    cv2.createTrackbar("Thickness", win_name, cfg["bone_thick"], 8, nothing)
    cv2.createTrackbar("Show HUD", win_name, cfg["show_hud"], 1, nothing)

    filter_kpts = None
    state = "LOCKING"
    tracked_box = None
    lost_frames = 0
    prev_time = time.time()
    last_snap_time = 0.0
    fps = 0.0

    print("\nTuned tracker ready!")
    print("Controls:")
    print("  'c' : Save current slider settings to config file")
    print("  'r' : Reset tracking lock")
    print("  's' : Save snapshot image")
    print("  'q' : Quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            # Read live trackbar values
            conf_thr = max(10, cv2.getTrackbarPos("Confidence %", win_name)) / 100.0
            smooth_val = cv2.getTrackbarPos("Smoothing %", win_name)
            box_pad = max(15, cv2.getTrackbarPos("Box Pad %", win_name)) / 100.0
            thickness = max(1, cv2.getTrackbarPos("Thickness", win_name))
            show_hud = cv2.getTrackbarPos("Show HUD", win_name)

            # Map slider (0-100) to OneEuro min_cutoff (0.3 - 5.0)
            oneeuro_cutoff = np.interp(100 - smooth_val, [0, 100], [0.3, 4.5])

            # Guide box (Center-Right where hands normally appear)
            box_w = int(min(h, w) * 0.48)
            box_h = box_w
            default_x1 = (w - box_w) // 2
            default_y1 = (h - box_h) // 2
            default_box = [default_x1, default_y1, default_x1 + box_w, default_y1 + box_h]

            current_roi = default_box if (state == "LOCKING" or tracked_box is None) else tracked_box
            rx1, ry1, rx2, ry2 = [int(v) for v in current_roi]
            rx1 = max(0, min(w - 20, rx1))
            ry1 = max(0, min(h - 20, ry1))
            rx2 = max(rx1 + 20, min(w, rx2))
            ry2 = max(ry1 + 20, min(h, ry2))

            crop = frame[ry1:ry2, rx1:rx2]
            found_hand = False

            if crop.shape[0] > 50 and crop.shape[1] > 50:
                for res in inferencer(crop, return_vis=False):
                    preds_list = res.get("predictions", [])
                    if preds_list and len(preds_list[0]) > 0:
                        pred = preds_list[0][0]
                        kpts = np.array(pred["keypoints"])
                        scores = np.array(pred["keypoint_scores"])

                        mean_score = np.mean(scores)

                        # Anatomical Hand Validation
                        # 1. Wrist to middle finger span
                        wrist_to_mid = np.linalg.norm(kpts[12] - kpts[0])
                        # 2. Keypoints span bounding box
                        span_w = np.max(kpts[:, 0]) - np.min(kpts[:, 0])
                        span_h = np.max(kpts[:, 1]) - np.min(kpts[:, 1])

                        # Must look like an actual hand, not tiny text or noise
                        if mean_score >= conf_thr and wrist_to_mid > 40 and (span_w > 50 or span_h > 50):
                            found_hand = True
                            lost_frames = 0
                            state = "TRACKING"

                            # Global coordinates
                            global_kpts = kpts.copy()
                            global_kpts[:, 0] += rx1
                            global_kpts[:, 1] += ry1

                            # OneEuro Filter smoothing
                            if filter_kpts is None:
                                filter_kpts = OneEuroFilter(curr_time, global_kpts, min_cutoff=oneeuro_cutoff)
                                smoothed_kpts = global_kpts
                            else:
                                smoothed_kpts = filter_kpts(curr_time, global_kpts, min_cutoff=oneeuro_cutoff)

                            # Draw skeleton
                            draw_skeleton(frame, smoothed_kpts, scores, score_thr=conf_thr, thickness=thickness)

                            # Draw flexion meters
                            if show_hud:
                                flex_vals = calculate_finger_flex(smoothed_kpts)
                                draw_flex_hud(frame, flex_vals)

                            # Dynamic Bounding Box update
                            min_x = np.min(smoothed_kpts[:, 0])
                            max_x = np.max(smoothed_kpts[:, 0])
                            min_y = np.min(smoothed_kpts[:, 1])
                            max_y = np.max(smoothed_kpts[:, 1])

                            pad = int(max(max_x - min_x, max_y - min_y) * box_pad)
                            new_box = [
                                max(0, min_x - pad),
                                max(0, min_y - pad),
                                min(w, max_x + pad),
                                min(h, max_y + pad),
                            ]

                            if tracked_box is not None:
                                alpha = 0.50
                                tracked_box = [
                                    alpha * nb + (1 - alpha) * ob
                                    for nb, ob in zip(new_box, tracked_box)
                                ]
                            else:
                                tracked_box = new_box

            if not found_hand:
                lost_frames += 1
                if lost_frames > 8:
                    state = "LOCKING"
                    tracked_box = None
                    filter_kpts = None

            # Render overlay box
            if state == "LOCKING":
                cv2.rectangle(frame, (default_box[0], default_box[1]), (default_box[2], default_box[3]), (0, 255, 255), 2)
                cv2.putText(frame, "HOLD GLOVE INSIDE BOX TO LOCK", (default_box[0], default_box[1] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            else:
                bx1, by1, bx2, by2 = [int(v) for v in tracked_box]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(frame, f"LOCKED ({mean_score*100:.0f}%)", (bx1, by1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # Top-left status
            cv2.putText(frame, f"FPS: {fps:.1f} ({args.device.upper()}) | State: {state}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Adjust sliders above | 'c': Save Config | 'r': Reset | 'q': Quit",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1, cv2.LINE_AA)

            cv2.imshow(win_name, frame)

            # Auto-save snapshot every 1.5s
            if curr_time - last_snap_time > 1.5:
                last_snap_time = curr_time
                cv2.imwrite("/Users/taminhtri/VR/live_snapshot.jpg", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break
            elif key == ord("r"):
                state = "LOCKING"
                tracked_box = None
                filter_kpts = None
            elif key == ord("s"):
                cv2.imwrite("/Users/taminhtri/VR/live_snapshot.jpg", frame)
                print("Snapshot saved.")
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
