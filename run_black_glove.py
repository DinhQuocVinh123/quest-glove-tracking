#!/usr/bin/env python3
"""
MMPose Black Cotton Glove Tracker with Continuous Keypoint Follower.

Features:
1. Pose-Guided Hand Following: Locks onto your black cotton glove and dynamically
   follows your hand across the screen as you move!
2. Natural Color Rendering: Fixed BGR/RGB color channels.
3. Live FPS, Skeleton Visualization, and Auto-Snapshots.

Usage:
    python run_black_glove.py
"""

import argparse
import sys
import time
import cv2
import numpy as np
import torch
from mmpose.apis.inferencers import MMPoseInferencer

# Hand skeleton connections (21 keypoints)
HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # Index
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
    (0, 13), (13, 14), (14, 15), (15, 16),# Ring
    (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
]

# Color palette for 5 fingers (BGR format)
FINGER_COLORS = [
    (0, 140, 255),  # Thumb - Orange
    (255, 0, 255),  # Index - Magenta
    (255, 255, 0),  # Middle - Cyan
    (0, 0, 255),    # Ring - Red
    (0, 255, 0),    # Pinky - Green
]


def parse_args():
    parser = argparse.ArgumentParser(description="Black Cotton Glove Tracker")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Device: 'mps' or 'cpu'",
    )
    return parser.parse_args()


def draw_hand_skeleton(img, kpts, scores, score_thr=0.15):
    """Draw colorful 21-keypoint skeleton on image."""
    h, w = img.shape[:2]

    # Draw bones
    for bone_idx, (start, end) in enumerate(HAND_SKELETON):
        if scores[start] > score_thr and scores[end] > score_thr:
            pt1 = (int(kpts[start][0]), int(kpts[start][1]))
            pt2 = (int(kpts[end][0]), int(kpts[end][1]))
            finger_id = min(bone_idx // 4, 4)
            color = FINGER_COLORS[finger_id]
            cv2.line(img, pt1, pt2, color, 3, cv2.LINE_AA)

    # Draw joints
    for i, (x, y) in enumerate(kpts):
        if scores[i] > score_thr:
            pt = (int(x), int(y))
            color = (255, 255, 255) if i == 0 else FINGER_COLORS[min((i - 1) // 4, 4)]
            cv2.circle(img, pt, 5, color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, 6, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    print("==================================================")
    print("   MMPose Black Cotton Glove Tracker (macOS)     ")
    print("==================================================")
    print(f"Device: {args.device.upper()}")
    print("Controls:")
    print("  'r' : Reset tracking lock to center guide box")
    print("  's' : Save snapshot to disk")
    print("  'q' : Quit")
    print("==================================================")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        sys.exit(1)

    print("\nLoading RTMPose model on Apple Silicon MPS...")
    inferencer = MMPoseInferencer(pose2d="hand", device=args.device)

    cv2.namedWindow("Black Cotton Glove Tracker", cv2.WINDOW_NORMAL)

    # Tracking state
    state = "LOCKING"  # "LOCKING" or "TRACKING"
    tracked_box = None  # [x1, y1, x2, y2]
    lost_frames = 0
    prev_time = time.time()
    last_snap_time = 0.0
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror webcam
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            # Define default guide box (center of frame)
            box_w = int(min(h, w) * 0.50)
            box_h = box_w
            default_x1 = (w - box_w) // 2
            default_y1 = (h - box_h) // 2
            default_box = [default_x1, default_y1, default_x1 + box_w, default_y1 + box_h]

            if state == "LOCKING" or tracked_box is None:
                current_roi = default_box
            else:
                current_roi = tracked_box

            rx1, ry1, rx2, ry2 = [int(v) for v in current_roi]
            rx1 = max(0, min(w - 10, rx1))
            ry1 = max(0, min(h - 10, ry1))
            rx2 = max(rx1 + 10, min(w, rx2))
            ry2 = max(ry1 + 10, min(h, ry2))

            crop = frame[ry1:ry2, rx1:rx2]

            # Run inference on cropped region
            found_hand = False
            if crop.shape[0] > 30 and crop.shape[1] > 30:
                for res in inferencer(crop, return_vis=False):
                    preds_list = res.get("predictions", [])
                    if preds_list and len(preds_list[0]) > 0:
                        pred = preds_list[0][0]
                        kpts = np.array(pred["keypoints"])  # (21, 2)
                        scores = np.array(pred["keypoint_scores"])  # (21,)

                        mean_score = np.mean(scores)
                        if mean_score > 0.20:
                            found_hand = True
                            lost_frames = 0
                            state = "TRACKING"

                            # Transform keypoints back to full frame coordinates
                            global_kpts = kpts.copy()
                            global_kpts[:, 0] += rx1
                            global_kpts[:, 1] += ry1

                            # Draw skeleton on main frame
                            draw_hand_skeleton(frame, global_kpts, scores, score_thr=0.15)

                            # Calculate new bounding box with 25% padding to follow hand
                            min_x = np.min(global_kpts[:, 0])
                            max_x = np.max(global_kpts[:, 0])
                            min_y = np.min(global_kpts[:, 1])
                            max_y = np.max(global_kpts[:, 1])

                            box_width = max_x - min_x
                            box_height = max_y - min_y
                            pad = int(max(box_width, box_height) * 0.35)

                            new_box = [
                                max(0, min_x - pad),
                                max(0, min_y - pad),
                                min(w, max_x + pad),
                                min(h, max_y + pad),
                            ]

                            # Smooth box with previous box (EMA)
                            if tracked_box is not None:
                                alpha = 0.55
                                tracked_box = [
                                    alpha * nb + (1 - alpha) * ob
                                    for nb, ob in zip(new_box, tracked_box)
                                ]
                            else:
                                tracked_box = new_box

            if not found_hand:
                lost_frames += 1
                if lost_frames > 15:
                    state = "LOCKING"
                    tracked_box = None

            # Visual overlay
            if state == "LOCKING":
                cv2.rectangle(frame, (default_box[0], default_box[1]), (default_box[2], default_box[3]), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    "Place Black Glove Here to Lock",
                    (default_box[0], default_box[1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
            else:
                bx1, by1, bx2, by2 = [int(v) for v in tracked_box]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "TRACKING LOCK",
                    (bx1, by1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # Info text
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} ({args.device.upper()}) | State: {state}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                "'r': Reset Lock | 's': Save Snapshot | 'q': Quit",
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 200, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Black Cotton Glove Tracker", frame)

            # Auto-save snapshot every 1.5s for inspection
            if curr_time - last_snap_time > 1.5:
                last_snap_time = curr_time
                cv2.imwrite("/Users/taminhtri/VR/live_snapshot.jpg", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break
            elif key == ord("r"):
                state = "LOCKING"
                tracked_box = None
                print("Reset tracking lock.")
            elif key == ord("s"):
                cv2.imwrite("/Users/taminhtri/VR/live_snapshot.jpg", frame)
                print("Saved snapshot.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
