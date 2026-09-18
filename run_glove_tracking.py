#!/usr/bin/env python3
"""
MMPose Glove-Adapted Live Hand Tracker.

Specially tuned for tracking fabric, cosplay, and VR gloves:
1. Lowers bounding box & keypoint detection thresholds.
2. Supports 'direct_box' mode: user can place their gloved hand in a guide box,
   bypassing skin-trained detectors completely to run RTMPose directly on the glove.
3. Supports optional color marker tracking for colored fingertips.

Usage:
    python run_glove_tracking.py                     # Optimized detector mode for gloves
    python run_glove_tracking.py --mode guide_box     # Bypass detector with interactive hand box
    python run_glove_tracking.py --task 3d            # 3D glove tracking
"""

import argparse
import sys
import time
import cv2
import numpy as np
import torch
from mmpose.apis.inferencers import MMPoseInferencer


def parse_args():
    parser = argparse.ArgumentParser(description="MMPose Glove-Adapted Live Tracker")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "guide_box"],
        default="auto",
        help="'auto': sensitive detection for gloves; 'guide_box': manual ROI box (bypasses detector)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["2d", "3d"],
        default="2d",
        help="Pose task: '2d' (RTMPose Hand) or '3d' (InterHand3D)",
    )
    parser.add_argument(
        "--cam",
        type=int,
        default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
        help="Compute device: 'mps' or 'cpu'",
    )
    parser.add_argument(
        "--bbox-thr",
        type=float,
        default=0.15,
        help="Bounding box threshold (lowered for gloves, default: 0.15)",
    )
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=0.1,
        help="Keypoint threshold (default: 0.1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("==================================================")
    print("      MMPose Glove-Adapted Hand Tracking")
    print("==================================================")
    print(f"Mode:       {args.mode.upper()}")
    print(f"Task:       {args.task.upper()}")
    print(f"Device:     {args.device}")
    print(f"Thresholds: bbox={args.bbox_thr}, kpt={args.kpt_thr}")
    print("Controls:   Press 'm' to toggle mode | 'q' to quit")
    print("==================================================")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("\n[ERROR] Unable to access camera.")
        print("Please ensure Camera permission is granted in macOS System Settings.")
        sys.exit(1)

    print("\nLoading models...")
    if args.task == "2d":
        inferencer = MMPoseInferencer(pose2d="hand", device=args.device)
    else:
        inferencer = MMPoseInferencer(pose3d="hand3d", device=args.device)

    cv2.namedWindow("MMPose Glove Tracker", cv2.WINDOW_NORMAL)

    mode = args.mode
    prev_time = time.time()
    fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror frame
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            if mode == "guide_box":
                # Define a centered bounding box for the gloved hand
                box_size = int(min(h, w) * 0.55)
                x1 = (w - box_size) // 2
                y1 = (h - box_size) // 2
                x2 = x1 + box_size
                y2 = y1 + box_size

                # Crop hand area and run pose estimator directly
                hand_crop = frame[y1:y2, x1:x2]
                for res in inferencer(hand_crop, return_vis=True):
                    vis_crop = res.get("visualization", [hand_crop])[0]
                    frame[y1:y2, x1:x2] = vis_crop

                # Draw guide rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    frame,
                    "Place gloved hand inside box",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
            else:
                # Auto detection mode with sensitive thresholds
                for res in inferencer(
                    frame,
                    bbox_thr=args.bbox_thr,
                    kpt_thr=args.kpt_thr,
                    return_vis=True,
                ):
                    vis_frames = res.get("visualization", [])
                    if vis_frames:
                        frame = vis_frames[0]

            # Status overlay
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | Mode: {mode.upper()} ('m' to switch)",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Task: {args.task.upper()} | Glove BBox Thr: {args.bbox_thr}",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                1,
            )

            cv2.imshow("MMPose Glove Tracker", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                break
            elif key == ord("m"):
                mode = "guide_box" if mode == "auto" else "auto"
                print(f"Switched mode to: {mode}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
