#!/usr/bin/env python3
"""
MMPose Live Webcam Hand Tracking (2D / 3D) for macOS.

Usage:
    python run_live.py              # Run 2D hand detection & pose (default)
    python run_live.py --task 3d    # Run 3D hand pose estimation
    python run_live.py --cam 1      # Use an external camera (index 1)
"""

import argparse
import sys
import time
import cv2
import torch
from mmpose.apis.inferencers import MMPoseInferencer


def parse_args():
    parser = argparse.ArgumentParser(description="MMPose Live Webcam Hand Tracking")
    parser.add_argument(
        "--task",
        type=str,
        choices=["2d", "3d"],
        default="2d",
        help="Tracking task: '2d' (RTMPose Hand, recommended for real-time) or '3d' (InterHand3D)",
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
        help="Inference device: 'mps' or 'cpu'",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("==================================================")
    print("      MMPose Live Webcam Hand Tracking")
    print("==================================================")
    print(f"Task:    {args.task.upper()}")
    print(f"Camera:  Device index {args.cam}")
    print(f"Device:  {args.device}")
    print("Controls: Press 'q' or 'ESC' in the window to quit")
    print("==================================================")

    # Initialize Camera
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print("\n[ERROR] Unable to access camera.")
        print("On macOS, please ensure Camera access is granted to your Terminal app:")
        print("    System Settings -> Privacy & Security -> Camera -> Enable Terminal")
        sys.exit(1)

    print("\nLoading models... (this takes a couple seconds)")
    if args.task == "2d":
        inferencer = MMPoseInferencer(pose2d="hand", device=args.device)
    else:
        inferencer = MMPoseInferencer(pose3d="hand3d", device=args.device)

    print("\nCamera stream started! Displaying window...")
    prev_time = time.time()
    fps = 0.0

    cv2.namedWindow("MMPose Live Hand Tracking", cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break

            # Mirror frame horizontally for natural user experience
            frame = cv2.flip(frame, 1)

            # Inference
            curr_time = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
            prev_time = curr_time

            for res in inferencer(frame, return_vis=True):
                vis_frames = res.get("visualization", [])
                if vis_frames:
                    vis_frame = vis_frames[0]
                else:
                    vis_frame = frame

                # Overlay FPS and instructions
                cv2.putText(
                    vis_frame,
                    f"FPS: {fps:.1f} ({args.device.upper()}) | Task: {args.task.upper()}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    vis_frame,
                    "Press 'q' to exit",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    1,
                    cv2.LINE_AA,
                )

                cv2.imshow("MMPose Live Hand Tracking", vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord("q"), 27]:
                print("\nExiting live hand tracking...")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Goodbye!")


if __name__ == "__main__":
    main()
