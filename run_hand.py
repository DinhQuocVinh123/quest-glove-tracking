#!/usr/bin/env python3
"""
MMPose Hand Detection & 2D/3D Pose Runner for macOS (Apple Silicon / Intel).

Usage Examples:
    # Run 2D hand detection & pose estimation on a test image:
    python run_hand.py --task 2d --input mmpose/tests/data/onehand10k/9.jpg

    # Run 3D hand pose estimation on a test image:
    python run_hand.py --task 3d --input mmpose/tests/data/interhand2.6m/image29590.jpg

    # Run real-time 2D hand pose from your Mac webcam:
    python run_hand.py --task 2d --webcam --show

    # Specify output directory and device:
    python run_hand.py --task 2d --input /path/to/image.jpg --out-dir output/ --device mps
"""

import argparse
import os
import sys
import torch

try:
    from mmpose.apis.inferencers import MMPoseInferencer
except ImportError:
    print("Error: MMPose is not installed in the active Python environment.")
    print("Please activate the mmpose conda environment first:")
    print("    conda activate mmpose")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run MMPose Hand Detection & 2D/3D Pose Estimation"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["2d", "3d"],
        default="2d",
        help="Pose estimation mode: '2d' (RTMPose Hand + RTMDet) or '3d' (InterHand3D)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input image, video, directory, or URL",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Use Mac webcam as live input (webcam id 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: 'mps' (Metal/GPU) or 'cpu'. Defaults to 'mps' if available.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="vis_results",
        help="Directory to save visual results",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the results interactively in a window",
    )
    parser.add_argument(
        "--draw-heatmap",
        action="store_true",
        help="Draw predicted heatmaps if model supports it",
    )
    return parser.parse_args()


def get_default_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()
    device = args.device or get_default_device()

    # Determine input source
    if args.webcam:
        inputs = "webcam:0"
    elif args.input:
        inputs = args.input
    else:
        # Provide sample images if no input specified
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if args.task == "2d":
            inputs = os.path.join(
                script_dir, "mmpose/tests/data/onehand10k/9.jpg"
            )
        else:
            inputs = os.path.join(
                script_dir, "mmpose/tests/data/interhand2.6m/image29590.jpg"
            )
        print(f"No --input or --webcam specified. Using sample image: {inputs}")

    print("=== MMPose Hand Runner ===")
    print(f"Task:    {args.task.upper()} Hand Pose Estimation")
    print(f"Input:   {inputs}")
    print(f"Device:  {device} (Apple Silicon MPS: {torch.backends.mps.is_available()})")
    print(f"Out Dir: {args.out_dir}")
    print("===========================")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.task == "2d":
        inferencer = MMPoseInferencer(
            pose2d="hand",
            device=device,
            show_progress=True,
        )
    else:
        inferencer = MMPoseInferencer(
            pose3d="hand3d",
            device=device,
            show_progress=True,
        )

    for result in inferencer(
        inputs=inputs,
        vis_out_dir=args.out_dir,
        show=args.show,
        draw_heatmap=args.draw_heatmap,
    ):
        pass

    print(f"\nInference completed successfully!")
    print(f"Results saved to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
