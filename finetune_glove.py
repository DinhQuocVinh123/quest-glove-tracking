#!/usr/bin/env python3
"""
Fine-Tune RTMPose-m on Paired White Haptic Glove Dataset.
Uses Apple Silicon Metal (MPS) to adapt keypoint estimation specifically
to the fabric, markers, wires, and sensor geometry of the haptic glove.
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from mmpose.apis import init_model

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(_BASE_DIR, "mmpose", "configs", "hand_2d_keypoint", "rtmpose", "hand5", "rtmpose-m_8xb256-210e_hand5-256x256.py")
CHECKPOINT_FILE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"
DATASET_DIR = os.path.join(_BASE_DIR, "dataset")
ANNOTATIONS_FILE = os.path.join(DATASET_DIR, "annotations.json")
OUTPUT_DIR = os.path.join(_BASE_DIR, "checkpoints")
OUTPUT_CKPT = os.path.join(OUTPUT_DIR, "rtmpose_glove_finetuned.pth")


class GlovePairDataset(Dataset):
    """Dataset loading paired glove crops and aligned ground-truth keypoints."""
    def __init__(self, ann_file, img_dir, input_size=(256, 256)):
        with open(ann_file, "r") as f:
            self.records = json.load(f)
        self.img_dir = img_dir
        self.input_size = input_size
        print(f"Loaded {len(self.records)} paired training samples.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = os.path.join(self.img_dir, rec["image_file"])
        img = cv2.imread(img_path)
        if img is None:
            # Fallback blank
            img = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)

        h_orig, w_orig = img.shape[:2]
        resized = cv2.resize(img, self.input_size)

        # gt_kpts_crop (written by run_split_screen_finetune.py) is already
        # relative to the saved crop image, so just scale it to 256x256.
        crop_kpts = np.array(rec["gt_kpts_crop"], dtype=np.float32)
        scale_x = self.input_size[0] / max(w_orig, 1)
        scale_y = self.input_size[1] / max(h_orig, 1)
        norm_kpts = crop_kpts.copy()
        norm_kpts[:, 0] *= scale_x
        norm_kpts[:, 1] *= scale_y

        # Manual-label records (manual_label_quest.py) only supply real
        # coordinates for 9/21 points (wrist+thumb+index); the rest are
        # filler values with keypoint_mask=0 so they never contribute to
        # the loss. Records without this field (auto-labeled) are treated
        # as fully supervised (mask of all 1s), same as before.
        mask = rec.get("keypoint_mask", [1] * len(norm_kpts))
        mask_tensor = torch.tensor(mask, dtype=torch.float32)

        # Normalize image to [0, 1] tensor (mean/std as per COCO)
        img_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        kpts_tensor = torch.from_numpy(norm_kpts).float()
        return img_tensor, kpts_tensor, mask_tensor


def soft_argmax_1d(logits, split_ratio):
    """Differentiable decode of a SimCC 1D logit distribution to a pixel
    coordinate: a softmax-weighted average over bin positions, instead of
    the (non-differentiable) hard argmax used at inference time.

    logits: (B, K, L) raw SimCC logits along one axis.
    Returns: (B, K) coordinates in the same pixel space as the input image
    (dividing by split_ratio undoes the SimCC sub-pixel bin split).
    """
    probs = torch.softmax(logits, dim=-1)
    bins = torch.arange(logits.shape[-1], device=logits.device, dtype=logits.dtype)
    return torch.sum(probs * bins, dim=-1) / split_ratio


def main():
    parser = argparse.ArgumentParser(description="Fine-tune RTMPose on White Haptic Glove")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"[ERROR] Annotations file {ANNOTATIONS_FILE} not found!")
        print("Please run 'python run_dual_hand_finetune.py' and press 't' to record pairs first.")
        sys.exit(1)

    dataset = GlovePairDataset(ANNOTATIONS_FILE, os.path.join(DATASET_DIR, "images"))
    if len(dataset) < 10:
        print(f"[WARNING] Only {len(dataset)} samples recorded. Recommend at least 30-50 pairs for good fine-tuning.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print(f"Loading RTMPose model on {args.device.upper()}...")
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=args.device)

    # Freeze backbone, train head
    for param in model.backbone.parameters():
        param.requires_grad = False

    head_params = [p for p in model.head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(head_params, lr=args.lr, weight_decay=1e-3)
    # reduction='none' so we can mask out unsupervised keypoints (manual-label
    # records only have real coordinates for 9/21 points) before averaging.
    loss_fn = nn.SmoothL1Loss(reduction="none")

    print(f"\nStarting fine-tuning for {args.epochs} epochs on {args.device.upper()}...")
    model.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        t0 = time.time()
        for batch_imgs, batch_kpts, batch_mask in loader:
            batch_imgs = batch_imgs.to(args.device)
            batch_kpts = batch_kpts.to(args.device)
            batch_mask = batch_mask.to(args.device)  # (B, 21)

            optimizer.zero_grad()

            # Forward through backbone & head
            feats = model.extract_feat(batch_imgs)
            # NOTE: model.head.predict() is an INFERENCE-only path -- it
            # decodes the SimCC distributions to pixel coordinates via a
            # numpy argmax, which has no gradient. That breaks backprop
            # entirely (loss.backward() fails with "does not require grad").
            # For training we instead take the raw (pre-decode) logits from
            # head.forward() and decode them ourselves with a *soft*-argmax
            # (softmax-weighted average over bins), which is differentiable
            # and converges to the same pixel coordinates as the real decode.
            pred_x_logits, pred_y_logits = model.head.forward(feats)
            split_ratio = model.head.simcc_split_ratio
            pred_x = soft_argmax_1d(pred_x_logits, split_ratio)
            pred_y = soft_argmax_1d(pred_y_logits, split_ratio)
            pred_tensor = torch.stack([pred_x, pred_y], dim=-1)

            # per_point_loss: (B, 21, 2) -> average the x/y pair per point,
            # then mask out unsupervised points before averaging over the
            # batch, so filler coordinates never influence the gradient.
            per_point_loss = loss_fn(pred_tensor, batch_kpts).mean(dim=-1)  # (B, 21)
            masked = per_point_loss * batch_mask
            loss = masked.sum() / batch_mask.sum().clamp(min=1.0)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        dt = time.time() - t0
        avg_loss = epoch_loss / max(len(loader), 1)
        print(f"Epoch [{epoch+1}/{args.epochs}] - Loss: {avg_loss:.4f} - Time: {dt:.2f}s")

    # Save fine-tuned checkpoint
    torch.save(model.state_dict(), OUTPUT_CKPT)
    print(f"\n[SUCCESS] Fine-tuned model saved to {OUTPUT_CKPT}!")
    print("You can now run the tracker with the fine-tuned checkpoint!")


if __name__ == "__main__":
    main()
