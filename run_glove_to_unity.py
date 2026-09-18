#!/usr/bin/env python3
"""
Glove -> Unity UDP Bridge.

Live-tracks the white haptic glove using the fine-tuned RTMPose model
(same detection/tracking logic as run_white_haptics_glove.py) and streams
finger data to Unity's FingerUDPReceiver.cs over UDP, in the format it
expects: "thumb:0.xx,index:0.xx,middle:0.xx,ring:0.xx,pinky:0.xx,pinch:0.xx"

Scope decision (current state of fine-tuning): only THUMB and INDEX are
tracked reliably enough to send real live values -- these are also the
only 2 fingers that matter for the pinch gesture. MIDDLE/RING/PINKY are
sent as a FIXED closed value instead of the live (still unreliable)
model prediction, since bad live values for those would be worse than a
static default.
"""

import argparse
import os
import socket
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

import cv2
import numpy as np
import torch
from mmpose.apis import inference_topdown, init_model

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(_BASE_DIR, "mmpose", "configs", "hand_2d_keypoint", "rtmpose", "hand5", "rtmpose-m_8xb256-210e_hand5-256x256.py")
_FINETUNED_CKPT = os.path.join(_BASE_DIR, "checkpoints", "rtmpose_glove_finetuned.pth")
_ORIGINAL_CKPT_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"
CHECKPOINT_FILE = _FINETUNED_CKPT if os.path.exists(_FINETUNED_CKPT) else _ORIGINAL_CKPT_URL

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_MCPS = [1, 5, 9, 13, 17]

FINGER_COLORS = [
    (0, 145, 255),   # Thumb: Orange
    (240, 200, 0),   # Index: Sky Blue
    (210, 80, 255),  # Middle: Pink
    (220, 60, 160),  # Ring: Purple
    (50, 230, 80),   # Pinky: Neon Green
]
HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Fixed bend sent for middle/ring/pinky -- not tracked live (see docstring).
# 1.0 = fully closed, matches the deadzone-adjusted "closed fist" look on
# the Unity side.
FIXED_MRP_BEND = 1.0

# Pinch distance thresholds, as a ratio of palm length (scale-invariant,
# unlike raw pixel distance) -- FAR = fingers apart (pinch=0), CLOSE =
# fingertips touching (pinch=1).
PINCH_FAR_RATIO = 1.3
PINCH_CLOSE_RATIO = 0.15
PINCH_SMOOTH_ALPHA = 0.3


class OneEuroFilter:
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
    """0-1 flexion for each of the 5 fingers (thumb..pinky)."""
    wrist = kpts[0]
    mcp_mid = kpts[9]
    palm_len = max(np.linalg.norm(mcp_mid - wrist), 1e-4)
    flex_values = []
    for tip_idx, mcp_idx in zip(FINGER_TIPS, FINGER_MCPS):
        dist_tip_wrist = np.linalg.norm(kpts[tip_idx] - wrist)
        ratio = dist_tip_wrist / palm_len
        bend = np.clip((1.90 - ratio) / 1.05, 0.0, 1.0)
        flex_values.append(float(bend))
    return flex_values


# Bend joint that goc/giua/dau cua rieng ngon cai va ngon tro, tinh THANG
# tu hinh dang that cua 4 diem da detect (khong con dung khoang cach dau
# ngon-co tay nua -- cach do bi lan giua "ngon cong" voi "ca ban tay xoay
# ra xa camera", khien tay ao khong bam sat dang tay that).
MAX_BEND_DEG = 80.0


def _joint_bend_angle_deg(v1, v2):
    """Goc (do) giua 2 vector doan xuong: 0 = thang hang (khop khong cong),
    cang lon = khop cang gap/cong nhieu."""
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-6:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def calculate_finger_joint_bends(kpts, base_idx):
    """3 gia tri 0..1 (khop goc/giua/dau) cho 1 ngon, tu 4 diem that
    [wrist -> P0(goc) -> P1(giua) -> P2(gan dau) -> P3(dau ngon)] tai
    kpts[base_idx : base_idx+4]. Day la GOC THAT tai tung khop, khong phai
    1 gia tri chung uoc luong tho roi chia deu qua 3 khop nhu truoc."""
    wrist = kpts[0]
    p0, p1, p2, p3 = kpts[base_idx], kpts[base_idx + 1], kpts[base_idx + 2], kpts[base_idx + 3]
    seg0 = p0 - wrist
    seg1 = p1 - p0
    seg2 = p2 - p1
    seg3 = p3 - p2
    angle_proximal = _joint_bend_angle_deg(seg0, seg1)
    angle_intermediate = _joint_bend_angle_deg(seg1, seg2)
    angle_distal = _joint_bend_angle_deg(seg2, seg3)
    return [
        float(np.clip(angle_proximal / MAX_BEND_DEG, 0.0, 1.0)),
        float(np.clip(angle_intermediate / MAX_BEND_DEG, 0.0, 1.0)),
        float(np.clip(angle_distal / MAX_BEND_DEG, 0.0, 1.0)),
    ]


def is_valid_hand(kpts, scores, conf_thr=0.25):
    wrist = kpts[0]
    core_scores = scores[[0, 1, 5, 9, 13, 17]]
    core_conf = float(np.mean(core_scores))
    if core_conf < conf_thr:
        return False, core_conf
    palm_len = np.linalg.norm(kpts[9] - wrist)
    if palm_len < 30.0:
        return False, core_conf
    knuckle_span = np.linalg.norm(kpts[17] - kpts[5])
    if knuckle_span < 20.0 or knuckle_span > palm_len * 2.2:
        return False, core_conf
    thumb_tip_dist = np.linalg.norm(kpts[4] - kpts[1])
    if thumb_tip_dist > palm_len * 2.0:
        return False, core_conf
    return True, core_conf


def draw_skeleton(img, kpts, scores, score_thr=0.20, thickness=3):
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
            cv2.circle(img, pt, max(4, thickness + 3), (20, 20, 20), 1, cv2.LINE_AA)


def draw_hud(img, thumb_bend, index_bend, pinch, fps, sent_ok):
    h, w = img.shape[:2]
    panel_w = 230
    x0, y0 = w - panel_w - 15, 18
    overlay = img.copy()
    cv2.rectangle(overlay, (x0 - 10, y0 - 6), (w - 10, y0 + 150), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.putText(img, "UNITY UDP BRIDGE", (x0, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    status_color = (0, 255, 0) if sent_ok else (0, 0, 255)
    cv2.putText(img, f"Sending: {'OK' if sent_ok else 'HAND NOT FOUND'}", (x0, y0 + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA)
    cv2.putText(img, f"Thumb: {thumb_bend:.2f}", (x0, y0 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, FINGER_COLORS[0], 1)
    cv2.putText(img, f"Index: {index_bend:.2f}", (x0, y0 + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.42, FINGER_COLORS[1], 1)
    cv2.putText(img, f"Pinch: {pinch:.2f}", (x0, y0 + 106), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
    cv2.putText(img, "Mid/Ring/Pinky: fixed closed", (x0, y0 + 128), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1)
    cv2.putText(img, f"FPS: {fps:.1f}", (x0, y0 + 146), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)


def main():
    global CHECKPOINT_FILE
    parser = argparse.ArgumentParser(description="Glove tracker -> Unity UDP bridge")
    parser.add_argument("--cam", type=int, default=0, help="Camera index")
    parser.add_argument("--udp-ip", type=str, required=True, help="Target IP for Unity's FingerUDPReceiver (the Quest's IP on your LAN)")
    parser.add_argument("--udp-port", type=int, default=5005, help="Target UDP port (must match FingerUDPReceiver's _port)")
    parser.add_argument("--conf-thr", type=float, default=0.26, help="Min core-joint confidence to accept a detection")
    parser.add_argument(
        "--original",
        action="store_true",
        help="Force the ORIGINAL (non-fine-tuned) checkpoint. Thumb/index tracking on the glove was "
             "found to be more stable with this than with the fine-tuned checkpoint; the other 3 "
             "fingers are ignored downstream (FingerUDPReceiver locks them) so their instability "
             "with the original model doesn't matter.",
    )
    if torch.cuda.is_available():
        _default_device = "cuda"
    elif torch.backends.mps.is_available():
        _default_device = "mps"
    else:
        _default_device = "cpu"
    parser.add_argument("--device", type=str, default=_default_device, help="Device: 'cuda', 'mps' or 'cpu'")
    args = parser.parse_args()

    if args.original:
        CHECKPOINT_FILE = _ORIGINAL_CKPT_URL

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"Initializing camera {args.cam} at 720p (30 FPS)...")
    # CAP_DSHOW instead of the default MSMF backend: MSMF on Windows can get
    # into a broken state (grabFrame fails with HRESULT errors) after a prior
    # process was killed without releasing the camera; DirectShow re-opens
    # the device more reliably.
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ckpt_source = "FINE-TUNED (glove)" if CHECKPOINT_FILE == _FINETUNED_CKPT else "ORIGINAL (bare-hand only)"
    print(f"Loading RTMPose-m on {args.device.upper()}... [{ckpt_source}]")
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=args.device)

    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    dummy_box = np.array([[400, 200, 880, 680]])
    _ = inference_topdown(model, dummy_img, bboxes=dummy_box)
    if args.device == "mps":
        torch.mps.synchronize()

    win_name = "Glove -> Unity Bridge"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    filter_kpts = None
    state = "SEARCHING"
    tracked_box = None
    lost_frames = 0
    prev_time = time.time()
    fps = 0.0
    pinch_amount = 0.0

    print(f"\n[READY] Streaming to {args.udp_ip}:{args.udp_port}")
    print("Only thumb+index are live-tracked; middle/ring/pinky are sent fixed-closed.")
    print("Press 'q' to quit.\n")

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

            candidate_boxes = []
            if state == "TRACKING" and tracked_box is not None:
                candidate_boxes.append(tracked_box)
            else:
                candidate_boxes.append(np.array([w * 0.35, h * 0.25, w * 0.85, h * 0.85]))
                candidate_boxes.append(np.array([w * 0.20, h * 0.40, w * 0.80, h * 0.98]))

            found_hand = False
            best_kpts = None
            best_scores = None

            for c_box in candidate_boxes:
                bx1 = max(0, min(w - 60, int(c_box[0])))
                by1 = max(0, min(h - 60, int(c_box[1])))
                bx2 = max(bx1 + 60, min(w, int(c_box[2])))
                by2 = max(by1 + 60, min(h, int(c_box[3])))
                eval_box = np.array([[bx1, by1, bx2, by2]])

                results = inference_topdown(model, frame, bboxes=eval_box)
                if args.device == "mps":
                    torch.mps.synchronize()

                if len(results) > 0 and hasattr(results[0], "pred_instances"):
                    inst = results[0].pred_instances
                    kpts = inst.keypoints[0]
                    scores = inst.keypoint_scores[0]
                    valid, _ = is_valid_hand(kpts, scores, conf_thr=args.conf_thr)
                    if valid:
                        best_kpts = kpts
                        best_scores = scores
                        found_hand = True
                        break

            thumb_bend = 0.0
            index_bend = 0.0

            if found_hand and best_kpts is not None:
                lost_frames = 0
                state = "TRACKING"

                if filter_kpts is None:
                    filter_kpts = OneEuroFilter(curr_time, best_kpts)
                    smoothed_kpts = best_kpts
                else:
                    smoothed_kpts = filter_kpts(curr_time, best_kpts)

                min_x, max_x = np.min(smoothed_kpts[:, 0]), np.max(smoothed_kpts[:, 0])
                min_y, max_y = np.min(smoothed_kpts[:, 1]), np.max(smoothed_kpts[:, 1])
                cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
                palm_len = np.linalg.norm(smoothed_kpts[9] - smoothed_kpts[0])
                box_side = max((max_x - min_x) * 1.35, (max_y - min_y) * 1.35, palm_len * 2.8, 140.0)
                half = box_side / 2.0
                new_box = np.array([max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)])
                tracked_box = 0.70 * new_box + 0.30 * tracked_box if tracked_box is not None else new_box

                draw_skeleton(frame, smoothed_kpts, best_scores)

                # Goc that tai tung khop (goc/giua/dau) cho rieng ngon cai(1..4)
                # va ngon tro(5..8), tinh THANG tu hinh dang 4 diem da detect --
                # thay the calculate_finger_flex (chi dung khoang cach dau ngon-co
                # tay, khong phan anh dung hinh dang tung khop).
                thumb_joints = calculate_finger_joint_bends(smoothed_kpts, 1)
                index_joints = calculate_finger_joint_bends(smoothed_kpts, 5)
                # Van giu 1 gia tri dai dien (khop giua) chi de hien HUD debug,
                # khong dung de dieu khien Unity nua.
                thumb_bend, index_bend = thumb_joints[1], index_joints[1]

                # Continuous pinch amount from thumb-tip <-> index-tip distance,
                # normalized by palm length so it works at any distance from camera.
                pinch_dist = np.linalg.norm(smoothed_kpts[4] - smoothed_kpts[8])
                pinch_ratio = pinch_dist / max(palm_len, 1e-4)
                raw_pinch = (PINCH_FAR_RATIO - pinch_ratio) / (PINCH_FAR_RATIO - PINCH_CLOSE_RATIO)
                raw_pinch = float(np.clip(raw_pinch, 0.0, 1.0))
                pinch_amount = PINCH_SMOOTH_ALPHA * raw_pinch + (1 - PINCH_SMOOTH_ALPHA) * pinch_amount

                msg = (
                    f"thumb:{thumb_bend:.3f},index:{index_bend:.3f},"
                    f"thumb0:{thumb_joints[0]:.3f},thumb1:{thumb_joints[1]:.3f},thumb2:{thumb_joints[2]:.3f},"
                    f"index0:{index_joints[0]:.3f},index1:{index_joints[1]:.3f},index2:{index_joints[2]:.3f},"
                    f"middle:{FIXED_MRP_BEND:.3f},ring:{FIXED_MRP_BEND:.3f},pinky:{FIXED_MRP_BEND:.3f},"
                    f"pinch:{pinch_amount:.3f}"
                )
                sock.sendto(msg.encode("utf-8"), (args.udp_ip, args.udp_port))

                bx1, by1, bx2, by2 = [int(v) for v in tracked_box]
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
            else:
                lost_frames += 1
                if lost_frames > 12:
                    state = "SEARCHING"
                    tracked_box = None
                    filter_kpts = None

            draw_hud(frame, thumb_bend, index_bend, pinch_amount, fps, found_hand)

            header_color = (0, 255, 0) if state == "TRACKING" else (0, 200, 255)
            cv2.putText(frame, f"FPS: {fps:.1f} ({args.device.upper()}) | State: {state} -> {args.udp_ip}:{args.udp_port}",
                        (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, header_color, 2, cv2.LINE_AA)

            cv2.imshow(win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sock.close()


if __name__ == "__main__":
    main()
