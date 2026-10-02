"""Xuat model RTMPose (.pth) sang ONNX de chay TREN KINH (Unity Inference Engine).

File ONNX nhan anh vung tay da cat san 256x256, RGB, gia tri 0..1 (dung nhu texture Unity),
va tra ve luon 21 diem (toa do trong anh 256x256) + do tin cay -- khong can xu ly them.
Phan chuan hoa mau (mean/std) va giai ma SimCC (argmax / 2) nam SAN trong file ONNX.

Chay (CPU la du, khong dung GPU):
    mmpose_venv\\Scripts\\python.exe export_onnx.py --checkpoint checkpoints\\rtmpose_glove_pinch_color3.pth
Kiem tra: so voi mmpose tren cac anh da ghi (--check recordings\\<gio>).
"""
import argparse
import json
import os
import sys
import time

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:  # giong run_glove_quest_stream.py: dung goi mmpose da cai, khong phai thu muc clone
    sys.path.remove(_script_dir)

import cv2
import numpy as np
import torch

from mmpose.apis import inference_topdown, init_model

CONFIG_FILE = os.path.join(_script_dir, "mmpose", "configs", "hand_2d_keypoint", "rtmpose", "hand5",
                           "rtmpose-m_8xb256-210e_hand5-256x256.py")
INPUT = 256
SPLIT = 2.0          # simcc_split_ratio trong config
BBOX_PADDING = 1.25  # GetBBoxCenterScale mac dinh
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)


class Wrapped(torch.nn.Module):
    """anh RGB 0..1 (1,3,256,256) -> keypoints (1,21,2) pixel trong anh 256, scores (1,21)."""

    def __init__(self, model, flip):
        super().__init__()
        self.flip = flip
        self.backbone = model.backbone
        self.head = model.head
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1) / 255.0)
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1) / 255.0)

    def forward(self, x):
        x = (x - self.mean) / self.std
        if self.flip:
            # Giong flip_test cua mmpose (config bat san): chay them anh lat ngang roi lay trung binh.
            # Tay: 21 diem khong co cap trai/phai -> flip_indices = giu nguyen thu tu.
            sx, sy = self.head(self.backbone(torch.cat([x, x.flip(3)], 0)))
            sx = (sx[0:1] + sx[1:2].flip(2)) * 0.5
            sy = (sy[0:1] + sy[1:2]) * 0.5
        else:
            sx, sy = self.head(self.backbone(x))       # (1,21,512) moi truc
        mx, ix = sx.max(dim=2)
        my, iy = sy.max(dim=2)
        kpts = torch.stack([ix, iy], dim=2).float() / SPLIT
        scores = torch.minimum(mx, my)                 # giong get_simcc_maximum
        return kpts, scores


def crop_affine(img_bgr, box):
    """Giong TopdownAffine cua mmpose: vung vuong = canh dai * 1.25 quanh tam hop. Tra ve anh 256 + (tam, canh)."""
    x1, y1, x2, y2 = box
    c = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
    s = max(x2 - x1, y2 - y1) * BBOX_PADDING
    k = INPUT / s
    M = np.array([[k, 0, INPUT / 2 - k * c[0]], [0, k, INPUT / 2 - k * c[1]]], np.float32)
    crop = cv2.warpAffine(img_bgr, M, (INPUT, INPUT), flags=cv2.INTER_LINEAR)
    return crop, c, s


def to_input(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join("checkpoints", "rtmpose_glove_pinch_color3.pth"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", default=None, help="thu muc recordings/<gio> de so voi mmpose")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--no-flip", action="store_true", help="bo anh lat ngang (mmpose bat san): nhanh gap doi, lech nhe so voi Python")
    args = ap.parse_args()
    ckpt = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(_script_dir, args.checkpoint)
    out = args.out or os.path.splitext(ckpt)[0] + ("_noflip" if args.no_flip else "") + ".onnx"

    model = init_model(CONFIG_FILE, ckpt, device="cpu")
    model.eval()
    net = Wrapped(model, flip=not args.no_flip).eval()
    dummy = torch.rand(1, 3, INPUT, INPUT)
    with torch.no_grad():
        torch.onnx.export(net, dummy, out, opset_version=15, input_names=["image"],
                          output_names=["keypoints", "scores"], do_constant_folding=True)
    import onnx
    onnx.checker.check_model(onnx.load(out))
    print("da xuat", out, f"{os.path.getsize(out) / 1e6:.1f} MB")

    if not args.check:
        return
    import onnxruntime as ort
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    rec = args.check if os.path.isabs(args.check) else os.path.join(_script_dir, args.check)
    lines = [json.loads(l) for l in open(os.path.join(rec, "keypoints.jsonl"))]
    lines = [l for l in lines if l.get("found") and l.get("kpts")]
    step = max(1, len(lines) // args.n)
    errs, confd, t_ort, t_mm = [], [], [], []
    for l in lines[::step][:args.n]:
        img = cv2.imread(os.path.join(rec, f"{l['frame']:05d}.jpg"))
        if img is None:
            continue
        k = np.array(l["kpts"])[:21]
        x1, y1 = k.min(0); x2, y2 = k.max(0)
        m = 0.15 * max(x2 - x1, y2 - y1)
        box = np.array([x1 - m, y1 - m, x2 + m, y2 + m], np.float32)
        t0 = time.perf_counter()
        r = inference_topdown(model, img, bboxes=box[None])[0].pred_instances
        t_mm.append(time.perf_counter() - t0)
        ref_k, ref_s = r.keypoints[0], r.keypoint_scores[0]
        crop, c, s = crop_affine(img, box)
        t0 = time.perf_counter()
        kp, sc = sess.run(None, {"image": to_input(crop)})
        t_ort.append(time.perf_counter() - t0)
        img_k = kp[0] / INPUT * s + c - s / 2
        palm = np.linalg.norm(ref_k[9] - ref_k[0]) + 1e-6
        errs.append(np.linalg.norm(img_k - ref_k, axis=1) / palm)
        confd.append(np.abs(sc[0] - ref_s))
    errs, confd = np.array(errs), np.array(confd)
    print(f"{len(errs)} anh | lech diem ONNX vs mmpose (/ long ban tay): med {np.median(errs):.4f} "
          f"p99 {np.percentile(errs, 99):.4f} max {errs.max():.4f} | lech do tin cay med {np.median(confd):.4f}")
    print(f"thoi gian CPU: onnxruntime {1000 * np.median(t_ort):.1f} ms, mmpose {1000 * np.median(t_mm):.1f} ms")


if __name__ == "__main__":
    main()
