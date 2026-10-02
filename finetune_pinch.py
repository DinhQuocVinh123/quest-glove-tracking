#!/usr/bin/env python3
"""
Fine-tune RTMPose (ban goc, dung voi --original) cho tay deo GANG, tu nhan da
co trong labels/:

  * labels/auto_labels.jsonl  (auto_label_glove.py) -- status "auto":
      - source "model": khung model da bam tot  -> giu cho model KHONG QUEN
      - source "flow" : khung loi, bam diem qua flow 2 chieu
  * labels/glove_labels.jsonl (label_glove_frames.py / Claude) -- nhan tay/9 diem
      visible: 1 = NHIN THAY (trong so 1), 2 = UOC LUONG (trong so 0.6), 0 = KHONG BIET (bo qua)

Xu ly anh GIONG HET luc chay that (run_glove_quest_stream.py):
  - dem vien xam 114 quanh anh (EDGE_PAD_RATIO = 0.25)
  - khung bao hinh vuong: khung GOI Y tu Quest (hint, canh = max(hsize*2.8, 140))
    hoac khung bam theo 21 diem (canh = max(1.35*be rong diem, 2.8*long ban tay, 140))
  - GetBBoxCenterScale(padding 1.25) + TopdownAffine 256x256 cua chinh model

Tach tap KIEM TRA theo DOAN THOI GIAN (moi doan 5 giay, cu 5 doan lay 1), khong
tron ngau nhien -- khung ke nhau gan nhu giong het, tron ngau nhien se cho ket
qua dep gia. Tot nhat: kiem tra tren 1 BAN GHI KHAC hoan toan (--eval-rec).

Cach dung (TAT run_glove_quest_stream.py truoc -- GPU chi co 2 GB):
  python finetune_pinch.py                       # train + so sanh truoc/sau
  python finetune_pinch.py --eval-only           # chi do model goc
  python finetune_pinch.py --eval-rec recordings/<ban ghi khac>   # them do tren ban ghi rieng
Ket qua: checkpoints/rtmpose_glove_pinch.pth. Chay voi:
  python run_glove_quest_stream.py --checkpoint checkpoints/rtmpose_glove_pinch.pth --record
"""
import argparse
import csv
import glob
import importlib.util
import json
import math
import os
import random
import sys
import time

# Thu muc nay co thu muc con ten "mmpose" (repo) che mat package mmpose that.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != _BASE_DIR]

import cv2
import numpy as np
import torch
from mmcv.transforms import Compose
from mmengine.dataset import pseudo_collate
from mmpose.apis import inference_topdown, init_model

CONFIG_FILE = os.path.join(_BASE_DIR, "mmpose", "configs", "hand_2d_keypoint", "rtmpose", "hand5",
                           "rtmpose-m_8xb256-210e_hand5-256x256.py")
ORIGINAL_CKPT = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                 "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth")
AUTO_FILE = os.path.join(_BASE_DIR, "labels", "auto_labels.jsonl")
LABEL_FILE = os.path.join(_BASE_DIR, "labels", "glove_labels.jsonl")
COLOR_FILE = os.path.join(_BASE_DIR, "labels", "color_labels.jsonl")   # color_label_glove.py
OUT_CKPT = os.path.join(_BASE_DIR, "checkpoints", "rtmpose_glove_pinch.pth")

EDGE_PAD_RATIO = 0.25          # giong run_glove_quest_stream.py
PAD_VALUE = (114, 114, 114)
HINT_BOX_SCALE = 2.8
ESTIMATED_WEIGHT = 0.6         # >= 0.5 (SimCCLabel bo qua diem < 0.5)
HARD_REPEAT = 3                # khung kho (flow / nhan tay) lap lai 3 lan moi epoch
PINCH_POINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8]   # co tay + ngon cai + ngon tro


def load_runtime_module():
    """Lay is_valid_hand / hint_offset tu chinh script chay that (khong chay main)."""
    spec = importlib.util.spec_from_file_location("rgqs", os.path.join(_BASE_DIR, "run_glove_quest_stream.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Du lieu ----------------------------------------------------------------

def load_samples(use_claude=False, use_color=True, color_stride=1):
    """Tra ve list mau: recording, frame, image, kpts (21,2), weight (21,), hard, hint, full."""
    by_key = {}
    if use_color and os.path.exists(COLOR_FILE):
        # Nhan tu bang keo mau: 6 diem co bang (visible 1), diem goc/co tay tu model (visible 2)
        for line in open(COLOR_FILE, encoding="utf-8"):
            r = json.loads(line)
            # Khung DE ke nhau gan nhu giong het: chi lay 1/color_stride; khung KHO giu het
            if not r.get("hard") and r["frame"] % color_stride:
                continue
            vis = np.array(r["visible"], np.float32)
            w = np.where(vis == 1, 1.0, np.where(vis == 2, ESTIMATED_WEIGHT, 0.0)).astype(np.float32)
            by_key[(r["recording"], r["frame"])] = dict(
                recording=r["recording"], frame=r["frame"], kpts=np.array(r["kpts"], np.float32),
                weight=w, hard=bool(r.get("hard")), full=False, source="color")
    if os.path.exists(AUTO_FILE):
        for line in open(AUTO_FILE, encoding="utf-8"):
            r = json.loads(line)
            if r["status"] != "auto" or not r.get("kpts"):
                continue
            w = np.array(r["visible"], np.float32)
            if (r["recording"], r["frame"]) in by_key:
                continue  # da co nhan mau (chinh xac hon nhan tu dong theo model/flow)
            by_key[(r["recording"], r["frame"])] = dict(
                recording=r["recording"], frame=r["frame"], kpts=np.array(r["kpts"], np.float32),
                weight=w, hard=r["source"] != "model", full=r["source"] == "model", source=r["source"])
    if os.path.exists(LABEL_FILE):
        for line in open(LABEL_FILE, encoding="utf-8"):
            r = json.loads(line)
            key = (r["recording"], r["frame"])
            if r.get("skip"):
                by_key.pop(key, None)       # nguoi/Claude xem va BO khung nay -> bo luon nhan tu dong
                continue
            if r.get("source") == "claude" and not use_claude:
                # Nhan 9 diem Claude dat bang mat -- nguoi dung danh gia KHONG du chinh xac.
                # Chi dung sau khi nguoi da sua (label_glove_frames.py --review-claude luu lai
                # thanh nhan cua nguoi, khong con source "claude").
                by_key.pop(key, None)
                continue
            vis = np.array(r["visible"], np.float32)
            w = np.where(vis == 1, 1.0, np.where(vis == 2, ESTIMATED_WEIGHT, 0.0)).astype(np.float32)
            by_key[key] = dict(recording=r["recording"], frame=r["frame"], kpts=np.array(r["kpts"], np.float32),
                               weight=w, hard=True, full=bool((vis > 0).all()), source=r.get("source", "human"))

    hints, times = {}, {}
    for rec in {k[0] for k in by_key}:
        d = os.path.join(_BASE_DIR, "recordings", rec)
        for line in open(os.path.join(d, "keypoints.jsonl"), encoding="utf-8"):
            k = json.loads(line)
            hints[(rec, k["frame"])] = k.get("hint")
        for row in csv.DictReader(open(os.path.join(d, "log.csv"), encoding="utf-8")):
            times[(rec, int(row["frame"]))] = float(row["t"])
    samples = []
    for key, s in by_key.items():
        s["image"] = os.path.join(_BASE_DIR, "recordings", key[0], f"{key[1]:05d}.jpg")
        s["hint"] = hints.get(key)
        s["t"] = times.get(key, 0.0)
        if s["weight"].sum() > 0 and os.path.exists(s["image"]):
            samples.append(s)
    return samples


def split_by_time(samples, chunk=5.0, every=5):
    """Doan 5 giay thu 5, 10, 15... -> tap kiem tra."""
    train, val = [], []
    for s in samples:
        (val if int(s["t"] // chunk) % every == every - 1 else train).append(s)
    return train, val


def runtime_boxes(s):
    """Cac khung bao ma luc chay that co the dung cho khung nay (toa do anh GOC)."""
    boxes = []
    if s["hint"]:
        hx, hy, hs = s["hint"]
        half = max(hs * HINT_BOX_SCALE, 140.0) / 2
        boxes.append([hx - half, hy - half, hx + half, hy + half])
    if s["full"]:
        p = s["kpts"]
        lo, hi = p.min(0), p.max(0)
        palm = float(np.linalg.norm(p[9] - p[0]))
        side = max((hi - lo).max() * 1.35, palm * 2.8, 140.0)
        c = (lo + hi) / 2
        boxes.append([c[0] - side / 2, c[1] - side / 2, c[0] + side / 2, c[1] + side / 2])
    if not boxes:  # khong co hint va nhan thieu diem: khung quanh cac diem da biet
        p = s["kpts"][s["weight"] > 0]
        lo, hi = p.min(0), p.max(0)
        side = max((hi - lo).max() * 1.8, 140.0)
        c = (lo + hi) / 2
        boxes.append([c[0] - side / 2, c[1] - side / 2, c[0] + side / 2, c[1] + side / 2])
    return boxes


def padded_image(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    pad = int(EDGE_PAD_RATIO * max(w, h))
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=PAD_VALUE), pad


class GloveDataset(torch.utils.data.Dataset):
    def __init__(self, samples, codec, input_size):
        self.items = []
        for s in samples:
            self.items += [s] * (HARD_REPEAT if s["hard"] else 1)
        self.pipeline = Compose([
            dict(type="GetBBoxCenterScale"),
            dict(type="RandomBBoxTransform", shift_factor=0.1, scale_factor=[0.8, 1.25], rotate_factor=40,
                 rotate_prob=0.6),
            dict(type="TopdownAffine", input_size=input_size),
            dict(type="mmdet.YOLOXHSVRandomAug"),
            dict(type="GenerateTarget", encoder=codec),
            dict(type="PackPoseInputs"),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        s = self.items[i]
        img, pad = padded_image(s["image"])
        box = np.array(random.choice(runtime_boxes(s)), np.float32) + pad
        kp = s["kpts"] + pad
        img = cutout(img, box, kp, s["weight"])
        data = dict(img=img, img_shape=img.shape[:2], ori_shape=img.shape[:2], img_path=s["image"],
                    bbox=box[None], bbox_score=np.ones(1, np.float32),
                    keypoints=kp[None], keypoints_visible=s["weight"][None].copy(),
                    id=i, img_id=i, category_id=1)
        return self.pipeline(data)


def cutout(img, box, kp, weight, prob=0.3):
    """Che 1 o chu nhat xam len phan tay KHONG phai ngon cai/tro (mo phong bi che khuat)."""
    if random.random() > prob:
        return img
    x1, y1, x2, y2 = box
    side = x2 - x1
    for _ in range(10):
        w, h = random.uniform(0.15, 0.3) * side, random.uniform(0.15, 0.3) * side
        cx, cy = random.uniform(x1, x2), random.uniform(y1, y2)
        r = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        inside = [(r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]) for p, wt in zip(kp[:9], weight[:9]) if wt > 0]
        if not any(inside):
            img = img.copy()
            a, b = int(max(r[1], 0)), int(max(r[0], 0))
            img[a:int(r[3]), b:int(r[2])] = PAD_VALUE
            return img
    return img


# --- Danh gia (giong luc chay that) ----------------------------------------

def evaluate(model, samples, rt, tag):
    """Chay model tren tung khung voi khung GOI Y (nhu luc chua bam) va do:
    - sai so 9 diem ngon cai/tro (don vi = kich thuoc tay theo Quest)
    - % khung dau ngon cai & tro dung (< 0.2) VA qua kiem tra hop le luc chay that."""
    model.eval()
    errs, tips_ok, accepted, n = [], 0, 0, 0
    for s in samples:
        if not s["hint"]:
            continue
        img, pad = padded_image(s["image"])
        box = np.array(runtime_boxes(s)[0], np.float32) + pad
        with torch.no_grad():
            res = inference_topdown(model, img, bboxes=box[None])
        kpts = res[0].pred_instances.keypoints[0] - pad
        scores = res[0].pred_instances.keypoint_scores[0]
        size = max(s["hint"][2], 1.0)
        m = s["weight"][:9] > 0
        e = np.linalg.norm(kpts[:9] - s["kpts"][:9], axis=1) / size
        errs.append(e[m])
        tip_good = all(e[k] < 0.2 for k in (4, 8) if m[k])
        valid, _, _ = rt.is_valid_hand(kpts, scores, conf_thr=0.15, frame_shape=cv2.imread(s["image"]).shape)
        valid = valid and rt.hint_offset(kpts, s["hint"]) <= rt.HINT_MAX_OFFSET
        tips_ok += tip_good
        accepted += tip_good and valid
        n += 1
    if n == 0:
        print(f"  [{tag}] khong co khung nao (thieu hint)")
        return None
    allerr = np.concatenate(errs)
    print(f"  [{tag}] {n} khung | sai so 9 diem: trung vi {np.median(allerr):.3f}, p90 {np.percentile(allerr, 90):.3f} "
          f"| dau ngon cai+tro dung: {100 * tips_ok / n:.0f}% | dung VA qua kiem tra luc chay: {100 * accepted / n:.0f}%")
    return float(np.median(allerr)), accepted / n


def eval_recording(model, rec_dir, rt, max_frames=400):
    """Ban ghi KHONG co nhan: do % khung tim duoc tay hop le (tu hint), nhu luc chay that."""
    rows = list(csv.DictReader(open(os.path.join(rec_dir, "log.csv"), encoding="utf-8")))
    hints = {json.loads(l)["frame"]: json.loads(l).get("hint") for l in open(os.path.join(rec_dir, "keypoints.jsonl"), encoding="utf-8")}
    frames = [int(r["frame"]) for r in rows if hints.get(int(r["frame"]))]
    frames = frames[::max(1, len(frames) // max_frames)]
    ok = 0
    model.eval()
    for f in frames:
        path = os.path.join(rec_dir, f"{f:05d}.jpg")
        img, pad = padded_image(path)
        hx, hy, hs = hints[f]
        half = max(hs * HINT_BOX_SCALE, 140.0) / 2
        box = np.array([[hx - half, hy - half, hx + half, hy + half]], np.float32) + pad
        with torch.no_grad():
            res = inference_topdown(model, img, bboxes=box)
        kpts = res[0].pred_instances.keypoints[0] - pad
        scores = res[0].pred_instances.keypoint_scores[0]
        valid, _, _ = rt.is_valid_hand(kpts, scores, conf_thr=0.15, frame_shape=img.shape)
        ok += valid and rt.hint_offset(kpts, hints[f]) <= rt.HINT_MAX_OFFSET
    print(f"  [{os.path.basename(rec_dir)}] {len(frames)} khung co tay (theo Quest): tim duoc tay hop le {100 * ok / max(len(frames), 1):.0f}%")


# --- Train -------------------------------------------------------------------

def set_bn_eval(model):
    """Batch nho (GPU 2 GB): giu nguyen thong ke BatchNorm cua model goc."""
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def main():
    ap = argparse.ArgumentParser(description="Fine-tune RTMPose cho tay deo gang (pinch).")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4, help="Toc do hoc cua HEAD; backbone dung lr/4.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--eval-rec", nargs="*", default=[], help="Ban ghi KHAC (khong train) de do ti le tim duoc tay.")
    ap.add_argument("--out", default=OUT_CKPT)
    ap.add_argument("--use-claude-labels", action="store_true",
                    help="Dung ca nhan 9 diem Claude dat CHUA duoc nguoi sua (mac dinh: khong).")
    ap.add_argument("--no-color", action="store_true", help="Khong dung nhan bang keo mau (labels/color_labels.jsonl).")
    ap.add_argument("--compare", nargs="*", default=[], help="Checkpoint khac de do cung tap kiem tra (vd model dang dung).")
    ap.add_argument("--init", default=None, help="Train TIEP tu checkpoint nay thay vi model goc (can it epoch hon).")
    ap.add_argument("--color-stride", type=int, default=1, help="Nhan mau: khung de chi lay 1/N khung (khung kho giu het).")
    args = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    rt = load_runtime_module()
    samples = load_samples(args.use_claude_labels, use_color=not args.no_color, color_stride=max(1, args.color_stride))
    train, val = split_by_time(samples)
    n_hard = sum(s["hard"] for s in samples)
    n_color = sum(s["source"] == "color" for s in samples)
    print(f"{len(samples)} mau ({n_hard} kho, {len(samples) - n_hard} model tot, {n_color} nhan mau) "
          f"-> train {len(train)}, kiem tra {len(val)}")
    val_hard = [s for s in val if s["hard"]]
    val_easy = [s for s in val if not s["hard"]]
    val_color = [s for s in val if s["source"] == "color"]

    def report(m, tag):
        print(f"\n{tag}:")
        r = evaluate(m, val_hard, rt, "kiem tra - khung KHO")
        evaluate(m, val_easy, rt, "kiem tra - khung de")
        if val_color:
            evaluate(m, val_color, rt, "kiem tra - nhan MAU (gang moi)")
        for d in args.eval_rec:
            eval_recording(m, d, rt)
        return r

    start_ckpt = args.init or ORIGINAL_CKPT
    model = init_model(CONFIG_FILE, start_ckpt, device=args.device)
    base_hard = report(model, "MODEL KHOI DAU (" + os.path.basename(start_ckpt) + ")")
    for ck in args.compare:
        cm = init_model(CONFIG_FILE, ck, device=args.device)
        report(cm, f"MODEL {os.path.basename(ck)}")
        del cm
        torch.cuda.empty_cache()
    if args.eval_only:
        return

    codec = model.cfg.codec if "codec" in model.cfg else model.cfg.model.head.decoder
    ds = GloveDataset(train, codec, tuple(model.cfg.codec["input_size"]))
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                                         collate_fn=pseudo_collate, drop_last=True)
    head_params = list(model.head.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": backbone_params, "lr": args.lr / 4},
                             {"params": head_params, "lr": args.lr}], weight_decay=1e-4)
    total = args.epochs * len(loader)
    warm = min(50, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: (it + 1) / warm if it < warm else 0.5 * (1 + math.cos(math.pi * (it - warm) / max(total - warm, 1))))

    print(f"\nTRAIN {args.epochs} epoch x {len(loader)} buoc (batch {args.batch}) tren {args.device}")
    best = None
    for ep in range(args.epochs):
        model.train(); set_bn_eval(model)
        t0, run = time.time(), 0.0
        for batch in loader:
            data = model.data_preprocessor(batch, training=True)
            losses = model(**data, mode="loss")
            loss = sum(v for k, v in losses.items() if "loss" in k)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            run += loss.item()
        msg = f"epoch {ep + 1:2d}/{args.epochs} loss {run / len(loader):.4f} ({time.time() - t0:.0f}s)"
        if (ep + 1) % 5 == 0 or ep + 1 == args.epochs:
            print(msg)
            r = evaluate(model, val_hard, rt, "kiem tra - khung KHO")
            if r and (best is None or r[1] > best[0] or (r[1] == best[0] and r[0] < best[1])):
                best = (r[1], r[0], ep + 1)
                os.makedirs(os.path.dirname(args.out), exist_ok=True)
                torch.save({"state_dict": model.state_dict(),
                            "meta": {"dataset_meta": model.dataset_meta, "epoch": ep + 1,
                                     "samples": len(train), "base": start_ckpt}}, args.out)
                print(f"  -> luu {args.out}")
        else:
            print(msg)

    print("\nKET QUA (checkpoint tot nhat, epoch %d):" % best[2] if best else "\nKET QUA:")
    model = init_model(CONFIG_FILE, args.out, device=args.device)
    report(model, "MODEL MOI")
    if base_hard:
        print(f"(model goc, khung KHO: dung VA qua kiem tra {100 * base_hard[1]:.0f}%)")


if __name__ == "__main__":
    main()
