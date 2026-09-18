#!/usr/bin/env python3
"""
Chuyển dữ liệu thu từ Quest (dataset_from_quest/annotations.jsonl: ảnh đầy
đủ khung hình + 21 điểm khớp TAY TRẦN từ native tracking) thành đúng định
dạng dataset cũ (dataset/annotations.json: ảnh CẮT quanh tay găng + nhãn
đã quy đổi), để dùng lại y hệt review_dataset.py và finetune_glove.py.

Cách làm:
1. Với mỗi ảnh, biết trước vùng tay TRẦN (từ bare_hand_kpts) -- không cần
   dò tìm, đã có sẵn từ Quest.
2. Chạy model mmpose hiện có để tìm vùng TAY GĂNG trong phần còn lại của
   ảnh (né vùng tay trần ra, tránh nhận nhầm lại chính tay trần).
3. Áp đúng công thức quy đổi nhãn (xoay + co giãn, qua số phức) đã dùng ở
   run_split_screen_finetune.py -- lấy tay trần làm "thầy", quy đổi 21
   điểm khớp của nó sang không gian ảnh tay găng.
4. Lưu (ảnh cắt tay găng, nhãn) vào dataset/ -- CỘNG THÊM vào dữ liệu cũ,
   không ghi đè.
"""

import argparse
import json
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

import cv2
import numpy as np
import torch
from mmpose.apis import inference_topdown, init_model

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUEST_DIR = os.path.join(_BASE_DIR, "dataset_from_quest")
QUEST_IMG_DIR = os.path.join(QUEST_DIR, "images")
QUEST_ANNOTATIONS = os.path.join(QUEST_DIR, "annotations.jsonl")

OUT_DATASET_DIR = os.path.join(_BASE_DIR, "dataset")
OUT_IMG_DIR = os.path.join(OUT_DATASET_DIR, "images")
OUT_ANNOTATIONS = os.path.join(OUT_DATASET_DIR, "annotations.json")

CONFIG_FILE = os.path.join(_BASE_DIR, "mmpose", "configs", "hand_2d_keypoint", "rtmpose", "hand5", "rtmpose-m_8xb256-210e_hand5-256x256.py")
_FINETUNED_CKPT = os.path.join(_BASE_DIR, "checkpoints", "rtmpose_glove_finetuned.pth")
_ORIGINAL_CKPT_URL = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth"


def align_gt_to_glove(gt_kpts, glove_kpts):
    """Giống hệt hàm cùng tên trong run_split_screen_finetune.py -- xem
    file đó để biết giải thích đầy đủ. Ánh xạ 21 điểm tay trần sang không
    gian ảnh tay găng bằng phép biến đổi đồng dạng (xoay + co giãn qua số
    phức), mirror trước để bù đối xứng gương giữa 2 tay.
    """
    gt_wrist = gt_kpts[0]
    gl_wrist = glove_kpts[0]

    mirrored_gt = gt_kpts.copy()
    mirrored_gt[:, 0] = gt_wrist[0] - (mirrored_gt[:, 0] - gt_wrist[0])

    gt_forward = complex(*(mirrored_gt[9] - gt_wrist))
    gl_forward = complex(*(glove_kpts[9] - gl_wrist))
    if abs(gt_forward) < 1e-4:
        gt_forward = complex(1e-4, 0)
    similarity = gl_forward / gt_forward

    offsets = mirrored_gt - gt_wrist
    offsets_c = offsets[:, 0] + 1j * offsets[:, 1]
    transformed_c = offsets_c * similarity
    aligned_gt = np.stack(
        [gl_wrist[0] + transformed_c.real, gl_wrist[1] + transformed_c.imag],
        axis=1,
    )
    return aligned_gt


def is_valid_hand(kpts, scores, conf_thr=0.45):
    # Nguong cao hon nhieu so voi cac script webcam (0.20-0.26), vi o day
    # dang do tim MU MO tren toan bo khung hinh (khong co vung uu tien
    # nhu cac script webcam) -- nen deo doi hoi nghiem ngat hon de tranh
    # nhan nham day cap/do vat nen la "ban tay".
    wrist = kpts[0]
    core_scores = scores[[0, 1, 5, 9, 13, 17]]
    core_conf = float(np.mean(core_scores))
    if core_conf < conf_thr:
        return False, core_conf
    palm_len = np.linalg.norm(kpts[9] - wrist)
    if palm_len < 25.0:
        return False, core_conf
    # Kiem tra ty le hinh hoc (giong is_valid_hand trong cac script webcam)
    # -- loai nhung ket qua co hinh dang phi ly (khong giong ban tay that).
    knuckle_span = np.linalg.norm(kpts[17] - kpts[5])
    if knuckle_span < 15.0 or knuckle_span > palm_len * 2.2:
        return False, core_conf
    thumb_tip_dist = np.linalg.norm(kpts[4] - kpts[1])
    if thumb_tip_dist > palm_len * 2.0:
        return False, core_conf
    return True, core_conf


def find_glove_hand(model, frame, avoid_box, device):
    """Dò tìm tay găng trong `frame`, tránh vùng `avoid_box` (nơi tay trần
    đang ở, để không nhận nhầm lại chính nó). Thử vài vùng ứng viên khác
    nhau, chọn kết quả có độ tin cậy cao nhất.
    """
    h, w = frame.shape[:2]
    ax1, ay1, ax2, ay2 = avoid_box

    def overlap_ratio(box):
        bx1, by1, bx2, by2 = box
        ix1, iy1 = max(bx1, ax1), max(by1, ay1)
        ix2, iy2 = min(bx2, ax2), min(by2, ay2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area = max((bx2 - bx1) * (by2 - by1), 1.0)
        return inter / area

    # KHONG thu toan khung hinh -- de qua de nhan nham lai chinh tay tran
    # (da tren that, tu nhien "tu tin" hon vai gang). Chi tim trong phan
    # doi dien voi vung tay tran, dua theo tay tran dang lech ve phia nao
    # cua khung hinh.
    bare_cx = (ax1 + ax2) / 2.0
    bare_cy = (ay1 + ay2) / 2.0
    candidates = []
    if bare_cx < w / 2:
        candidates.append([w * 0.35, 0, w, h])  # tay tran ben trai -> tim ben phai
    else:
        candidates.append([0, 0, w * 0.65, h])  # tay tran ben phai -> tim ben trai
    if bare_cy < h / 2:
        candidates.append([0, h * 0.35, w, h])  # tay tran o tren -> tim ben duoi
    else:
        candidates.append([0, 0, w, h * 0.65])  # tay tran o duoi -> tim ben tren

    best = None
    best_conf = 0.0
    for c in candidates:
        bx1 = max(0, min(w - 40, int(c[0])))
        by1 = max(0, min(h - 40, int(c[1])))
        bx2 = max(bx1 + 40, min(w, int(c[2])))
        by2 = max(by1 + 40, min(h, int(c[3])))
        eval_box = np.array([[bx1, by1, bx2, by2]])
        results = inference_topdown(model, frame, bboxes=eval_box)
        if not results or not hasattr(results[0], "pred_instances"):
            continue
        inst = results[0].pred_instances
        kpts = inst.keypoints[0]
        scores = inst.keypoint_scores[0]
        valid, conf = is_valid_hand(kpts, scores)
        if not valid:
            continue
        # Bỏ qua nếu kết quả này chồng lấn quá nhiều với vùng tay trần
        # (nghĩa là model đang nhận nhầm lại chính tay trần).
        kx1, kx2 = kpts[:, 0].min(), kpts[:, 0].max()
        ky1, ky2 = kpts[:, 1].min(), kpts[:, 1].max()
        if overlap_ratio([kx1, ky1, kx2, ky2]) > 0.15:
            continue
        if conf > best_conf:
            best_conf = conf
            best = (kpts, scores)

    return best


def main():
    parser = argparse.ArgumentParser(description="Import Quest dataset -> dataset cu")
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    args = parser.parse_args()

    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    if not os.path.exists(OUT_ANNOTATIONS):
        with open(OUT_ANNOTATIONS, "w") as f:
            json.dump([], f)
    with open(OUT_ANNOTATIONS) as f:
        existing = json.load(f)
    next_id = (max((r["id"] for r in existing), default=-1) + 1)

    if not os.path.exists(QUEST_ANNOTATIONS):
        print(f"[ERROR] Khong tim thay {QUEST_ANNOTATIONS}")
        return

    with open(QUEST_ANNOTATIONS) as f:
        quest_records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(quest_records)} ban ghi tu Quest.")

    ckpt = _FINETUNED_CKPT if os.path.exists(_FINETUNED_CKPT) else _ORIGINAL_CKPT_URL
    print(f"Loading model tren {args.device.upper()}... [{'FINE-TUNED' if ckpt == _FINETUNED_CKPT else 'ORIGINAL'}]")
    model = init_model(CONFIG_FILE, ckpt, device=args.device)

    added = 0
    skipped_no_hand = 0
    skipped_bad_img = 0

    for rec in quest_records:
        img_path = os.path.join(QUEST_IMG_DIR, rec["image_file"])
        frame = cv2.imread(img_path)
        if frame is None:
            skipped_bad_img += 1
            continue

        gt_kpts = np.array(rec["bare_hand_kpts"], dtype=np.float32)

        # Vung tay tran (de tranh nhan nham) -- lay tu chinh nhan da co san,
        # them dem xung quanh.
        h, w = frame.shape[:2]
        gx1, gx2 = gt_kpts[:, 0].min(), gt_kpts[:, 0].max()
        gy1, gy2 = gt_kpts[:, 1].min(), gt_kpts[:, 1].max()
        pad = max(gx2 - gx1, gy2 - gy1) * 0.3
        avoid_box = [gx1 - pad, gy1 - pad, gx2 + pad, gy2 + pad]

        found = find_glove_hand(model, frame, avoid_box, args.device)
        if found is None:
            skipped_no_hand += 1
            continue
        glove_kpts, glove_scores = found

        aligned_gt = align_gt_to_glove(gt_kpts, glove_kpts)

        min_x = min(aligned_gt[:, 0].min(), glove_kpts[:, 0].min())
        max_x = max(aligned_gt[:, 0].max(), glove_kpts[:, 0].max())
        min_y = min(aligned_gt[:, 1].min(), glove_kpts[:, 1].min())
        max_y = max(aligned_gt[:, 1].max(), glove_kpts[:, 1].max())
        cpad = max(max_x - min_x, max_y - min_y) * 0.15

        bx1 = max(0, int(min_x - cpad))
        by1 = max(0, int(min_y - cpad))
        bx2 = min(w, int(max_x + cpad))
        by2 = min(h, int(max_y + cpad))

        crop = frame[by1:by2, bx1:bx2]
        if crop.shape[0] < 40 or crop.shape[1] < 40:
            skipped_no_hand += 1
            continue

        img_name = f"quest_import_{next_id:05d}.jpg"
        cv2.imwrite(os.path.join(OUT_IMG_DIR, img_name), crop)

        crop_kpts_glove = (glove_kpts - np.array([bx1, by1])).tolist()
        crop_kpts_gt = (aligned_gt - np.array([bx1, by1])).tolist()

        record = {
            "id": next_id,
            "image_file": img_name,
            "crop_box": [bx1, by1, bx2, by2],
            "glove_kpts_crop": crop_kpts_glove,
            "gt_kpts_crop": crop_kpts_gt,
            "timestamp": rec.get("timestamp", 0) / 1000.0,
            "source": "quest_passthrough",
        }
        existing.append(record)
        next_id += 1
        added += 1

    with open(OUT_ANNOTATIONS, "w") as f:
        json.dump(existing, f, indent=2)

    print(f"\n[DONE] Them {added} mau moi vao {OUT_ANNOTATIONS}")
    print(f"Bo qua (khong tim thay tay gang): {skipped_no_hand}")
    print(f"Bo qua (anh loi): {skipped_bad_img}")
    print(f"Tong so mau trong dataset: {len(existing)}")


if __name__ == "__main__":
    main()
