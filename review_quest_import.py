#!/usr/bin/env python3
"""
Xác nhận THỦ CÔNG vị trí tay găng cho dữ liệu thu từ Quest, thay vì để
máy tự động dò tìm (đã chứng minh không đủ tin cậy -- đôi khi tìm ra thứ
"có hình dạng giống tay" nhưng sai vị trí thật).

Với mỗi ảnh thô trong dataset_from_quest/, công cụ này:
1. Vẽ sẵn 21 điểm tay TRẦN (đã biết chắc chắn đúng, từ native tracking).
2. Dò vài vùng ứng viên cho tay GĂNG, vẽ ứng viên đang chọn lên ảnh.
3. Bạn tự mắt xác nhận: ứng viên đó có thực sự nằm đúng trên tay găng
   không -- bấm phím để Đồng ý (lưu vào dataset chính) / Bỏ qua ứng viên
   này thử cái khác / Bỏ hẳn mẫu này (không có tay găng rõ ràng trong ảnh).

Controls:
  k / SPACE : Đồng ý -- lưu mẫu này (cắt ảnh + quy đổi nhãn) vào dataset/
  n         : Thử ứng viên tiếp theo cho MẪU HIỆN TẠI (nếu ứng viên đang
              chọn sai vị trí, nhưng bạn nghĩ tay găng vẫn có trong ảnh)
  d         : Bỏ hẳn mẫu này (không lưu), sang mẫu tiếp theo
  q / ESC   : Dừng lại
"""

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

HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
FINGER_COLORS = [
    (0, 145, 255), (240, 200, 0), (210, 80, 255), (220, 60, 160), (50, 230, 80),
]


def align_gt_to_glove(gt_kpts, glove_kpts):
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
    return np.stack([gl_wrist[0] + transformed_c.real, gl_wrist[1] + transformed_c.imag], axis=1)


def draw_skeleton(img, kpts, color_scheme, thickness=2, radius=5):
    for bone_idx, (s, e) in enumerate(HAND_SKELETON):
        pt1 = (int(kpts[s][0]), int(kpts[s][1]))
        pt2 = (int(kpts[e][0]), int(kpts[e][1]))
        color = color_scheme if isinstance(color_scheme, tuple) else color_scheme[min(bone_idx // 4, 4)]
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
    for i, (x, y) in enumerate(kpts):
        pt = (int(x), int(y))
        color = (255, 255, 255) if i == 0 else (color_scheme if isinstance(color_scheme, tuple) else color_scheme[min((i - 1) // 4, 4)])
        cv2.circle(img, pt, radius, color, -1, cv2.LINE_AA)


def get_candidates(model, frame, avoid_box):
    """Tra ve TAT CA cac ung vien tay gang tim duoc (khong chi 1 cai tot
    nhat), sap xep theo do tin cay giam dan, de nguoi dung tu chon."""
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

    bare_cx = (ax1 + ax2) / 2.0
    bare_cy = (ay1 + ay2) / 2.0
    search_boxes = []
    if bare_cx < w / 2:
        search_boxes.append([w * 0.30, 0, w, h])
    else:
        search_boxes.append([0, 0, w * 0.70, h])
    if bare_cy < h / 2:
        search_boxes.append([0, h * 0.30, w, h])
    else:
        search_boxes.append([0, 0, w, h * 0.70])
    search_boxes.append([0, 0, w, h])  # toan khung hinh, thu cuoi cung

    results_list = []
    for c in search_boxes:
        bx1 = max(0, min(w - 40, int(c[0])))
        by1 = max(0, min(h - 40, int(c[1])))
        bx2 = max(bx1 + 40, min(w, int(c[2])))
        by2 = max(by1 + 40, min(h, int(c[3])))
        eval_box = np.array([[bx1, by1, bx2, by2]])
        res = inference_topdown(model, frame, bboxes=eval_box)
        if not res or not hasattr(res[0], "pred_instances"):
            continue
        inst = res[0].pred_instances
        kpts = inst.keypoints[0]
        scores = inst.keypoint_scores[0]
        core_conf = float(np.mean(scores[[0, 1, 5, 9, 13, 17]]))
        kx1, kx2 = kpts[:, 0].min(), kpts[:, 0].max()
        ky1, ky2 = kpts[:, 1].min(), kpts[:, 1].max()
        ov = overlap_ratio([kx1, ky1, kx2, ky2])
        results_list.append((core_conf, ov, kpts, scores))

    # Sap xep: uu tien do tin cay cao VA it chong lan voi tay tran.
    results_list.sort(key=lambda r: (r[1] > 0.15, -r[0]))
    return [(k, s) for _, _, k, s in results_list]


def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    if not os.path.exists(OUT_ANNOTATIONS):
        with open(OUT_ANNOTATIONS, "w") as f:
            json.dump([], f)
    with open(OUT_ANNOTATIONS) as f:
        existing = json.load(f)
    next_id = max((r["id"] for r in existing), default=-1) + 1

    if not os.path.exists(QUEST_ANNOTATIONS):
        print(f"[ERROR] Khong tim thay {QUEST_ANNOTATIONS}")
        return
    with open(QUEST_ANNOTATIONS) as f:
        quest_records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(quest_records)} anh tho tu Quest.")

    ckpt = _FINETUNED_CKPT if os.path.exists(_FINETUNED_CKPT) else _ORIGINAL_CKPT_URL
    print(f"Loading model tren {device.upper()}...")
    model = init_model(CONFIG_FILE, ckpt, device=device)

    win_name = "Xac nhan tay gang (Quest import)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    accepted = 0
    skipped = 0

    for rec_idx, rec in enumerate(quest_records):
        img_path = os.path.join(QUEST_IMG_DIR, rec["image_file"])
        frame = cv2.imread(img_path)
        if frame is None:
            skipped += 1
            continue

        gt_kpts = np.array(rec["bare_hand_kpts"], dtype=np.float32)
        h, w = frame.shape[:2]
        gx1, gx2 = gt_kpts[:, 0].min(), gt_kpts[:, 0].max()
        gy1, gy2 = gt_kpts[:, 1].min(), gt_kpts[:, 1].max()
        pad = max(gx2 - gx1, gy2 - gy1) * 0.3
        avoid_box = [gx1 - pad, gy1 - pad, gx2 + pad, gy2 + pad]

        candidates = get_candidates(model, frame, avoid_box)

        cand_idx = 0
        while True:
            disp = frame.copy()
            # Ve tay tran (mau xam nhat, chi de tham khao vi tri).
            draw_skeleton(disp, gt_kpts, (140, 140, 140), thickness=1, radius=3)

            if candidates:
                cand_idx = cand_idx % len(candidates)
                glove_kpts, glove_scores = candidates[cand_idx]
                draw_skeleton(disp, glove_kpts, FINGER_COLORS, thickness=3, radius=6)
                status = f"Ung vien {cand_idx+1}/{len(candidates)}"
            else:
                status = "KHONG TIM THAY ung vien nao -- bam 'd' de bo qua"

            header = f"[{rec_idx+1}/{len(quest_records)}] {rec['image_file']}  Accepted={accepted} Skipped={skipped}  {status}"
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 34), (20, 20, 20), -1)
            cv2.putText(disp, header, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(disp, "k/SPACE=dong y  n=doi ung vien  d=bo qua mau nay  q=dung",
                        (8, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)

            cv2.imshow(win_name, disp)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("q"), 27):
                print(f"\n[DUNG] Accepted={accepted} Skipped={skipped}")
                with open(OUT_ANNOTATIONS, "w") as f:
                    json.dump(existing, f, indent=2)
                cv2.destroyAllWindows()
                return

            elif key == ord("n"):
                if candidates:
                    cand_idx += 1
                continue

            elif key == ord("d"):
                skipped += 1
                break

            elif key in (ord("k"), 32):
                if not candidates:
                    print("[!] Khong co ung vien nao de dong y, tu dong bo qua.")
                    skipped += 1
                    break
                glove_kpts, glove_scores = candidates[cand_idx]
                aligned_gt = align_gt_to_glove(gt_kpts, glove_kpts)

                # Tinh khung cat CHI dua tren aligned_gt (nhan dang tin cay,
                # lay tu tay tran) -- KHONG dua vao 21 diem tho cua glove_kpts,
                # vi ung vien co the doan sai giua/ap ut/ut (khong co marker
                # rieng) du cho co/goc ngon giua (dung de neo) van dung.
                # Dung glove_kpts CHI cho 2 diem neo (co tay + goc ngon giua)
                # de dam bao khung it nhat bao trum toi vung do.
                anchor_pts = glove_kpts[[0, 9]]
                min_x = min(aligned_gt[:, 0].min(), anchor_pts[:, 0].min())
                max_x = max(aligned_gt[:, 0].max(), anchor_pts[:, 0].max())
                min_y = min(aligned_gt[:, 1].min(), anchor_pts[:, 1].min())
                max_y = max(aligned_gt[:, 1].max(), anchor_pts[:, 1].max())
                cpad = max(max_x - min_x, max_y - min_y) * 0.15
                bx1 = max(0, int(min_x - cpad))
                by1 = max(0, int(min_y - cpad))
                bx2 = min(w, int(max_x + cpad))
                by2 = min(h, int(max_y + cpad))

                crop = frame[by1:by2, bx1:bx2]
                if crop.shape[0] < 40 or crop.shape[1] < 40:
                    print("[!] Vung cat qua nho, tu dong bo qua.")
                    skipped += 1
                    break

                img_name = f"quest_confirmed_{next_id:05d}.jpg"
                cv2.imwrite(os.path.join(OUT_IMG_DIR, img_name), crop)
                record = {
                    "id": next_id,
                    "image_file": img_name,
                    "crop_box": [bx1, by1, bx2, by2],
                    "glove_kpts_crop": (glove_kpts - np.array([bx1, by1])).tolist(),
                    "gt_kpts_crop": (aligned_gt - np.array([bx1, by1])).tolist(),
                    "timestamp": rec.get("timestamp", 0) / 1000.0,
                    "source": "quest_passthrough_confirmed",
                }
                existing.append(record)
                next_id += 1
                accepted += 1
                break

    with open(OUT_ANNOTATIONS, "w") as f:
        json.dump(existing, f, indent=2)
    cv2.destroyAllWindows()
    print(f"\n[DONE] Accepted={accepted} Skipped={skipped}. Tong dataset: {len(existing)}")


if __name__ == "__main__":
    main()
