#!/usr/bin/env python3
"""
Dán nhãn THỦ CÔNG (click chuột) cho 9 điểm quan trọng nhất (cổ tay + ngón
cái + ngón trỏ) trên từng ảnh thô thu từ Quest -- dùng khi việc dò tìm tự
động không đủ tin cậy.

Chỉ dán 9/21 điểm (không phải cả 5 ngón) vì hiện tại chỉ ngón cái/trỏ
(dùng cho pinch) mới thực sự cần chính xác cao. Dữ liệu lưu ra có thêm
trường "keypoint_mask" để finetune_glove.py biết CHỈ tính loss trên 9
điểm này, không dùng nhãn giả cho 3 ngón còn lại.

Thứ tự click (hiện trên màn hình lúc dán):
  1. WRIST (cổ tay)
  2-5. THUMB: goc -> giua -> gan dau -> DAU NGON
  6-9. INDEX: goc -> giua -> gan dau -> DAU NGON

Controls:
  Click chuột trái : dat diem tiep theo
  u : Undo (bo diem vua dat)
  r : Reset lai anh nay tu dau
  s : Bo qua anh nay (khong ro tay gang / khong dan duoc)
  q : Dung lai (da luu nhung anh xong truoc do)
"""

import json
import os

import cv2
import numpy as np

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QUEST_DIR = os.path.join(_BASE_DIR, "dataset_from_quest")
QUEST_IMG_DIR = os.path.join(QUEST_DIR, "images")
QUEST_ANNOTATIONS = os.path.join(QUEST_DIR, "annotations.jsonl")

OUT_DATASET_DIR = os.path.join(_BASE_DIR, "dataset")
OUT_IMG_DIR = os.path.join(OUT_DATASET_DIR, "images")
OUT_ANNOTATIONS = os.path.join(OUT_DATASET_DIR, "annotations.json")

POINT_LABELS = [
    "WRIST (co tay)",
    "THUMB goc (metacarpal)",
    "THUMB giua",
    "THUMB gan dau",
    "THUMB DAU NGON",
    "INDEX goc (MCP)",
    "INDEX giua (PIP)",
    "INDEX gan dau (DIP)",
    "INDEX DAU NGON",
]
N_LABELED = len(POINT_LABELS)  # 9
N_TOTAL = 21

POINT_COLORS = [
    (255, 255, 255),  # wrist
    (0, 145, 255), (0, 145, 255), (0, 145, 255), (0, 145, 255),  # thumb
    (240, 200, 0), (240, 200, 0), (240, 200, 0), (240, 200, 0),  # index
]

_clicked_points = []
_display_scale = 1.0


def mouse_callback(event, x, y, flags, param):
    global _clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(_clicked_points) < N_LABELED:
            _clicked_points.append((x / _display_scale, y / _display_scale))


def render(base_img):
    disp = base_img.copy()
    for i, (x, y) in enumerate(_clicked_points):
        px, py = int(x * _display_scale), int(y * _display_scale)
        cv2.circle(disp, (px, py), 6, POINT_COLORS[i], -1, cv2.LINE_AA)
        cv2.circle(disp, (px, py), 7, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(disp, str(i), (px + 9, py - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(disp, str(i), (px + 9, py - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    # Ve duong noi thumb va index cho de nhin
    if len(_clicked_points) >= 2:
        pts = [(int(x * _display_scale), int(y * _display_scale)) for x, y in _clicked_points]
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8)]:
            if b < len(pts):
                cv2.line(disp, pts[a], pts[b], (0, 255, 0), 1, cv2.LINE_AA)

    next_label = POINT_LABELS[len(_clicked_points)] if len(_clicked_points) < N_LABELED else "DA XONG -- tu dong luu"
    h = disp.shape[0]
    cv2.rectangle(disp, (0, 0), (disp.shape[1], 40), (20, 20, 20), -1)
    cv2.putText(disp, f"Click tiep theo: {next_label}  ({len(_clicked_points)}/{N_LABELED})",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(disp, (0, h - 34), (disp.shape[1], h), (20, 20, 20), -1)
    cv2.putText(disp, "click=dat diem  u=undo  r=reset  s=bo qua anh  q=dung",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
    return disp


def main():
    global _clicked_points, _display_scale

    if not os.path.exists(QUEST_ANNOTATIONS):
        print(f"[ERROR] Khong tim thay {QUEST_ANNOTATIONS}")
        return
    with open(QUEST_ANNOTATIONS) as f:
        quest_records = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(quest_records)} anh tho tu Quest.")

    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    if not os.path.exists(OUT_ANNOTATIONS):
        with open(OUT_ANNOTATIONS, "w") as f:
            json.dump([], f)
    with open(OUT_ANNOTATIONS) as f:
        existing = json.load(f)
    next_id = max((r["id"] for r in existing), default=-1) + 1

    # Bo qua nhung anh da tung dan thu cong roi (neu chay lai tool).
    already_done = {r.get("manual_source_image") for r in existing if r.get("source") == "manual_label"}

    win_name = "Dan nhan thu cong (9 diem: co tay + cai + tro)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_callback)

    labeled_count = 0
    skipped_count = 0

    for rec in quest_records:
        if rec["image_file"] in already_done:
            continue

        img_path = os.path.join(QUEST_IMG_DIR, rec["image_file"])
        base_img = cv2.imread(img_path)
        if base_img is None:
            continue

        h, w = base_img.shape[:2]
        _display_scale = min(1.5, 1000.0 / max(w, h))
        disp_w, disp_h = int(w * _display_scale), int(h * _display_scale)
        base_disp = cv2.resize(base_img, (disp_w, disp_h))

        _clicked_points = []
        action = None
        while True:
            frame = render(base_disp)
            cv2.imshow(win_name, frame)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                action = "quit"
                break
            elif key == ord("s"):
                action = "skip"
                break
            elif key == ord("r"):
                _clicked_points = []
                continue
            elif key == ord("u"):
                if _clicked_points:
                    _clicked_points.pop()
                continue

            if len(_clicked_points) >= N_LABELED:
                action = "done"
                break

        if action == "quit":
            break
        if action == "skip":
            skipped_count += 1
            continue

        # action == "done": 9 diem da co, xay dung ban ghi.
        pts9 = np.array(_clicked_points, dtype=np.float32)  # toa do trong khong gian ANH GOC (chua scale)
        wrist = pts9[0]
        index_mcp = pts9[5]

        # Crop box phai co TI LE giong voi luc chay live (run_white_haptics_glove.py
        # dung box_side = max(span, palm_len*2.8) de bo quanh CA BAN TAY, khong
        # phai chi rieng ngon cai/tro). Neu crop o day chi bo sat 2 ngon da label
        # thi anh se bi "zoom" khac han luc inference -> model hoc sai ti le,
        # du loss training thap van cho ket qua rat te khi chay thuc te.
        min_x, max_x = pts9[:, 0].min(), pts9[:, 0].max()
        min_y, max_y = pts9[:, 1].min(), pts9[:, 1].max()
        cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
        palm_len_proxy = max(float(np.linalg.norm(index_mcp - wrist)), 1e-4)
        box_side = max(max_x - min_x, max_y - min_y, palm_len_proxy * 2.8, 140.0) * 1.3
        half = box_side / 2.0
        bx1 = max(0, int(cx - half))
        by1 = max(0, int(cy - half))
        bx2 = min(w, int(cx + half))
        by2 = min(h, int(cy + half))

        crop = base_img[by1:by2, bx1:bx2]
        if crop.shape[0] < 40 or crop.shape[1] < 40:
            print(f"[!] Vung cat qua nho cho {rec['image_file']}, bo qua.")
            skipped_count += 1
            continue

        # 21 diem: 9 diem that + 12 diem "gia" (dat trung vi tri co tay,
        # khong dung de tinh loss vi keypoint_mask=0 cho chung).
        full_kpts = np.tile(wrist, (N_TOTAL, 1)).astype(np.float32)
        full_kpts[:N_LABELED] = pts9
        crop_kpts = (full_kpts - np.array([bx1, by1])).tolist()
        mask = [1] * N_LABELED + [0] * (N_TOTAL - N_LABELED)

        img_name = f"quest_manual_{next_id:05d}.jpg"
        cv2.imwrite(os.path.join(OUT_IMG_DIR, img_name), crop)

        record = {
            "id": next_id,
            "image_file": img_name,
            "crop_box": [bx1, by1, bx2, by2],
            "gt_kpts_crop": crop_kpts,
            "keypoint_mask": mask,
            "timestamp": rec.get("timestamp", 0) / 1000.0,
            "source": "manual_label",
            "manual_source_image": rec["image_file"],
        }
        existing.append(record)
        next_id += 1
        labeled_count += 1

        with open(OUT_ANNOTATIONS, "w") as f:
            json.dump(existing, f, indent=2)

    cv2.destroyAllWindows()
    print(f"\n[DONE] Da dan {labeled_count} anh, bo qua {skipped_count}. Tong dataset: {len(existing)}")


if __name__ == "__main__":
    main()
