#!/usr/bin/env python3
"""
Dataset Review & Cleanup Tool.

Shows each recorded training pair with the SAVED GROUND-TRUTH LABEL
(gt_kpts_crop, i.e. the thing the model will actually be trained to
predict) drawn on top of the glove crop image, so you can visually judge
whether the label lines up with the real fingers -- not just eyeball the
raw image.

By default only the label is shown (colored, filled dots) -- that's the
only thing that matters for judging a sample. Press 'v' to additionally
overlay the glove model's OWN raw prediction on that frame (gray, hollow
dots) for comparison, if you want to see "what it guessed at the time"
vs "what we're teaching it" -- purely informational, not a sign of
correctness either way.

Controls:
  SPACE / k : keep this sample, go to next
  d / BACKSPACE : mark this sample for deletion, go to next
  b : go back one sample (undo last keep/delete decision)
  v : toggle the gray comparison overlay on/off (off by default)
  q / ESC : stop reviewing now and apply deletions so far
"""

import json
import os

import cv2
import numpy as np

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(_BASE_DIR, "dataset")
DATASET_IMG_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_FILE = os.path.join(DATASET_DIR, "annotations.json")

HAND_SKELETON = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
]

# Thumb: Orange, Index: Sky Blue, Middle: Pink, Ring: Purple, Pinky: Green
FINGER_COLORS = [
    (0, 145, 255),
    (240, 200, 0),
    (210, 80, 255),
    (220, 60, 160),
    (50, 230, 80),
]

MIN_DISPLAY_SIZE = 480


def draw_skeleton(img, kpts, colors, thickness=2, point_radius=4, filled=True):
    for bone_idx, (start, end) in enumerate(HAND_SKELETON):
        pt1 = (int(kpts[start][0]), int(kpts[start][1]))
        pt2 = (int(kpts[end][0]), int(kpts[end][1]))
        color = colors if isinstance(colors, tuple) else colors[min(bone_idx // 4, 4)]
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
    for i, (x, y) in enumerate(kpts):
        pt = (int(x), int(y))
        if isinstance(colors, tuple):
            color = colors
        else:
            color = (255, 255, 255) if i == 0 else colors[min((i - 1) // 4, 4)]
        cv2.circle(img, pt, point_radius, color, -1 if filled else 1, cv2.LINE_AA)


def load_and_scale(img_path, scale_kpts_list):
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    scale = max(1.0, MIN_DISPLAY_SIZE / max(h, w))
    if scale > 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)
    scaled = []
    for kpts in scale_kpts_list:
        k = np.array(kpts, dtype=np.float32) * scale
        scaled.append(k)
    return img, scaled


def main():
    if not os.path.exists(ANNOTATIONS_FILE):
        print(f"[ERROR] {ANNOTATIONS_FILE} not found.")
        return

    with open(ANNOTATIONS_FILE, "r") as f:
        records = json.load(f)

    if not records:
        print("[INFO] No records to review.")
        return

    print(f"Loaded {len(records)} records.")
    print("Controls: SPACE/k = keep | d/BACKSPACE = delete | b = back | q/ESC = stop & apply\n")

    win_name = "Dataset Review (label = bold, live glove guess = thin)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    decisions = {}  # id -> "keep" | "delete"
    idx = 0
    n = len(records)
    show_compare = False  # default: only show the bold label, no comparison clutter

    while 0 <= idx < n:
        rec = records[idx]
        img_path = os.path.join(DATASET_IMG_DIR, rec["image_file"])
        gt_kpts = rec.get("gt_kpts_crop")
        gl_kpts = rec.get("glove_kpts_crop")

        if gt_kpts is None:
            print(f"[SKIP] record id {rec.get('id')} has no gt_kpts_crop, deleting.")
            decisions[rec["id"]] = "delete"
            idx += 1
            continue

        scale_list = [gt_kpts] + ([gl_kpts] if gl_kpts is not None else [])
        img, scaled = load_and_scale(img_path, scale_list)
        if img is None:
            print(f"[SKIP] could not read {img_path}, deleting.")
            decisions[rec["id"]] = "delete"
            idx += 1
            continue

        disp = img.copy()
        if show_compare and gl_kpts is not None:
            # Live model guess at record time: gray, hollow dots. Off by default.
            draw_skeleton(disp, scaled[1], (150, 150, 150), thickness=1, point_radius=5, filled=False)
        # Saved training label: full color, filled dots -- this is what matters.
        draw_skeleton(disp, scaled[0], FINGER_COLORS, thickness=3, point_radius=6, filled=True)

        kept = sum(1 for v in decisions.values() if v == "keep")
        deleted = sum(1 for v in decisions.values() if v == "delete")
        status = decisions.get(rec["id"], "")
        cmp_hint = "compare:ON ('v' off)" if show_compare else "compare:off ('v' on)"
        header = f"[{idx+1}/{n}] id={rec['id']}  kept={kept} deleted={deleted}  {status}  {cmp_hint}"
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (20, 20, 20), -1)
        cv2.putText(disp, header, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(win_name, disp)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("v"):
            show_compare = not show_compare
            continue
        elif key in (ord("k"), 32):  # space
            decisions[rec["id"]] = "keep"
            idx += 1
        elif key in (ord("d"), 8):  # backspace
            decisions[rec["id"]] = "delete"
            idx += 1
        elif key == ord("b"):
            idx = max(0, idx - 1)
        # any other key: redisplay same record

    cv2.destroyAllWindows()

    to_delete_ids = {rid for rid, d in decisions.items() if d == "delete"}
    if not to_delete_ids:
        print("\n[INFO] No records marked for deletion. Nothing changed.")
        return

    kept_records = [r for r in records if r["id"] not in to_delete_ids]
    removed_records = [r for r in records if r["id"] in to_delete_ids]

    for r in removed_records:
        img_path = os.path.join(DATASET_IMG_DIR, r["image_file"])
        if os.path.exists(img_path):
            os.remove(img_path)

    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(kept_records, f, indent=2)

    print(f"\n[DONE] Removed {len(removed_records)} bad samples. {len(kept_records)} remain in the dataset.")


if __name__ == "__main__":
    main()
