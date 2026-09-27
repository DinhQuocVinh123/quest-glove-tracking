#!/usr/bin/env python3
"""
Gan nhan 21 diem ban tay tren cac khung da ghi bang --record / --protocol.

Moi khung da DIEN SAN 21 diem do model doan (luu trong keypoints.jsonl luc
ghi) -- ban chi KEO SUA nhung diem sai. Moi diem co 3 trang thai: NHIN THAY,
UOC LUONG (bi che, vi tri doan theo dang tay -- vd phim k cho nam tay) va
KHONG BIET (bo qua khi huan luyen). Xem SEEN / ESTIMATED / UNKNOWN.

Cong cu tu chon khung de gan: UU TIEN khung model bi MAT (dung loai can
hoc), them mot phan khung model da bam (de model khong "quen" cai da biet),
va bo qua cac khung qua giong nhau (cach nhau < --min-gap giay).

Cach dung:
  python label_glove_frames.py recordings/20260926_125425_protocol
  python label_glove_frames.py recordings/*_protocol_fist --poses nam_mu,nam_long
  python label_glove_frames.py --review-auto     # chi sua cac khung auto_label_glove.py chua tu gan duoc
  python label_glove_frames.py --review-claude   # sua nhan 9 diem Claude da dat (dien san, chi keo diem sai)

Dieu khien:
  Chuot trai (giu + keo) : di chuyen diem gan nhat
  Chuot phai             : bat/tat "KHONG BIET" cho diem gan nhat (bo qua khi huan luyen)
  b roi keo chuot        : ve khung quanh ban tay -> model doan lai (khi dien san qua te)
  Enter / Space          : LUU va sang khung sau
  n                      : BO khung nay (khong dung duoc: mo, khong thay tay...)
  r                      : ve lai diem dien san ban dau
  z                      : hoan tac buoc vua sua
  a                      : quay lai khung truoc
  f                      : xem ca anh / phong to quanh ban tay
  k                      : NAM TAY -- dat xong CO TAY + 4 khop dot tay (diem "goc") roi bam k:
                           tu xep 3 khop con lai cua moi ngon theo dang nam tay, danh dau
                           UOC LUONG (bi che nhung van cho model hoc, trong so thap)
  h                      : bat/tat bang huong dan gan nhan
  q / Esc                : thoat (nhung khung da luu van con)

Quy tac gan nhan NAM TAY nhin tu mu ban tay: chi dat diem NHIN THAY RO --
  hang 1 (4 u xuong lon, cho go cua) = 4 diem "goc"; hang 2 (khop gap thu hai,
  mat phia truoc nam tay) = 4 diem "giua" neu thay; co tay; ngon cai (thuong
  thay ro). Phan giau trong long tay: bam k (UOC LUONG) thay vi bo trong.

Nhan duoc luu vao labels/glove_labels.jsonl (moi dong 1 khung). Chay lai se
tiep tuc tu cho dung, khong hoi lai khung da gan.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_FILE = os.path.join(BASE_DIR, "labels", "glove_labels.jsonl")
WIN = "Gan nhan gang tay"
VIEW_W, VIEW_H = 1280, 900
PAD = 300          # vien xam quanh anh: diem (vd co tay) duoc phep nam ngoai anh
PICK_RADIUS = 22   # pixel tren man hinh

POINT_NAMES = [
    "CO TAY",
    "CAI goc", "CAI giua", "CAI gan dau", "CAI DAU",
    "TRO goc", "TRO giua", "TRO gan dau", "TRO DAU",
    "GIUA goc", "GIUA giua", "GIUA gan dau", "GIUA DAU",
    "AP UT goc", "AP UT giua", "AP UT gan dau", "AP UT DAU",
    "UT goc", "UT giua", "UT gan dau", "UT DAU",
]
FINGER_COLORS = [(0, 140, 255), (255, 200, 0), (255, 0, 255), (160, 60, 255), (0, 220, 0)]  # BGR: cai, tro, giua, ap ut, ut
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def finger_of(i):
    return 0 if i == 0 else (i - 1) // 4


HELP_LINES = [
    "HUONG DAN GAN NHAN (bam h de tat)",
    "",
    "Cham DAC = nhin thay | vong co cham = UOC LUONG (bi che, vi tri doan) | vong co x = KHONG BIET",
    "Keo diem nao thi diem do thanh NHIN THAY. Chuot phai: bat/tat KHONG BIET (bo qua khi huan luyen).",
    "",
    "NAM TAY nhin tu MU ban tay:",
    "  1. CO TAY: cho noi ban tay voi cang tay.",
    "  2. Hang u xuong LON (cho go cua) = 4 diem 'goc' cua tro/giua/ap ut/ut.",
    "  3. Bam k: tu xep phan giau trong long tay theo dang nam tay (UOC LUONG).",
    "     Can thiet: de model hoc duoc 'nam tay thi ngon cuon lai', khong doan ngon duoi thang.",
    "  4. Hang khop thu hai (mat truoc nam tay) = 4 diem 'giua': thay ro thi keo cho khop.",
    "  5. Ngon CAI thuong vat ngang phia truoc va thay ro: dat du 4 diem.",
    "",
    "XOE TAY / PINCH: keo tung diem ve dung khop; dau ngon co chop mau de de nhan.",
    "Khung mo, khong thay tay, khong chac chan -> n (bo qua).",
]


# Trang thai tung diem (luu trong truong "visible"):
SEEN, UNKNOWN, ESTIMATED = 1, 0, 2
# SEEN      : nhin thay, nguoi gan tu dat          -> huan luyen trong so day du
# ESTIMATED : bi che, vi tri UOC LUONG (phim k)    -> huan luyen trong so thap
# UNKNOWN   : khong biet (chuot phai)              -> bo qua khi huan luyen
# Vi sao can ESTIMATED: neu bo qua het khop bi che, model khong bao gio hoc
# duoc "nam tay thi dau ngon cuon trong long ban tay" -- se tiep tuc doan ngon
# duoi thang, va tay ao van khong nam lai.


def fist_fill(pts, vis):
    """Tu xep 4 ngon theo dang NAM TAY tu CO TAY + 4 khop dot tay (diem goc).

    Nhin tu mu ban tay (ngon cuon ra xa mat): dot 1 chia ra xa nen tren anh khop
    giua (PIP) nam ngay tren hang khop dot tay; khop gan dau ngon + dau ngon gap
    nguoc vao long ban tay, nam PHIA SAU mu ban tay (giua khop dot tay va co tay).
    Ca 3 diem nay danh dau UOC LUONG -- keo diem nao thay ro thi no thanh
    NHIN THAY. Ngon cai giu nguyen (thuong thay ro, nguoi gan tu dat)."""
    pts, vis = pts.copy(), vis.copy()
    wrist = pts[0]
    for mcp in (5, 9, 13, 17):
        d = pts[mcp] - wrist
        length = max(float(np.linalg.norm(d)), 1e-3)
        u = d / length
        pts[mcp + 1] = pts[mcp] + u * 0.15 * length   # khop giua: ngay tren hang khop dot tay
        pts[mcp + 2] = pts[mcp] - u * 0.08 * length   # khop gan dau: gap nguoc ra sau
        pts[mcp + 3] = pts[mcp] - u * 0.32 * length   # dau ngon: trong long ban tay
        vis[mcp + 1:mcp + 4] = ESTIMATED
    return pts, vis


# --- Chon khung ------------------------------------------------------------

def load_recording(rec_dir):
    rows = list(csv.DictReader(open(os.path.join(rec_dir, "log.csv"), encoding="utf-8")))
    kp_path = os.path.join(rec_dir, "keypoints.jsonl")
    kps = [json.loads(line) for line in open(kp_path, encoding="utf-8")] if os.path.exists(kp_path) else [None] * len(rows)
    return rows, kps


def select_frames(rec_dirs, poses, total, lost_share, min_gap, done):
    lost, kept = [], []
    for rec_dir in rec_dirs:
        name = os.path.basename(os.path.normpath(rec_dir))
        rows, kps = load_recording(rec_dir)
        last_t = {}
        for r, k in zip(rows, kps):
            pose = r.get("pose", "free")
            if poses and pose not in poses:
                continue
            key = (name, int(r["frame"]))
            if key in done:
                continue
            is_lost = r["found"] == "0"
            cat = (pose, is_lost)
            t = float(r["t"])
            if t - last_t.get(cat, -1e9) < min_gap:  # qua giong khung vua chon
                continue
            last_t[cat] = t
            item = {"rec_dir": rec_dir, "recording": name, "frame": int(r["frame"]), "pose": pose,
                    "found": not is_lost, "prefill": (k or {}).get("kpts")}
            (lost if is_lost else kept).append(item)

    n_lost = min(len(lost), int(round(total * lost_share)))
    n_kept = min(len(kept), total - n_lost)
    rng = np.random.default_rng(0)
    pick = [lost[i] for i in sorted(rng.choice(len(lost), n_lost, replace=False))] if n_lost else []
    pick += [kept[i] for i in sorted(rng.choice(len(kept), n_kept, replace=False))] if n_kept else []
    rng.shuffle(pick)  # tron de khong phai gan lien 1 kieu kho
    return pick, len(lost), len(kept)


def load_done():
    done = {}
    if os.path.exists(LABEL_FILE):
        for line in open(LABEL_FILE, encoding="utf-8"):
            rec = json.loads(line)
            done[(rec["recording"], rec["frame"])] = rec
    return done


def template_hand(cx, cy, size):
    """Ban tay mau (xoe, mu ban tay) -- dung khi khong co diem dien san."""
    unit = np.array([[0, 0], [-.45, -.25], [-.7, -.5], [-.85, -.75], [-.95, -1.0],
                     [-.3, -1.0], [-.35, -1.45], [-.38, -1.75], [-.4, -2.0],
                     [0, -1.05], [0, -1.55], [0, -1.9], [0, -2.15],
                     [.28, -1.0], [.32, -1.45], [.35, -1.75], [.37, -1.95],
                     [.52, -.9], [.6, -1.25], [.65, -1.5], [.68, -1.7]])
    return unit * size * 0.5 + np.array([cx, cy])


# --- Giao dien -------------------------------------------------------------

class Labeler:
    def __init__(self, items, done):
        self.items, self.done = items, done
        self.idx = 0
        self.model = None
        self.full_view = False
        self.box_mode, self.box_start, self.box_end = False, None, None
        self.drag, self.hover = None, None
        self.show_help = not any(not v.get("skip") for v in done.values())  # lan dau: hien huong dan
        self.load(0)

    # Nap khung i: anh + diem (nhan da luu > dien san > ban tay mau)
    def load(self, i):
        self.idx = i
        it = self.items[i]
        self.img = cv2.imread(os.path.join(it["rec_dir"], f"{it['frame']:05d}.jpg"))
        self.padded = cv2.copyMakeBorder(self.img, PAD, PAD, PAD, PAD, cv2.BORDER_CONSTANT, value=(90, 90, 90))
        saved = self.done.get((it["recording"], it["frame"]))
        h, w = self.img.shape[:2]
        if saved and not saved.get("skip"):
            pts, vis = np.array(saved["kpts"], float), np.array(saved["visible"], int)
        elif it["prefill"] is not None and self._sane(np.array(it["prefill"], float)):
            pts, vis = np.array(it["prefill"], float), np.full(21, SEEN)
        else:
            pts, vis = template_hand(w * 0.5, h * 0.75, min(w, h) * 0.3), np.full(21, SEEN)
        self.prefill = pts.copy()
        self.pts, self.vis = pts, vis
        self.history = []
        self._fit_view()

    @staticmethod
    def _sane(p):
        """Diem dien san 'don cum' (long ban tay vai pixel) thi vo dung."""
        return np.linalg.norm(p[9] - p[0]) > 25

    def _fit_view(self):
        h, w = self.img.shape[:2]
        if self.full_view:
            x0, y0, x1, y1 = -PAD * 0.3, -PAD * 0.3, w + PAD * 0.3, h + PAD * 0.3
        else:
            p = self.pts
            cx, cy = (p[:, 0].min() + p[:, 0].max()) / 2, (p[:, 1].min() + p[:, 1].max()) / 2
            side = max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) * 1.7
            side = max(side, 320)
            half_w, half_h = side * 0.5 * VIEW_W / VIEW_H, side * 0.5
            x0, y0, x1, y1 = cx - half_w, cy - half_h, cx + half_w, cy + half_h
        self.view = (x0, y0, x1, y1)
        self.scale = min(VIEW_W / (x1 - x0), VIEW_H / (y1 - y0))

    def to_screen(self, p):
        x0, y0, _, _ = self.view
        return (p[..., 0] - x0) * self.scale, (p[..., 1] - y0) * self.scale

    def to_image(self, sx, sy):
        x0, y0, _, _ = self.view
        return np.array([sx / self.scale + x0, sy / self.scale + y0])

    def nearest(self, sx, sy):
        xs, ys = self.to_screen(self.pts)
        d = np.hypot(xs - sx, ys - sy)
        i = int(np.argmin(d))
        return i if d[i] <= PICK_RADIUS else None

    def on_mouse(self, event, sx, sy, flags, _):
        if self.box_mode:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.box_start = self.box_end = (sx, sy)
            elif event == cv2.EVENT_MOUSEMOVE and self.box_start:
                self.box_end = (sx, sy)
            elif event == cv2.EVENT_LBUTTONUP and self.box_start:
                self.box_end = (sx, sy)
                self.repredict()
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = self.nearest(sx, sy)
            if self.drag is not None:
                self.history.append((self.pts.copy(), self.vis.copy()))
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag is not None:
                self.pts[self.drag] = self.to_image(sx, sy)
                self.vis[self.drag] = SEEN  # nguoi gan tu dat diem nay -> nhin thay
            self.hover = self.nearest(sx, sy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            i = self.nearest(sx, sy)
            if i is not None:
                self.history.append((self.pts.copy(), self.vis.copy()))
                self.vis[i] = SEEN if self.vis[i] == UNKNOWN else UNKNOWN

    def repredict(self):
        """Chay lai model voi khung nguoi dung ve (thuong tot hon khi dien san bi don cum)."""
        a, b = self.to_image(*self.box_start), self.to_image(*self.box_end)
        x1, y1 = np.minimum(a, b)
        x2, y2 = np.maximum(a, b)
        self.box_mode, self.box_start, self.box_end = False, None, None
        if x2 - x1 < 20 or y2 - y1 < 20:
            return
        if self.model is None:
            print("Dang nap model (lan dau mat vai giay)...")
            import importlib.util
            spec = importlib.util.spec_from_file_location("gq", os.path.join(BASE_DIR, "run_glove_quest_stream.py"))
            gq = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(gq)
            from mmpose.apis import inference_topdown, init_model
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = (init_model(gq.CONFIG_FILE, gq._ORIGINAL_CKPT_URL, device=device), inference_topdown)
        model, infer = self.model
        box = np.array([[x1 + PAD, y1 + PAD, x2 + PAD, y2 + PAD]])
        out = infer(model, self.padded, bboxes=box)[0].pred_instances
        self.history.append((self.pts.copy(), self.vis.copy()))
        self.pts = out.keypoints[0].astype(float) - PAD
        self.vis = np.full(21, SEEN)

    # --- ve ---
    def render(self):
        x0, y0, x1, y1 = self.view
        M = np.array([[self.scale, 0, -(x0 + PAD) * self.scale], [0, self.scale, -(y0 + PAD) * self.scale]])
        canvas = cv2.warpAffine(self.padded, M, (VIEW_W, VIEW_H), borderValue=(60, 60, 60))
        xs, ys = self.to_screen(self.pts)
        for a, b in BONES:
            color = FINGER_COLORS[finger_of(b)]
            if not (self.vis[a] == SEEN and self.vis[b] == SEEN):
                color = tuple(int(c * 0.4) for c in color)
            cv2.line(canvas, (int(xs[a]), int(ys[a])), (int(xs[b]), int(ys[b])), color, 2, cv2.LINE_AA)
        for i in range(21):
            c = (255, 255, 255) if i == 0 else FINGER_COLORS[finger_of(i)]
            center = (int(xs[i]), int(ys[i]))
            if self.vis[i] == SEEN:
                cv2.circle(canvas, center, 7, c, -1, cv2.LINE_AA)
                cv2.circle(canvas, center, 8, (0, 0, 0), 1, cv2.LINE_AA)
            elif self.vis[i] == ESTIMATED:  # uoc luong: vong rong + cham giua
                cv2.circle(canvas, center, 8, c, 2, cv2.LINE_AA)
                cv2.circle(canvas, center, 2, c, -1, cv2.LINE_AA)
            else:  # KHONG BIET: vong rong + dau x
                cv2.circle(canvas, center, 8, c, 2, cv2.LINE_AA)
                cv2.line(canvas, (center[0] - 5, center[1] - 5), (center[0] + 5, center[1] + 5), c, 2)
            if i == self.hover or i == self.drag:
                cv2.circle(canvas, center, 13, (0, 255, 255), 2, cv2.LINE_AA)
        if self.box_mode and self.box_start and self.box_end:
            cv2.rectangle(canvas, self.box_start, self.box_end, (0, 255, 255), 2)

        it = self.items[self.idx]
        labeled = sum(1 for k, v in self.done.items() if not v.get("skip"))
        top = (f"{self.idx + 1}/{len(self.items)}  {it['recording']} #{it['frame']}  dong tac: {it['pose']}  "
               f"({'model DA bam' if it['found'] else 'model bi MAT'})  | tong da gan: {labeled}")
        cv2.rectangle(canvas, (0, 0), (VIEW_W, 58), (0, 0, 0), -1)
        cv2.putText(canvas, top, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        help_txt = ("BAM b ROI KEO CHUOT QUANH BAN TAY" if self.box_mode else
                    "keo: sua | chuot phai: KHONG BIET | k: nam tay | b: doan lai | Enter: luu | n: bo | r/z: ve lai/hoan tac | a: lui | f: zoom | h: huong dan | q: thoat")
        cv2.putText(canvas, help_txt, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        name_i = self.drag if self.drag is not None else self.hover
        if name_i is not None:
            state = {SEEN: "", ESTIMATED: " (UOC LUONG)", UNKNOWN: " (KHONG BIET)"}[int(self.vis[name_i])]
            cv2.putText(canvas, POINT_NAMES[name_i] + state, (10, VIEW_H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)
        if self.show_help:
            box_h = 30 + 26 * len(HELP_LINES)
            overlay = canvas.copy()
            cv2.rectangle(overlay, (60, 90), (VIEW_W - 60, 90 + box_h), (0, 0, 0), -1)
            canvas = cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0)
            for j, line in enumerate(HELP_LINES):
                cv2.putText(canvas, line, (84, 120 + 26 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                            (0, 255, 255) if j == 0 else (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    # --- luu ---
    def save(self, skip=False):
        it = self.items[self.idx]
        moved = int(np.sum(np.linalg.norm(self.pts - self.prefill, axis=1) > 3))
        rec = {
            "recording": it["recording"], "frame": it["frame"],
            "image": os.path.relpath(os.path.join(it["rec_dir"], f"{it['frame']:05d}.jpg"), BASE_DIR),
            "pose": it["pose"], "model_found": it["found"], "skip": skip,
            "kpts": np.round(self.pts, 1).tolist(), "visible": [int(v) for v in self.vis],
            "points_moved": moved, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        os.makedirs(os.path.dirname(LABEL_FILE), exist_ok=True)
        with open(LABEL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self.done[(it["recording"], it["frame"])] = rec

    def run(self):
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WIN, self.on_mouse)
        while True:
            cv2.imshow(WIN, self.render())
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (13, 32, ord("n")):
                self.save(skip=key == ord("n"))
                if self.idx + 1 >= len(self.items):
                    print("Da gan het cac khung da chon.")
                    break
                self.load(self.idx + 1)
            elif key == ord("a") and self.idx > 0:
                self.load(self.idx - 1)
            elif key == ord("r"):
                self.history.append((self.pts.copy(), self.vis.copy()))
                self.pts, self.vis = self.prefill.copy(), np.full(21, SEEN)
            elif key == ord("z") and self.history:
                self.pts, self.vis = self.history.pop()
            elif key == ord("b"):
                self.box_mode = True
            elif key == ord("k"):
                self.history.append((self.pts.copy(), self.vis.copy()))
                self.pts, self.vis = fist_fill(self.pts, self.vis)
            elif key == ord("h"):
                self.show_help = not self.show_help
            elif key == ord("f"):
                self.full_view = not self.full_view
                self._fit_view()
        cv2.destroyAllWindows()


AUTO_FILE = os.path.join(BASE_DIR, "labels", "auto_labels.jsonl")


def review_auto(args):
    """Gan cac khung ma auto_label_glove.py khong tu gan duoc / Claude soat thay sai.
    Theo thu tu thoi gian (khung ke nhau giong nhau -> sua nhanh hon)."""
    if not os.path.exists(AUTO_FILE):
        sys.exit("Chua co labels/auto_labels.jsonl -- chay auto_label_glove.py truoc.")
    done = load_done()
    items = []
    for line in open(AUTO_FILE, encoding="utf-8"):
        r = json.loads(line)
        if r["status"] != "review" or (r["recording"], r["frame"]) in done:
            continue
        rec_dir = r["rec_dir"] if os.path.isabs(r["rec_dir"]) else os.path.join(BASE_DIR, r["rec_dir"])
        items.append({"rec_dir": rec_dir, "recording": r["recording"], "frame": r["frame"], "pose": "free",
                      "found": False, "prefill": r.get("kpts")})
    items.sort(key=lambda it: (it["recording"], it["frame"]))
    items = items[:args.count]
    print(f"Con {len(items)} khung can xem lai (da gan: {sum(1 for v in done.values() if not v.get('skip'))}).")
    if not items:
        sys.exit("Khong con khung nao can xem lai.")
    Labeler(items, done).run()


def review_claude(args):
    """Cac khung co nhan MOI NHAT la cua Claude -> dien san dung nhan do (ke ca trang thai
    tung diem), nguoi chi keo diem sai roi Enter. Luu lai se ghi de (thanh nhan cua nguoi)."""
    done = load_done()
    items = []
    for (rec, frame), r in done.items():
        if r.get("source") != "claude" or r.get("skip"):
            continue
        items.append({"rec_dir": os.path.join(BASE_DIR, "recordings", rec), "recording": rec, "frame": frame,
                      "pose": "free", "found": False, "prefill": r.get("kpts")})
    items.sort(key=lambda it: (it["recording"], it["frame"]))
    items = items[:args.count]
    print(f"Con {len(items)} khung nhan cua Claude chua duoc ban sua. Enter = dong y/luu, n = bo khung.")
    if not items:
        sys.exit("Khong con khung nao.")
    Labeler(items, done).run()


def main():
    ap = argparse.ArgumentParser(description="Gan nhan 21 diem tay deo gang tren cac khung da ghi.")
    ap.add_argument("recordings", nargs="*", help="Thu muc ban ghi (co the dung *, vd recordings/*_protocol*)")
    ap.add_argument("--review-claude", action="store_true",
                    help="Xem/sua cac khung Claude da gan 9 diem (source claude). Luu lai = thanh nhan cua ban.")
    ap.add_argument("--review-auto", action="store_true",
                    help="Chi gan cac khung auto_label_glove.py danh dau XEM LAI (labels/auto_labels.jsonl), da dien san de xuat.")
    ap.add_argument("--poses", default="", help="Chi lay cac dong tac nay, cach nhau dau phay (vd nam,nam_xoe). "
                                                 "De trong = moi dong tac.")
    ap.add_argument("--count", type=int, default=150, help="So khung can gan lan nay (mac dinh 150).")
    ap.add_argument("--lost-share", type=float, default=0.7,
                    help="Ti le khung model bi MAT trong so khung chon (mac dinh 0.7).")
    ap.add_argument("--min-gap", type=float, default=0.4,
                    help="Bo khung cach khung vua chon it hon so giay nay (tranh khung gan nhu giong het).")
    ap.add_argument("--dry-run", action="store_true", help="Chi in danh sach khung se chon va luu anh xem truoc.")
    args = ap.parse_args()

    if args.review_auto:
        review_auto(args)
        return
    if args.review_claude:
        review_claude(args)
        return
    if not args.recordings:
        sys.exit("Can chi ra thu muc ban ghi (hoac dung --review-auto).")

    rec_dirs = []
    for pattern in args.recordings:
        rec_dirs += [d for d in glob.glob(os.path.join(BASE_DIR, pattern) if not os.path.isabs(pattern) else pattern)
                     if os.path.isdir(d)]
    if not rec_dirs:
        sys.exit("Khong tim thay thu muc ban ghi nao.")
    poses = {p.strip() for p in args.poses.split(",") if p.strip()}
    done = load_done()
    items, n_lost, n_kept = select_frames(rec_dirs, poses, args.count, args.lost_share, args.min_gap, done)
    print(f"{len(rec_dirs)} ban ghi | ung vien (da loc khung trung): {n_lost} khung MAT, {n_kept} khung DA BAM | "
          f"chon {len(items)} khung | da gan tu truoc: {sum(1 for v in done.values() if not v.get('skip'))}")
    if not items:
        sys.exit("Khong con khung nao de gan.")
    labeler = Labeler(items, done)
    if args.dry_run:
        from collections import Counter
        print("Theo dong tac:", dict(Counter(i["pose"] for i in items)))
        out = os.path.join(BASE_DIR, "labels", "preview.jpg")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        cv2.imwrite(out, labeler.render())
        print("Anh xem truoc:", out)
        return
    labeler.run()


if __name__ == "__main__":
    main()
