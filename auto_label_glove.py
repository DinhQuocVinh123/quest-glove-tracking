#!/usr/bin/env python3
"""
TU DONG gan nhan cac khung model bi MAT trong ban ghi --record, de fine-tune
ma khong phai ngoi gan tay ca ngay.

Y tuong: model thuong chi mat vai khung lien tiep, xen giua cac khung no bam
DUNG. O cac khung dung ngay truoc va ngay sau, 21 diem da biet; "bam theo"
tung diem qua cac khung bi mat bang optical flow (Lucas-Kanade) -- CA 2 CHIEU:
tien tu khung truoc, lui tu khung sau. Hai chieu ra cung cho => nhan dang tin
(tu nhan). Lech nhau => dua vao danh sach XEM LAI (nguoi keo sua, da dien san).

Phan loai tung khung:
  good     : model bam tot (qua kiem tra) -> lay mau thua lam nhan "model"
             (de model khong quen cai da biet khi fine-tune)
  no_hand  : model mat VA Quest khong thay tay trong khung anh (thao kinh, tay
             ngoai tam nhin...) -> BO, khong phai loi cua model
  fail     : model mat NHUNG Quest thay tay nam trong anh -> can nhan
             * chuoi ngan, co khung tot 2 dau -> flow 2 chieu -> auto / review
             * chuoi dai -> lay mau thua -> review (dien san dang tay gan nhat)

Ket qua: labels/auto_labels.jsonl (moi dong 1 khung) + bang anh de soat
(labels/auto_sheets/). Khung "review" sua bang:
  python label_glove_frames.py --review-auto

Cach dung:
  python auto_label_glove.py recordings/20260927_144842
  python auto_label_glove.py recordings/2026092*  --max-flow-run 12
"""
import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUTO_FILE = os.path.join(BASE_DIR, "labels", "auto_labels.jsonl")
SHEET_DIR = os.path.join(BASE_DIR, "labels", "auto_sheets")

SEEN, UNKNOWN = 1, 0
# Phai khop moi tu nhan: dau ngon cai, goc + dau ngon tro, goc ngon giua (cac diem cua dong tac pinch).
# KHONG gom co tay / ngon ut: vai gang trang tron o do nen flow kho bam, va chinh model cung dat co tay
# lech qua lai (do tren ban ghi 20260927_144842: lech tien/lui trung vi 0.29 o co tay, 0.05-0.11 o 4 diem nay).
# Diem nao lech nhieu -> danh dau KHONG BIET (bo qua khi huan luyen) thay vi bo ca khung.
KEY_POINTS = [4, 5, 8, 9]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
LK = dict(winSize=(31, 31), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))


def palm_len(p):
    return float(np.linalg.norm(p[9] - p[0]))


def sane(p):
    return p is not None and palm_len(p) > 25


HINT_MAX_OFFSET = 0.8  # khung dung lech <= 0.51 (toi da tren 2 ban ghi); bam nham tay trai lech 1.5-3.6


def matches_hint(p, h):
    """Tam long ban tay trung vi tri Quest bao? (khong co hint -> khong kiem duoc -> cho qua)"""
    if h is None:
        return True
    c = p[[0, 5, 9, 13, 17]].mean(axis=0)
    return float(np.linalg.norm(c - np.array(h[:2])) / max(h[2], 1.0)) <= HINT_MAX_OFFSET


FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]
MAX_BONE = 0.9  # 1 dot ngon dai hon 0.9 lan long ban tay la vo ly (binh thuong <= 0.55, p99)


MAX_JUMP = 0.3  # diem ngon nhay > 0.3 lan long ban tay so voi CA khung truoc lan khung sau -> nghi ngo


def temporal_check(p, prev, nxt, vis):
    """Model hay 'chop' sai 1 khung (ngon vang ra ngoai roi ve lai). Ngon nao lech
    xa CA khung tot truoc va sau (sau khi bu chuyen dong ca ban tay) -> KHONG BIET."""
    vis = list(vis)
    pl = max(palm_len(p), 1.0)
    refs = [q for q in (prev, nxt) if q is not None]
    if not refs:
        return vis
    for ch in FINGERS:
        far = []
        for q in refs:
            shift = p[[0, 5, 9, 13, 17]].mean(axis=0) - q[[0, 5, 9, 13, 17]].mean(axis=0)
            far.append(max(np.linalg.norm(p[k] - (q[k] + shift)) / pl for k in ch))
        if min(far) > MAX_JUMP:
            for k in ch:
                vis[k] = UNKNOWN
    return vis


def finger_check(p, vis):
    """Ngon nao co dot dai vo ly (vd bi keo sang tan man hinh) -> ca ngon KHONG BIET.
    Tra ve (vis moi, ngon cai + ngon tro con dung khong)."""
    vis = list(vis)
    pl = max(palm_len(p), 1.0)
    bad = []
    for fi, ch in enumerate(FINGERS):
        if any(np.linalg.norm(p[b] - p[a]) / pl > MAX_BONE for a, b in zip(ch[:-1], ch[1:])):
            bad.append(fi)
            for k in ch:
                vis[k] = UNKNOWN
    return vis, 0 not in bad and 1 not in bad


# --- Optical flow -----------------------------------------------------------

class Frames:
    """Doc anh xam co cache (1 chuoi chi vai chuc khung)."""
    def __init__(self, rec_dir):
        self.rec_dir, self.cache = rec_dir, {}

    def gray(self, f):
        if f not in self.cache:
            img = cv2.imread(os.path.join(self.rec_dir, f"{f:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
            self.cache[f] = img
        return self.cache[f]


def track(frames, seq, start_pts):
    """Bam 21 diem qua day khung seq (seq[0] = khung co diem start_pts).
    Tra ve {khung: (diem, so_diem_mat)}. Diem nao flow khong tin (kiem tra
    tien-lui tung buoc) thi di theo chuyen dong trung binh cua cac diem con lai."""
    out = {}
    pts = start_pts.astype(np.float32).reshape(-1, 1, 2)
    lost = np.zeros(21, bool)
    for a, b in zip(seq[:-1], seq[1:]):
        ga, gb = frames.gray(a), frames.gray(b)
        if ga is None or gb is None:
            break
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(ga, gb, pts, None, **LK)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gb, ga, nxt, None, **LK)
        fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 2.0)
        move = (nxt - pts).reshape(-1, 2)
        if ok.sum() >= 5:
            mean_move = np.median(move[ok], axis=0)
            move[~ok] = mean_move
        lost |= ~ok
        pts = pts + move.reshape(-1, 1, 2)
        out[b] = (pts.reshape(-1, 2).copy(), int(lost.sum()))
    return out


# --- Phan loai + de xuat nhan ----------------------------------------------

def load(rec_dir):
    rows = list(csv.DictReader(open(os.path.join(rec_dir, "log.csv"), encoding="utf-8")))
    kps = [json.loads(l) for l in open(os.path.join(rec_dir, "keypoints.jsonl"), encoding="utf-8")]
    by_frame = {k["frame"]: k for k in kps}
    return rows, by_frame


def process(rec_dir, args):
    name = os.path.basename(os.path.normpath(rec_dir))
    rows, kps = load(rec_dir)
    first = cv2.imread(os.path.join(rec_dir, f"{int(rows[0]['frame']):05d}.jpg"))
    H, W = first.shape[:2]
    frames = Frames(rec_dir)

    fids = [int(r["frame"]) for r in rows]
    times = {int(r["frame"]): float(r["t"]) for r in rows}
    cat, model_pts, hint = {}, {}, {}
    for r in rows:
        f = int(r["frame"])
        k = kps.get(f, {})
        p = np.array(k["kpts"], float)[:, :2] if k.get("kpts") else None
        model_pts[f] = p
        hint[f] = k.get("hint")
        if r["sent"] == "good" and sane(p) and matches_hint(p, hint[f]):
            cat[f] = "good"
        else:
            h = hint[f]
            in_view = h is not None and -0.1 * W < h[0] < 1.1 * W and -0.1 * H < h[1] < 1.1 * H
            cat[f] = "fail" if in_view else "no_hand"

    records = []
    # 1) Khung tot: lay mau thua (cach nhau >= --model-gap giay)
    last = -1e9
    for n, f in enumerate(fids):
        if cat[f] == "good" and times[f] - last >= args.model_gap:
            prev = model_pts[fids[n - 1]] if n > 0 and cat[fids[n - 1]] == "good" else None
            nxt = model_pts[fids[n + 1]] if n + 1 < len(fids) and cat[fids[n + 1]] == "good" else None
            if prev is None and nxt is None:
                continue  # khung tot le loi giua 2 khung loi: khong du tin
            last = times[f]
            vis = temporal_check(model_pts[f], prev, nxt, [SEEN] * 21)
            vis, pinch_ok = finger_check(model_pts[f], vis)
            pinch_ok = pinch_ok and all(vis[k] for k in (1, 2, 3, 4, 5, 6, 7, 8))
            if not pinch_ok:
                continue
            records.append(dict(recording=name, rec_dir=rec_dir, frame=f, kpts=model_pts[f].round(1).tolist(),
                                visible=vis, source="model", status="auto", agree=0.0))

    # 2) Chuoi khung loi (fail/no_hand lien tiep) giua 2 khung tot
    i = 0
    stats = dict(flow_auto=0, flow_review=0, long_review=0, long_skipped=0)
    while i < len(fids):
        if cat[fids[i]] == "good":
            i += 1
            continue
        j = i
        while j < len(fids) and cat[fids[j]] != "good":
            j += 1
        run = fids[i:j]
        fails = [f for f in run if cat[f] == "fail"]
        a = fids[i - 1] if i > 0 else None
        b = fids[j] if j < len(fids) else None
        if fails:
            if a is not None and b is not None and len(run) <= args.max_flow_run:
                fwd = track(frames, [a] + run, model_pts[a])
                bwd = track(frames, [b] + run[::-1], model_pts[b])
                scale = 0.5 * (palm_len(model_pts[a]) + palm_len(model_pts[b]))
                for n, f in enumerate(run, start=1):
                    if cat[f] != "fail" or f not in fwd or f not in bwd:
                        continue
                    pf, lf = fwd[f]
                    pb, lb = bwd[f]
                    w = n / (len(run) + 1)                  # gan khung nao hon thi tin chieu do hon
                    prop = (1 - w) * pf + w * pb
                    d = np.linalg.norm(pf - pb, axis=1) / max(scale, 1.0)
                    key_ok = bool(np.all(d[KEY_POINTS] < args.agree))
                    vis = [SEEN if di < 2 * args.agree else UNKNOWN for di in d]
                    vis, pinch_ok = finger_check(prop, vis)
                    auto = key_ok and pinch_ok and sum(vis) >= 14 and sane(prop)
                    stats["flow_auto" if auto else "flow_review"] += 1
                    records.append(dict(recording=name, rec_dir=rec_dir, frame=f, kpts=prop.round(1).tolist(),
                                        visible=vis if auto else [SEEN] * 21, source="flow",
                                        status="auto" if auto else "review", agree=round(float(np.median(d)), 3)))
            else:
                # Chuoi dai: lay mau thua, dien san dang tay cua khung tot gan nhat,
                # dich theo tam ban tay Quest bao (hint) de nguoi chi con keo ngon.
                last = -1e9
                for f in fails:
                    if times[f] - last < args.review_gap:
                        stats["long_skipped"] += 1
                        continue
                    last = times[f]
                    near = min((x for x in (a, b) if x is not None), key=lambda x: abs(x - f), default=None)
                    prop = model_pts[f] if sane(model_pts[f]) else None
                    if prop is None and near is not None:
                        prop = model_pts[near].copy()
                        if hint[f] and hint[near]:
                            prop += np.array(hint[f][:2]) - np.array(hint[near][:2])
                    records.append(dict(recording=name, rec_dir=rec_dir, frame=f,
                                        kpts=None if prop is None else np.round(prop, 1).tolist(),
                                        visible=[SEEN] * 21, source="prefill", status="review", agree=-1.0))
                    stats["long_review"] += 1
        i = j

    counts = {c: sum(1 for f in fids if cat[f] == c) for c in ("good", "fail", "no_hand")}
    return records, counts, stats


# --- Bang anh de soat --------------------------------------------------------

def draw(img, p, vis, color_by_finger=True):
    colors = [(0, 140, 255), (255, 200, 0), (255, 0, 255), (160, 60, 255), (0, 220, 0)]
    for a, b in BONES:
        c = colors[(b - 1) // 4]
        cv2.line(img, tuple(np.int32(p[a])), tuple(np.int32(p[b])), c, 2)
    for k, q in enumerate(p):
        cv2.circle(img, tuple(np.int32(q)), 4 if vis[k] else 2, (255, 255, 255) if vis[k] else (0, 0, 255), -1)


def crop_tile(rec_dir, rec, size=240):
    img = cv2.imread(os.path.join(rec_dir, f"{rec['frame']:05d}.jpg"))
    p = np.array(rec["kpts"], float)
    draw(img, p, rec["visible"])
    c = p.mean(axis=0)
    s = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), 80) * 0.8
    x0, y0 = int(c[0] - s), int(c[1] - s)
    pad = int(s) + 10
    big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(60, 60, 60))
    tile = big[y0 + pad:y0 + pad + int(2 * s), x0 + pad:x0 + pad + int(2 * s)]
    tile = cv2.resize(tile, (size, size))
    cv2.putText(tile, f"{rec['frame']} {rec['source'][0]} {rec['agree']:.2f}", (4, 16), 0, 0.45, (0, 255, 255), 1)
    return tile


def make_sheets(recs, tag, per=30, cols=6):
    os.makedirs(SHEET_DIR, exist_ok=True)
    paths = []
    for s in range(0, len(recs), per):
        tiles = [crop_tile(r["rec_dir"], r) for r in recs[s:s + per]]
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[k:k + cols]) for k in range(0, len(tiles), cols)])
        path = os.path.join(SHEET_DIR, f"{tag}_{s // per:02d}.jpg")
        cv2.imwrite(path, sheet)
        paths.append(path)
    return paths


def main():
    ap = argparse.ArgumentParser(description="Tu dong gan nhan khung model mat tren ban ghi --record.")
    ap.add_argument("recordings", nargs="+")
    ap.add_argument("--max-flow-run", type=int, default=6, help="Chuoi loi dai toi da (khung) de bam bang flow 2 chieu.")
    ap.add_argument("--agree", type=float, default=0.12, help="Lech toi da giua flow tien/lui (don vi = chieu dai long ban tay).")
    ap.add_argument("--model-gap", type=float, default=0.5, help="Khoang cach (giay) giua cac khung 'model tot' lay lam nhan.")
    ap.add_argument("--review-gap", type=float, default=0.6, help="Khoang cach (giay) khi lay mau chuoi loi dai de xem lai.")
    ap.add_argument("--no-sheets", action="store_true")
    args = ap.parse_args()

    rec_dirs = [d for pat in args.recordings for d in sorted(glob.glob(pat)) if os.path.exists(os.path.join(d, "keypoints.jsonl"))]
    names = {os.path.basename(os.path.normpath(d)) for d in rec_dirs}
    kept = []
    if os.path.exists(AUTO_FILE):  # chay lai: thay ban ghi cu cua cung recording
        kept = [l for l in open(AUTO_FILE, encoding="utf-8") if json.loads(l)["recording"] not in names]

    all_recs = []
    for d in rec_dirs:
        recs, counts, stats = process(d, args)
        all_recs += recs
        print(f"{os.path.basename(os.path.normpath(d))}: {counts} -> {stats}")
    os.makedirs(os.path.dirname(AUTO_FILE), exist_ok=True)
    with open(AUTO_FILE, "w", encoding="utf-8") as f:
        f.writelines(kept)
        for r in all_recs:
            f.write(json.dumps(r) + "\n")
    n_auto = sum(1 for r in all_recs if r["status"] == "auto" and r["source"] == "flow")
    n_rev = sum(1 for r in all_recs if r["status"] == "review")
    n_model = sum(1 for r in all_recs if r["source"] == "model")
    print(f"=> {n_auto} khung loi tu gan nhan (flow), {n_rev} khung can xem lai, {n_model} khung model tot. Luu: {AUTO_FILE}")
    if not args.no_sheets:
        flow = [r for r in all_recs if r["source"] == "flow" and r["status"] == "auto"]
        for p in make_sheets(flow, "flow_auto"):
            print("  bang soat:", p)


if __name__ == "__main__":
    main()
