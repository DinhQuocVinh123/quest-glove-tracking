#!/usr/bin/env python3
"""
Tu gan nhan 9 diem (co tay + ngon cai + ngon tro) nho BANG KEO MAU quan quanh ngon gang:
  ngon cai: 3 vong bang CAM  -> khop 2, khop 3, dau ngon 4
  ngon tro: 3 vong bang XANH -> khop 6, khop 7, dau ngon 8
  3 diem KHONG dan (co tay 0, goc ngon cai 1, goc ngon tro 5): lay tu model, trong so 0.6
  ("uoc luong"), va bo han neu model lech xa bang mau (model dang doan sai ca ngon).

Tim bang mau (khong can AI):
  1. Loc mau HSV trong vung quanh ban tay.
  2. Hat giong = manh mau gan dau ngon model doan nhat; khung model mat tay thi dung vi tri
     bang o khung truoc (toi da MAX_GAP khung) -- day chinh la nhung khung quy nhat.
  3. Gom cac manh nam lien nhau (cach < 0.12 kich thuoc tay); bo manh nho (chu/icon tren man hinh).
  4. Gop 2 nua cua 1 bang bi ong den cat ngang; 2 bang dinh nhau thi cat doi o khe it mau nhat.
  5. Dung 3 cum -> xep tu goc ra dau ngon.
  Bang dau ngon nam lui vao ~1 cm so voi dau ngon that -> keo dai theo huong ngon, ti le
  lay tu du lieu (trung vi vi tri dau ngon model doan o nhung khung model dung).

Loc khung xau: khoang cach giua cac bang hop ly, khong nhay qua xa so voi khung truoc.

Cach dung (khong can GPU, chay duoc khi run_glove_quest_stream.py dang tat):
  python color_label_glove.py                          # ban ghi moi nhat
  python color_label_glove.py recordings/20260930_160426 recordings/...
Ket qua: labels/color_labels.jsonl (finetune_pinch.py tu doc) + labels/color_sheets/*.jpg de xem nhanh.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_FILE = os.path.join(_BASE_DIR, "labels", "color_labels.jsonl")
SHEET_DIR = os.path.join(_BASE_DIR, "labels", "color_sheets")

FINGERS = {  # mau bang, diem goc (khong dan), 3 diem co bang [gan, giua, dau ngon]
    "thumb": dict(color="orange", base=1, joints=[2, 3, 4]),
    "index": dict(color="blue", base=5, joints=[6, 7, 8]),
}
MAX_GAP = 5            # khung model mat tay: dung bang cua khung truoc neu cach <= bay nhieu khung
MAX_JUMP = 0.25        # diem nhay xa hon (x kich thuoc tay) so voi khung truoc -> bo
MODEL_AGREE = 0.25     # dau ngon model lech bang mau hon muc nay -> khong tin diem goc cua model
HARD_ERR = 0.08        # dau ngon model lech hon muc nay (hoac mat tay) -> khung "kho", train lap lai


# --- Tim bang mau -------------------------------------------------------------

def color_mask(hsv, color):
    h, s, v = (hsv[..., i].astype(np.int32) for i in range(3))
    if color == "blue":   # bang keo giay xanh da troi: H~100, S~130-160
        m = (h >= 92) & (h <= 110) & (s >= 90) & (v >= 70)
    else:                 # bang cam: H~6, S~160 (nhua mau da S~55, khong lot vao)
        m = ((h <= 14) | (h >= 172)) & (s >= 120) & (v >= 55)
    m = m.astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


def finger_bands(mask, seed, base, size):
    """Tam 3 bang (x, y) xep tu GOC ra DAU ngon, hoac (None, ly do)."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    min_area = max(12, int((0.012 * size) ** 2))
    comps = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not comps:
        return None, "khong thay mau"
    d_seed = {i: float(np.hypot(*(cent[i] - seed))) for i in comps}
    first = min(comps, key=lambda i: d_seed[i])
    if d_seed[first] > 0.25 * size:
        return None, "mau xa dau ngon"
    # Gom lan: no rong vung da gom `link` px (distance transform), manh nao cham vao thi gom
    link = 0.12 * size
    near = {i for i in comps if d_seed[i] <= 0.45 * size}
    group, lut = {first}, np.zeros(n, np.uint8)
    while True:
        lut[:] = 0
        lut[list(group)] = 1
        dist = cv2.distanceTransform((1 - lut[lab]).astype(np.uint8), cv2.DIST_L2, 3)
        touched = set(np.unique(lab[(dist <= link) & (lab > 0)]).tolist()) & near
        if touched <= group:
            break
        group |= touched
    big = max(stats[i, cv2.CC_STAT_AREA] for i in group)
    group = [i for i in group if stats[i, cv2.CC_STAT_AREA] >= 0.15 * big]
    pts = {i: np.column_stack(np.nonzero(lab == i))[:, ::-1].astype(float) for i in group}
    # 2 nua cua 1 bang (ong den cat ngang): tam gan nhau -> 1 cum
    clusters = []
    for i in sorted(group, key=lambda i: -stats[i, cv2.CC_STAT_AREA]):
        for cl in clusters:
            if min(np.hypot(*(cent[i] - cent[j])) for j in cl) < 0.08 * size:
                cl.append(i)
                break
        else:
            clusters.append([i])
    cl_pts = [np.vstack([pts[i] for i in cl]) for cl in clusters]
    if len(cl_pts) == 2:
        # 2 bang sat nhau dinh thanh 1 vet: vet do dai gap doi vet kia theo huong ngon -> cat doi
        d = cl_pts[1].mean(0) - cl_pts[0].mean(0)
        d = d / (np.linalg.norm(d) + 1e-9)
        ext = [np.percentile(q @ d, 95) - np.percentile(q @ d, 5) for q in cl_pts]
        li = int(np.argmax(ext))
        if ext[li] > 1.7 * ext[1 - li]:
            q = cl_pts[li]
            t = q @ d
            hist, edges = np.histogram(t, bins=np.linspace(np.percentile(t, 5), np.percentile(t, 95), 21))
            hs = np.convolve(hist, np.ones(3) / 3, mode="same")
            cut = edges[6 + int(np.argmin(hs[6:15]))] + (edges[1] - edges[0]) / 2
            a, b = q[t < cut], q[t >= cut]
            if len(a) > 0.2 * len(q) and len(b) > 0.2 * len(q):
                cl_pts = [cl_pts[1 - li], a, b]
    if len(cl_pts) != 3:
        return None, f"{len(cl_pts)} cum mau"
    cs = [q.mean(0) for q in cl_pts]
    order, cur, left = [], np.asarray(base, float), [0, 1, 2]
    while left:  # tu goc ngon: bang gan nhat, roi bang gan bang do nhat...
        nxt = min(left, key=lambda k: np.hypot(*(cs[k] - cur)))
        order.append(cs[nxt])
        cur = cs[nxt]
        left.remove(nxt)
    return order, "ok"


def detect(img, region, seeds, size):
    """region = (x0, y0, x1, y1); seeds[fn] = (dau ngon, goc ngon) toa do anh. Tra ve {fn: 3 tam bang}."""
    x0, y0, x1, y1 = (int(round(v)) for v in region)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return {}, {}
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    off = np.array([x0, y0], float)
    out, why = {}, {}
    for fn, (tip, base) in seeds.items():
        c, w = finger_bands(color_mask(hsv, FINGERS[fn]["color"]), np.asarray(tip) - off, np.asarray(base) - off, size)
        why[fn] = w
        if c is not None:
            out[fn] = [p + off for p in c]
    return out, why


def spacing_ok(b, size):
    d1, d2 = np.hypot(*(b[1] - b[0])), np.hypot(*(b[2] - b[1]))
    return 0.03 * size < d1 < 0.35 * size and 0.03 * size < d2 < 0.35 * size and 0.35 < d1 / d2 < 2.8


# --- Gan nhan 1 ban ghi --------------------------------------------------------

def process(rec_dir):
    rec = os.path.basename(os.path.normpath(rec_dir))
    rows = [json.loads(l) for l in open(os.path.join(rec_dir, "keypoints.jsonl"), encoding="utf-8")]
    found = {}       # frame -> {fn: bands}
    info = {}        # frame -> (model kpts | None, size, from_model)
    last = {}        # fn -> (frame, bands, size)
    reasons = {fn: {} for fn in FINGERS}
    for r in rows:
        f = r["frame"]
        img = cv2.imread(os.path.join(rec_dir, f"{f:05d}.jpg"))
        if img is None:
            continue
        k = np.array(r["kpts"], float) if r.get("found") and r.get("kpts") else None
        seeds, region, size = {}, None, None
        if k is not None:
            size = float(max(np.ptp(k[:, 0]), np.ptp(k[:, 1])))
            lo, hi = k.min(0) - 0.2 * size, k.max(0) + 0.2 * size
            region = (*lo, *hi)
            seeds = {fn: (k[c["joints"][2]], k[c["base"]]) for fn, c in FINGERS.items()}
        else:
            # Model mat tay: tim tiep quanh vi tri bang o khung truoc
            recent = {fn: v for fn, v in last.items() if f - v[0] <= MAX_GAP}
            if recent:
                size = float(np.mean([v[2] for v in recent.values()]))
                allp = np.vstack([np.vstack(v[1]) for v in recent.values()])
                lo, hi = allp.min(0) - 0.5 * size, allp.max(0) + 0.5 * size
                region = (*lo, *hi)
                seeds = {fn: (v[1][2], v[1][0] - (v[1][1] - v[1][0])) for fn, v in recent.items()}
        if not seeds:
            for fn in FINGERS:
                reasons[fn]["mat tay"] = reasons[fn].get("mat tay", 0) + 1
            continue
        res, why = detect(img, region, seeds, size)
        for fn in FINGERS:
            w = why.get(fn, "mat tay")
            if fn in res and not spacing_ok(res[fn], size):
                res.pop(fn)
                w = "khoang cach bang la"
            if fn in res and k is None and fn in last:
                # khung tim theo khung truoc: bang phai o gan cho cu
                if max(np.hypot(*(a - b)) for a, b in zip(res[fn], last[fn][1])) > MAX_JUMP * size:
                    res.pop(fn)
                    w = "nhay xa"
            reasons[fn][w] = reasons[fn].get(w, 0) + 1
        for fn, b in res.items():
            last[fn] = (f, b, size)
        if res:
            found[f] = res
            info[f] = (k, size, r.get("hint"))
    return rec, found, info, reasons


def tip_ratio(found, info):
    """Dau ngon that = bang dau + r * (bang dau - bang giua). r = trung vi tu nhung khung model
    doan dau ngon nam tren truc ngon (model dang dung)."""
    ratios = {fn: [] for fn in FINGERS}
    for f, res in found.items():
        k = info[f][0]
        if k is None:
            continue
        for fn, b in res.items():
            seg = b[2] - b[1]
            L = np.linalg.norm(seg)
            if L < 1e-6:
                continue
            u = seg / L
            v = k[FINGERS[fn]["joints"][2]] - b[2]
            along, perp = float(v @ u) / L, abs(float(v[0] * u[1] - v[1] * u[0])) / L
            if perp < 0.4 and -0.5 < along < 1.5:
                ratios[fn].append(along)
    return {fn: float(np.clip(np.median(v), 0.0, 1.0)) if len(v) >= 20 else 0.3 for fn, v in ratios.items()}, \
           {fn: len(v) for fn, v in ratios.items()}


def build_labels(rec, found, info, ratio):
    labels, prev = [], {}
    rejected = 0
    for f in sorted(found):
        res = found[f]
        k, size, hint = info[f]
        kp = np.zeros((21, 2))
        vis = [0] * 21
        errs = []   # chi do o DAU NGON (diem ro rang nhat) de danh dau khung kho
        agree = {}
        for fn, b in res.items():
            cfg = FINGERS[fn]
            tip = b[2] + ratio[fn] * (b[2] - b[1])
            pts = [b[0], b[1], tip]
            for j, p in zip(cfg["joints"], pts):
                kp[j] = p
                vis[j] = 1
            if k is not None:
                e = [np.hypot(*(p - k[j])) / size for j, p in zip(cfg["joints"], pts)]
                errs.append(e[2])
                agree[fn] = e[2] < MODEL_AGREE
        # Nhay xa so voi khung truoc (con lai sau loc) -> bo ca khung
        if f - 1 in prev or f - 2 in prev:
            pk, pv = prev.get(f - 1) or prev.get(f - 2)
            jumps = [np.hypot(*(kp[j] - pk[j])) / size for j in range(9) if vis[j] == 1 and pv[j] == 1]
            if jumps and max(jumps) > MAX_JUMP:
                rejected += 1
                continue
        # Diem khong dan: lay tu model (uoc luong), chi khi model dang dung ngon do
        if k is not None:
            for fn, cfg in FINGERS.items():
                if agree.get(fn):
                    kp[cfg["base"]] = k[cfg["base"]]
                    vis[cfg["base"]] = 2
            if agree and all(agree.values()):
                kp[0] = k[0]
                vis[0] = 2
        model_err = float(max(errs)) if errs else None
        hard = k is None or (model_err is not None and model_err > HARD_ERR)
        prev[f] = (kp.copy(), vis[:])
        labels.append(dict(recording=rec, frame=f, image=os.path.join("recordings", rec, f"{f:05d}.jpg"),
                           kpts=[[round(float(x), 1), round(float(y), 1)] for x, y in kp], visible=vis,
                           source="color", hard=bool(hard), model_lost=k is None,
                           model_err=None if model_err is None else round(model_err, 3)))
    return labels, rejected


# --- Bang anh xem nhanh --------------------------------------------------------

def draw_tile(rec_dir, lab, info_f):
    img = cv2.imread(os.path.join(rec_dir, f"{lab['frame']:05d}.jpg"))
    kp, vis = np.array(lab["kpts"]), lab["visible"]
    k = info_f[0]
    known = kp[[j for j in range(9) if vis[j]]]
    c = known.mean(0)
    size = info_f[1]
    x0, y0 = int(max(0, c[0] - 0.55 * size)), int(max(0, c[1] - 0.55 * size))
    x1, y1 = int(min(img.shape[1], c[0] + 0.55 * size)), int(min(img.shape[0], c[1] + 0.55 * size))
    if k is not None:
        for j in range(9):
            cv2.circle(img, tuple(int(v) for v in k[j]), 4, (0, 255, 0), -1)
    for chain, col in (([0, 1, 2, 3, 4], (0, 0, 255)), ([0, 5, 6, 7, 8], (255, 0, 255))):
        pts = [(j, tuple(int(v) for v in kp[j])) for j in chain if vis[j]]
        for (ja, a), (jb, b) in zip(pts, pts[1:]):
            cv2.line(img, a, b, col, 2)
        for j, p in pts:
            cv2.circle(img, p, 7, col, 2 if vis[j] == 1 else 1)
    tile = img[y0:y1, x0:x1]
    tile = cv2.resize(tile, (300, 300))
    tag = f"f{lab['frame']}" + (" MAT TAY" if lab["model_lost"] else f" lech {lab['model_err']:.2f}")
    cv2.putText(tile, tag, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if lab["hard"] else (0, 160, 0), 2)
    return tile


def save_sheets(rec_dir, rec, labels, info):
    os.makedirs(SHEET_DIR, exist_ok=True)
    groups = {
        "mau": labels[::max(1, len(labels) // 20)][:20],
        "kho": [l for l in labels if l["hard"]][::max(1, sum(l["hard"] for l in labels) // 20)][:20],
    }
    paths = []
    for name, labs in groups.items():
        if not labs:
            continue
        tiles = [draw_tile(rec_dir, l, info[l["frame"]]) for l in labs]
        while len(tiles) % 5:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
        p = os.path.join(SHEET_DIR, f"{rec}_{name}.jpg")
        cv2.imwrite(p, sheet)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser(description="Tu gan nhan 9 diem nho bang keo mau tren gang.")
    ap.add_argument("recordings", nargs="*", help="Thu muc ban ghi (mac dinh: ban ghi moi nhat).")
    ap.add_argument("--out", default=OUT_FILE)
    args = ap.parse_args()
    recs = args.recordings or [max(glob.glob(os.path.join(_BASE_DIR, "recordings", "*")), key=os.path.getmtime)]
    recs = [r if os.path.isabs(r) else os.path.join(_BASE_DIR, r) for r in recs]

    # Giu nhan mau cua cac ban ghi KHAC da co trong file
    names = {os.path.basename(os.path.normpath(r)) for r in recs}
    kept = []
    if os.path.exists(args.out):
        kept = [l for l in open(args.out, encoding="utf-8") if json.loads(l)["recording"] not in names]

    all_labels = []
    for rec_dir in recs:
        rec, found, info, reasons = process(rec_dir)
        ratio, n_ratio = tip_ratio(found, info)
        labels, rejected = build_labels(rec, found, info, ratio)
        all_labels += labels
        n = len(open(os.path.join(rec_dir, "keypoints.jsonl"), encoding="utf-8").readlines())
        lost = sum(l["model_lost"] for l in labels)
        hard = sum(l["hard"] for l in labels)
        both = sum(1 for l in labels if l["visible"][4] == 1 and l["visible"][8] == 1)
        print(f"\n[{rec}] {n} khung -> {len(labels)} khung co nhan ({both} du ca 2 ngon), bo {rejected} khung nhay xa")
        print(f"  khung KHO (model lech > {HARD_ERR} hoac mat tay): {hard}, trong do model MAT TAY: {lost}")
        for fn in FINGERS:
            print(f"  {fn}: dau ngon keo dai {ratio[fn]:.2f} x (tu {n_ratio[fn]} khung) | {reasons[fn]}")
        for p in save_sheets(rec_dir, rec, labels, info):
            print("  bang anh:", p)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fo:
        fo.writelines(kept)
        for l in all_labels:
            fo.write(json.dumps(l, ensure_ascii=False) + "\n")
    print(f"\nGhi {len(all_labels)} nhan moi (+{len(kept)} nhan cu cua ban ghi khac) -> {args.out}")


if __name__ == "__main__":
    main()
