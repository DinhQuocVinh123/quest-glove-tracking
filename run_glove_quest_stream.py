#!/usr/bin/env python3
"""
Glove (Quest camera) -> Unity UDP Bridge.

Giống hệt run_glove_to_unity.py (cùng logic detect/tracking/pinch), nhưng
NGUỒN ẢNH là camera Passthrough thật của kính Quest (gửi liên tục qua WiFi
bởi GloveLiveStreamer.cs) thay vì webcam của máy tính. Không cần điện
thoại/webcam rời nữa -- camera duy nhất là của kính.

Máy tính đóng vai trò TCP SERVER (nhận khung hình liên tục từ kính) +
UDP CLIENT (gửi kết quả ngón tay/pinch ngược lại FingerUDPReceiver trên
kính). IP của kính được TỰ ĐỘNG lấy từ địa chỉ kết nối TCP đến -- không
cần điền tay.

Cách dùng:
  python run_glove_quest_stream.py --original
Rồi trong Unity (GloveLiveStreamer), điền đúng IP máy này vào _pcIpAddress
(cổng mặc định 5007, khớp --port ở đây).
"""

import argparse
import json
import os
import socket
import struct
import sys
import threading
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
# Model fine-tune cho PINCH voi gang (finetune_pinch.py, 27/09/2026): mat tay khi tay trong anh
# 12.5% -> 0-3% tren ban ghi thu. Mac dinh dung model nay neu co; --original = model goc (tay tran),
# --checkpoint = file khac. (_FINETUNED_CKPT la ban fine-tune CU, kem hon model goc -- khong dung mac dinh.)
_PINCH_CKPT = os.path.join(_BASE_DIR, "checkpoints", "rtmpose_glove_pinch.pth")
CHECKPOINT_FILE = _PINCH_CKPT if os.path.exists(_PINCH_CKPT) else _ORIGINAL_CKPT_URL

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


# Do XOE (dang) giua ngon cai va ngon tro -- bac tu do RIENG BIET voi do
# cong. Tu the "chu C" (2 ngon deu THANG nhung cach nhau vai cm) duoc tao
# ra hoan toan boi bac tu do nay, khong phai boi do cong -- nen neu chi gui
# do cong thi tay ao KHONG THE tao ra tu the do du hieu chinh kieu gi.
MAX_SPREAD_DEG = 60.0


def calculate_thumb_index_spread(kpts):
    """Goc xoe that giua ngon cai va ngon tro, nhin tu co tay.
    Tra ve (gia_tri_0_1, goc_do_that) -- 0 = 2 dau ngon chum lai (kep dong),
    1 = xoe rong toi da (MAX_SPREAD_DEG)."""
    wrist = kpts[0]
    thumb_dir = kpts[4] - wrist   # co tay -> dau ngon cai
    index_dir = kpts[8] - wrist   # co tay -> dau ngon tro
    angle = _joint_bend_angle_deg(thumb_dir, index_dir)
    return float(np.clip(angle / MAX_SPREAD_DEG, 0.0, 1.0)), angle


# 3 ngon giua/ap ut/ut: model gang moi khong hoc -> khong ve len cua so (Unity luon ep nam).
MRP_ALWAYS_CLOSED = True


def build_pixel_payload(kpts, scores, w, h):
    """9 diem dau (co tay, ngon cai 1-4, ngon tro 5-8) theo TI LE ANH: x/rong, y/cao (goc tren-trai)
    + do tin cay tung diem. Unity (ImageHandSolver) doi moi diem thanh 1 TIA tu camera luc chup va
    khop ca ban tay ao vao 9 tia."""
    pv = ";".join(f"{kpts[i][0] / w:.4f}|{kpts[i][1] / h:.4f}" for i in range(9))
    pc = ";".join(f"{float(scores[i]):.2f}" for i in range(9))
    return pv, pc


# --- Kiem tra dang tay co HOP LY khong (fallback ve tay binh thuong) ---------
# Model doi khi doan sai (tay chua vao tu the san sang, bi che khuat, mo...),
# tao ra dang tay cong venh / lat nguoc ra sau trong rat ky quac. Thay vi hien
# dang sai do, ta phat hien va bao Unity chuyen muot ve TU THE NGHI binh thuong.
#
# Cac nguong duoi day co y de RONG RAI: tha lot vai khung hoi sai con hon
# tu choi nham khung dung lien tuc (se gay giat/nhap nhay giua 2 dang tay).
# Ha thap hon truoc (0.30): model chua tung hoc chiec gang nay nen do tin cay
# von da sat nguong -- nen rat co (vd nen man hinh) la tut xuong duoi va bi loai
# oan. Bu lai, ta da co cac kiem tra HINH DANG (chieu dai dot, nhay dot ngot) o
# duoi, loai duoc dang sai mot cach doc lap voi do tin cay.
# Ha tiep (0.18 -> 0.12): tren nen roi, do tin cay tung diem dao dong manh va
# hay tut xuong sat nguong -- da tung bi loai vi thua dung 0.01. Diem so cua
# model von khong dang tin cho chiec gang nay, nen de cac kiem tra HINH DANG
# ben duoi lam nhiem vu gac cong thay vi phu thuoc vao nguong diem so.
FINGER_POINT_CONF_THR = 0.12    # do tin cay toi thieu cua tung diem tren ngon cai/tro
SEG_MIN_RATIO = 0.06            # 1 dot ngan nhat = 6% chieu dai long ban tay
SEG_MAX_RATIO = 1.00            # 1 dot dai nhat = 100% chieu dai long ban tay
# Dot NGAN chi la loi khi model khong chac: ngon CHIA VAO/RA camera thi dot giua + dot cuoi tren anh co lai con
# ~5% long ban tay -- tu the hop le. Ban ghi 03/10 13:44: 192 khung ngon tro chia ra xa bi loai oan (diem tin
# cay p5 0.80); khung model roi that (02/10) dot ngan co tin cay ~0.2.
SHORT_SEG_MIN_CONF = 0.5
FINGER_MIN_RATIO = 0.35         # ca ngon ngan nhat = 35% long ban tay
FINGER_MAX_RATIO = 2.00         # ca ngon dai nhat = 200% long ban tay
MAX_JUMP_RATIO = 1.20           # 1 diem khong the nhay qua 120% long ban tay trong 1 khung


def palm_length(kpts):
    """Chieu dai long ban tay = co tay -> khop goc ngon TRO (diem 5). Truoc dung diem 9 (ngon
    giua) -- model gang moi khong hoc 3 ngon giua/ap ut/ut nen diem 9 khong dang tin. Tren tay
    that |0-5| ~ |0-9| (do 0.95-1.0) nen cac nguong cu van dung."""
    return float(np.linalg.norm(np.asarray(kpts[5]) - np.asarray(kpts[0])))


def is_pose_plausible(kpts, scores, prev_kpts=None, bad_points=None):
    """Kiem tra rieng dang NGON CAI + NGON TRO co hop ly ve giai phau khong.

    Khac voi is_valid_hand (chi xet co tay + cac khop GOC), ham nay xet chinh
    cac diem tao nen hinh dang ngon tay -- ke ca dau ngon. Do la lo hong khien
    khung hinh co khop goc ro nhung dau ngon doan bay van lot qua.

    bad_points (set): neu dua vao, 1 dot sai chieu dai tren moi ngon KHONG lam loai ca khung --
    diem dau ngoai cua dot do duoc ghi vao set (gui do tin cay 0, Unity ImageHandSolver bo qua diem
    do, xuong co chieu dai co dinh nen van dung duoc ngon). Ban ghi 01/10 16:58: 61/538 khung bi
    loai chi vi 1 dot (thuong do long ban tay nghieng lam |0-5| tren anh ngan lai).

    Tra ve (hop_ly, ly_do)."""
    wrist = kpts[0]
    palm_len = palm_length(kpts)
    if palm_len < 1e-4:
        return False, "long ban tay ~ 0"

    # 1. Do tin cay cua TUNG diem tren 2 ngon (ke ca dau ngon). Bo diem 1, 2 (goc / khop gan goc
    # ngon cai): nhan da duoc dich tu tam bang ve tam khop nen model cham diem tu tin thap hon du
    # vi tri van dung -- ban ghi 01/10 16:03: 361/602 khung bi loai oan vi 2 diem nay.
    for i in (3, 4, 5, 6, 7, 8):
        if scores[i] < FINGER_POINT_CONF_THR:
            return False, f"diem {i} do tin cay thap ({scores[i]:.2f})"

    # 2. Tung dot xuong phai co chieu dai hop ly so voi long ban tay.
    for base, name in ((1, "cai"), (5, "tro")):
        total = 0.0
        flagged = []
        for j in range(3):
            seg = float(np.linalg.norm(kpts[base + j + 1] - kpts[base + j]))
            total += seg
            too_short = seg < SEG_MIN_RATIO * palm_len and min(scores[base + j], scores[base + j + 1]) < SHORT_SEG_MIN_CONF
            if too_short or seg > SEG_MAX_RATIO * palm_len:
                flagged.append(base + j + 1)
        if flagged and (bad_points is None or len(flagged) > 1):
            return False, f"dot {flagged[0] - base - 1} ngon {name} dai bat thuong"
        if flagged:
            bad_points.update(flagged)
            continue
        # 3. Tong chieu dai ca ngon cung phai hop ly.
        if total < FINGER_MIN_RATIO * palm_len or total > FINGER_MAX_RATIO * palm_len:
            return False, f"ngon {name} dai bat thuong"

    # 4. Khong diem nao duoc "nhay" qua xa so voi khung truoc -- dau hieu dien
    # hinh cua mot khung doan sai dot ngot (tay that khong the dich chuyen
    # nhanh nhu vay giua 2 khung lien tiep).
    if prev_kpts is not None:
        for i in (1, 2, 3, 4, 5, 6, 7, 8):
            if float(np.linalg.norm(kpts[i] - prev_kpts[i])) > MAX_JUMP_RATIO * palm_len:
                return False, f"diem {i} nhay dot ngot"

    return True, ""


# Ban tay phai chiem it nhat tung nay so voi canh ngan cua khung hinh.
# Camera gan TREN DAU va ban tay gan vao canh tay nguoi deo, nen no khong the
# nho ti hon duoc: du duoi thang het co thi long ban tay van chiem ~15-30%.
# Thieu kiem tra nay, model rat de KHOA NHAM vao mot chi tiet nho (cum module
# tren gang, mot vet tren man hinh...) va bao "bo xuong tay ti hon" -- ma cac
# ti le BEN TRONG cum do lai tu no hop ly nen moi kiem tra khac deu cho qua.
# Nguy hiem hon: khoa nham roi thi khung theo doi bam mai vao do, khong tu thoat.
MIN_PALM_FRAME_RATIO = 0.07

# So khung hinh duoc phep giu nguyen dang tot cuoi cung truoc khi bao Unity
# ve tu the nghi (~0.4 giay o 12 fps). Du dai de bac qua cac lan do tin cay
# chop nhoang tut xuong, nhung du ngan de khong giu mot dang sai qua lau khi
# tay that su ra khoi tam nhin.
HOLD_LAST_GOOD_FRAMES = 5


# Tam ban tay model tim duoc lech khoi vi tri Quest bao (hint) qua bao nhieu lan
# kich thuoc ban tay thi coi la bam NHAM (vd tay trai). Do tren 2 ban ghi: khung
# dung lech 0.25-0.36 (p90), toi da 0.51; bam nham tay trai lech 1.5-3.6.
HINT_MAX_OFFSET = 1.0


TAPE_HINT_MAX_OFFSET = 2.5  # co bang keo mau o dau ngon: chap nhan lech Quest toi muc nay


def tape_near_tips(frame, kpts):
    """Quanh dau ngon cai (diem 4) co bang CAM, hoac quanh dau ngon tro (diem 8) co bang XANH
    khong -- dau hieu chac chan day la tay deo gang (cung nguong mau voi color_label_glove.py)."""
    palm = palm_length(kpts)
    r = int(max(12, 0.25 * palm))
    for idx, color in ((4, "orange"), (8, "blue")):
        x, y = int(kpts[idx][0]), int(kpts[idx][1])
        x0, y0 = max(0, x - r), max(0, y - r)
        x1, y1 = min(frame.shape[1], x + r), min(frame.shape[0], y + r)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        h, s, v = (hsv[..., i].astype(np.int32) for i in range(3))
        if color == "blue":
            m = (h >= 92) & (h <= 105) & (s >= 90) & (s <= 200) & (v >= 70)
        else:
            m = ((h <= 14) | (h >= 172)) & (s >= 120) & (v >= 55)
        if m.mean() > 0.04:
            return True
    return False


# Cum mieng luc giac mau be tren mu ban tay gang (tren khop goc 3 ngon dang nam). 9 diem co tay /
# ngon cai / ngon tro gan nhu khong cho biet ban tay LAT bao nhieu -> Unity (ImageHandSolver) dung
# tam cum nay nhu diem thu 10, va "khong thay cum" = mu ban tay khong quay ve camera.
# Mau do tren ban ghi 01/10 17:45: luc giac H 11-30, S ~40-85, V sang; da tay tran dam mau hon
# (S 80-105) va toi hon; vai gang trang ngả xanh (H ~103).
TILE_HSV_LO = (8, 25, 95)
TILE_HSV_HI = (28, 80, 255)
TILE_MIN_AREA = 0.05  # dien tich cum / (chieu dai long ban tay)^2


def _seg_dist(px, a, b):
    ab = b - a
    t = np.clip(((px - a) @ ab) / max(float(ab @ ab), 1e-6), 0, 1)
    return np.linalg.norm(px - (a + t[:, None] * ab), axis=1)


def detect_dorsal_tiles(frame, kpts):
    """Tim cum luc giac quanh ban tay gang (KHONG xet toan anh -- tay trai tran cung mau gan giong).
    Tra ve (tim_thay, cx, cy, dien_tich / palm^2). Goi TRUOC khi ve skeleton len frame."""
    k = np.asarray(kpts, dtype=float)
    p0, p5 = k[0], k[5]
    L = float(np.linalg.norm(p5 - p0))
    if L < 20:
        return False, 0.0, 0.0, 0.0
    u = (p5 - p0) / L
    v = np.array([-u[1], u[0]])
    h, w = frame.shape[:2]
    c = p0 + u * 0.9 * L
    r = 1.3 * L
    x0, y0 = int(max(0, c[0] - r)), int(max(0, c[1] - r))
    x1, y1 = int(min(w, c[0] + r)), int(min(h, c[1] + r))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return False, 0.0, 0.0, 0.0
    mask = cv2.inRange(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV), TILE_HSV_LO, TILE_HSV_HI)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return False, 0.0, 0.0, 0.0
    px = np.stack([xs + x0, ys + y0], 1).astype(float)
    rel = px - p0
    a, b = rel @ u, rel @ v
    keep = (a > 0.35 * L) & (a < 1.7 * L) & (np.abs(b) < 1.1 * L)
    for chain in ((1, 2, 3, 4), (5, 6, 7, 8)):  # bo cac mieng dem tren ngon cai / ngon tro
        for i, j in zip(chain, chain[1:]):
            keep &= _seg_dist(px, k[i], k[j]) > 0.14 * L
    m2 = np.zeros_like(mask)
    sel = px[keep].astype(int)
    m2[sel[:, 1] - y0, sel[:, 0] - x0] = 255
    m2 = cv2.morphologyEx(m2, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(m2)
    if n <= 1:
        return False, 0.0, 0.0, 0.0
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[best, cv2.CC_STAT_AREA] / (L * L)
    if area < TILE_MIN_AREA:
        return False, 0.0, 0.0, float(area)
    cx, cy = cent[best]
    return True, float(cx + x0), float(cy + y0), float(area)


def hint_offset(kpts, hint):
    """Khoang cach tam long ban tay (co tay + 4 khop goc ngon) toi hint, chia kich thuoc hint."""
    hx, hy, hsize = hint
    center = np.asarray(kpts)[[0, 5, 9, 13, 17], :2].mean(axis=0)
    return float(np.linalg.norm(center - np.array([hx, hy])) / max(hsize, 1.0))


# Anh nhoe: ban ghi 03/10 14:22 -- 2 khung model "thay tay" tren anh san nha nhoe (quay dau) co do net 150-217,
# trong khi khung tot cua 3 ban ghi thap nhat 302 (canh it hoa van, vd tuong tron, ~390 van khop tot).
BLUR_THRESHOLD = 280.0


def image_sharpness(frame):
    """Do net ca anh: phuong sai Laplacian cua anh xam thu nho 320x240 (~1 ms). Anh nhoe chuyen dong -> thap."""
    g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 240), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def is_valid_hand(kpts, scores, conf_thr=0.25, frame_shape=None):
    """Tra ve (hop_le, do_tin_cay_loi, ly_do_loai).

    CAC NGUONG DA DUOC NOI RONG so voi ban goc: chung duoc viet cho webcam de
    ban (tay nho, nhin truc dien, luon cach camera mot khoang on dinh). Camera
    gio gan TREN DAU: tay co the sat ngay truoc mat va nhin nghieng, luc do cac
    khop chieu xuong anh 2D chong len nhau nen khoang cach pixel co lai rat
    nhieu -- cac dieu kien cu loai oan chinh nhung khung hinh tot.

    Viec noi long o day an toan hon truoc, vi da co is_pose_plausible() kiem
    tra HINH DANG ky hon (chieu dai tung dot, tong chieu dai ngon, nhay dot
    ngot) o buoc sau."""
    wrist = kpts[0]
    # Chi xet co tay + goc ngon cai/tro -- 3 ngon giua/ap ut/ut model gang moi khong hoc
    core_scores = scores[[0, 1, 5]]
    core_conf = float(np.mean(core_scores))
    if core_conf < conf_thr:
        return False, core_conf, f"do tin cay {core_conf:.2f} < {conf_thr:.2f}"

    palm_len = palm_length(kpts)
    if palm_len < 15.0:  # truoc: 30.0
        return False, core_conf, f"long ban tay qua nho ({palm_len:.0f}px)"

    # Kich thuoc so voi KHUNG HINH (khong phai pixel tuyet doi) -- bat truong
    # hop khoa nham vao mot chi tiet nho trong anh.
    if frame_shape is not None:
        frame_min = float(min(frame_shape[0], frame_shape[1]))
        ratio = palm_len / max(frame_min, 1.0)
        if ratio < MIN_PALM_FRAME_RATIO:
            return False, core_conf, (f"tay qua nho so voi khung hinh "
                                      f"({ratio * 100:.0f}% < {MIN_PALM_FRAME_RATIO * 100:.0f}%)")

    # (Bo kiem tra be ngang ban tay |17-5|: can diem ngon ut, model gang moi khong hoc.)

    thumb_tip_dist = np.linalg.norm(kpts[4] - kpts[1])
    if thumb_tip_dist > palm_len * 2.6:  # truoc: 2.0
        return False, core_conf, f"ngon cai qua dai ({thumb_tip_dist / palm_len:.1f}x)"

    return True, core_conf, ""


def draw_skeleton(img, kpts, scores, score_thr=0.20, thickness=3):
    for bone_idx, (start, end) in enumerate(HAND_SKELETON):
        if MRP_ALWAYS_CLOSED and bone_idx // 4 >= 2:
            continue  # bo qua 3 ngon giua/ap ut/ut
        if scores[start] > score_thr and scores[end] > score_thr:
            pt1 = (int(kpts[start][0]), int(kpts[start][1]))
            pt2 = (int(kpts[end][0]), int(kpts[end][1]))
            color = FINGER_COLORS[min(bone_idx // 4, 4)]
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
    for i, (x, y) in enumerate(kpts):
        if MRP_ALWAYS_CLOSED and i >= 9:
            continue
        if scores[i] > score_thr:
            pt = (int(x), int(y))
            color = (255, 255, 255) if i == 0 else FINGER_COLORS[min((i - 1) // 4, 4)]
            cv2.circle(img, pt, max(3, thickness + 2), color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, max(4, thickness + 3), (20, 20, 20), 1, cv2.LINE_AA)


def draw_protocol(img, state, step, seconds_left, frames_saved):
    """Chu TO o giua-tren man hinh: dang lam dong tac nao, con bao nhieu giay
    (du to de doc duoc qua passthrough khi dang deo kinh)."""
    h, w = img.shape[:2]
    if state == "wait":
        line1, color = f"CHUAN BI: dua tay ra truoc mat  ({seconds_left:.0f})", (0, 200, 255)
        line2 = f"Dau tien: {step[1]}"
    elif state == "rest":
        line1, color = f"NGHI - tiep theo ({seconds_left:.0f})", (0, 200, 255)
        line2 = step[1]
    elif state == "step":
        line1, color = f"DANG GHI: {step[0].upper()}  ({seconds_left:.0f})", (0, 0, 255)
        line2 = step[1]
    else:
        line1, color = "XONG KICH BAN - bam q de thoat", (0, 255, 0)
        line2 = f"Da ghi {frames_saved} khung"
    cv2.rectangle(img, (0, 110), (w, 200), (0, 0, 0), -1)
    cv2.putText(img, line1, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)
    cv2.putText(img, line2, (20, 186), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_hud(img, thumb_joints, index_joints, pinch, fps, sent_ok, spread_amount=0.0, spread_deg=0.0):
    h, w = img.shape[:2]
    panel_w = 230
    x0, y0 = w - panel_w - 15, 18
    overlay = img.copy()
    cv2.rectangle(overlay, (x0 - 10, y0 - 6), (w - 10, y0 + 218), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.putText(img, "QUEST CAMERA -> UNITY UDP", (x0, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    status_color = (0, 255, 0) if sent_ok else (0, 0, 255)
    cv2.putText(img, f"Sending: {'OK' if sent_ok else 'HAND NOT FOUND'}", (x0, y0 + 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1, cv2.LINE_AA)
    # Goc that tung khop (goc/giua/dau) -- so sanh truc tiep voi dang tay
    # trong kinh de hieu chuan MAX_BEND_DEG / _amplitudeDegrees ben Unity.
    cv2.putText(img, f"Thumb  goc:{thumb_joints[0]:.2f} giua:{thumb_joints[1]:.2f} dau:{thumb_joints[2]:.2f}",
                (x0, y0 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.38, FINGER_COLORS[0], 1)
    cv2.putText(img, f"Index  goc:{index_joints[0]:.2f} giua:{index_joints[1]:.2f} dau:{index_joints[2]:.2f}",
                (x0, y0 + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.38, FINGER_COLORS[1], 1)
    # Do XOE cai<->tro: bac tu do tao ra hinh "chu C" (2 ngon thang nhung cach nhau).
    cv2.putText(img, f"SPREAD (xoe cai<->tro): {spread_amount:.2f}  ({spread_deg:.0f} do)",
                (x0, y0 + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 255, 120), 1)
    cv2.putText(img, f"Pinch: {pinch:.2f}", (x0, y0 + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 220, 255), 1)
    cv2.putText(img, "Mid/Ring/Pinky: fixed closed", (x0, y0 + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1)
    cv2.putText(img, f"FPS: {fps:.1f}", (x0, y0 + 168), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1)
    cv2.putText(img, f"MAX_BEND_DEG={MAX_BEND_DEG:.0f}  MAX_SPREAD_DEG={MAX_SPREAD_DEG:.0f}",
                (x0, y0 + 190), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 150), 1)


def start_discovery_beacon(tcp_port, discovery_port):
    """Phat tin hieu "may chu o day" ra mang noi bo moi giay, de kinh TU TIM
    THAY may tinh ma khong can dien IP tay (va khong phai build lai moi lan
    doi mang).

    Chi chay luc ket noi: sau khi kinh da mo duoc ket noi TCP, co che nay nam
    HOAN TOAN NGOAI duong truyen du lieu -- khong them do tre nao cho moi
    khung hinh. Goi tin ~40 byte/giay, khong dang ke so voi luong video."""
    payload = f"GLOVE_SERVER|{tcp_port}".encode("utf-8")

    def run():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        while True:
            # Gui ca broadcast toan cuc lan broadcast cua tung mang con --
            # mot so mang/router chan cai nay nhung cho cai kia qua.
            targets = {"255.255.255.255"}
            try:
                for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
                    octets = ip.split(".")
                    if len(octets) == 4:
                        targets.add(".".join(octets[:3] + ["255"]))
            except Exception:
                pass
            for target in targets:
                try:
                    sock.sendto(payload, (target, discovery_port))
                except Exception:
                    pass  # mang chan broadcast -- van chay duoc neu dien IP tay
            time.sleep(1.0)

    threading.Thread(target=run, daemon=True).start()


def _recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Ket noi voi kinh bi dong.")
        buf += chunk
    return buf


# --- Quet tim tay tren TOAN khung hinh --------------------------------------
# Model nay thuoc loai "top-down": no KHONG tu do tim tay trong ca anh, ma phai
# duoc chi san mot khung de nhin vao. Truoc day ta chi chi cho no 2 vung co
# dinh o giua-duoi man hinh (tan du tu thoi dung webcam de ban, tay luon o
# giua). Nhung camera gio gan TREN DAU, tay co the o bat ky dau trong tam nhin
# -- dua tay ra ria trai/phai hay len tren la model khong he nhin toi, mat
# tracking hoan toan.
#
# Giai phap: quet ca khung hinh bang mot luoi o vuong chong lan nhau. De khong
# lam tut FPS, moi khung hinh chi thu vai o (luan phien) -- va viec quet nay
# chi chay khi DANG MAT tay, con khi da bam duoc thi dung khung da theo doi.
SEARCH_BOXES_PER_FRAME = 3    # so o luoi thu moi khung hinh khi dang tim
REACQUIRE_EXPAND = 1.8        # he so nong rong khung cu khi vua moi mat tay

# NHIEU CO O, khong phai mot co duy nhat. Ly do: neu o luon nho hon ban tay
# thi moi o chi chua MOT PHAN tay (vai ngon), model co nhet ca bo 21 khop vao
# manh do va sup do -- xuat ra mot cum diem chong khit len nhau, nhung van bao
# do tin cay CAO. Trieu chung dac trung: "long ban tay = 0px" du tay dang
# chiem nua khung hinh.
#   0.45 -> tay o xa, nho trong khung
#   0.70 -> co thong thuong
#   0.95 -> tay sat camera, chiem gan het khung hinh
SEARCH_BOX_SCALES = (0.45, 0.70, 0.95)

# Cho phep khung bao TRAN RA NGOAI anh (phan tran duoc to xam). Camera gan
# tren dau nen tay hay nam sat MEP DUOI anh, co tay bi cat ra ngoai. Neu ep
# khung nam trong anh, khung bi xen meo, khong om tron ban tay, model don 21
# diem thanh mot cum nho va bi loai ("long ban tay qua nho") du tay ro rang.
# Chay lai 1030 khung ghi that (recordings/20260925_140855): nhan ra tay
# 85% -> 89% khung, khong nhan nham them luc khong co tay.
EDGE_PAD_RATIO = 0.25

# Phan dau 24 byte truoc anh JPEG (GloveLiveStreamer.BuildHeader): "GLV1",
# uint32 so thu tu anh, float32 tam tay x/y (pixel, goc tren-trai), float32 kich
# thuoc tay (pixel, ~chieu dai long ban tay), uint8 hop le, 3 byte trong.
HINT_MAGIC = b"GLV1"
HINT_HEADER = struct.Struct(">4sIfffB3x")
# Khung tim quanh vi tri goi y: canh = HINT_BOX_SCALE x chieu dai long ban tay
# (ngon duoi thang dai ~1 long ban tay tinh tu goc ngon, ve moi phia).
HINT_BOX_SCALE = 2.8


# --- Thu du lieu theo kich ban (xem --protocol) ------------------------------
# Moi buoc: (ma, huong dan hien tren man hinh, so giay). Giua 2 buoc nghi
# PROTOCOL_REST_SECONDS giay (khong ghi). Tieng bip: 1 tieng = bat dau lam,
# 2 tieng = dung/nghi, 3 tieng = xong ca kich ban.
PROTOCOL_SETS = {
    # Cac dong tac co ban -- de do tung dong tac bam tot toi dau.
    "basic": [
        ("xoe_mu",     "XOE TAY - MU BAN TAY huong ve mat, giu yen",          8),
        ("xoe_long",   "XOE TAY - LONG BAN TAY huong ve mat, giu yen",        8),
        ("lat",        "LAT QUA LAT LAI cham rai (mu <-> long ban tay)",      15),
        ("nam",        "NAM TAY - mu ban tay huong ve mat, giu yen",          8),
        ("nam_xoe",    "NAM - XOE lien tuc, cham rai",                        15),
        ("pinch_giu",  "PINCH (cai cham tro) - giu yen",                      8),
        ("pinch_lap",  "PINCH - THA lien tuc, cham rai",                      15),
        ("pinch_xoay", "PINCH va XOAY co tay cham rai",                       15),
    ],
    # Giong CACH DUNG THAT: tay di chuyen, nghieng, ra ria tam nhin, dang cam
    # bong. Bo "basic" giu tay yen o giua tam nhin nen lac quan hon thuc te --
    # khi dung that van mat dau luc pinch roi dua bong di, hay xoe tay hoi nghieng.
    # Nen thu KHI DANG O TRONG UNG DUNG, pinch vao qua bong that.
    "pinch": [
        ("xoe_nghieng",  "XOE TAY, nghieng nhe qua lai (trai-phai, truoc-sau)",          15),
        ("xoe_di",       "XOE TAY, dua cham khap tam nhin: trai, phai, len, xuong",      15),
        ("pinch_di",     "PINCH qua bong, dua bong cham di khap noi (trai, phai, len, xuong)", 20),
        ("pinch_gan_xa", "PINCH qua bong, dua lai GAN mat roi ra XA",                    15),
        ("pinch_xoay",   "PINCH qua bong, XOAY co tay: lat, nghieng len-xuong, trai-phai", 20),
        ("pinch_lap_di", "PINCH - THA lien tuc trong khi di chuyen tay",                 15),
        ("pinch_lech",   "PINCH qua bong, NHIN sang cho khac (tay o ria tam nhin)",      15),
    ],
    # Rieng NAM TAY o nhieu goc -- dong tac model goc bam kem nhat (ban ghi
    # 20260926_125425_protocol: gan nhu khong khung nam tay that nao duoc nhan).
    # Thu de lay du lieu gan nhan / fine-tune.
    "fist": [
        ("nam_mu",       "NAM TAY - MU ban tay huong ve mat, giu, nghieng nhe",   10),
        ("nam_long",     "NAM TAY - LONG ban tay huong ve mat, giu",              10),
        ("nam_canh",     "NAM TAY - CANH ban tay (ngon cai o tren) ve mat",       10),
        ("nam_lat",      "NAM TAY va LAT qua lat lai cham rai",                   15),
        ("nam_xoe_cham", "NAM - XOE that cham (khoang 3 giay moi lan)",           15),
        ("nam_di",       "NAM TAY, dua tay khap tam nhin: gan, xa, trai, phai",   15),
    ],
}
PROTOCOL_STEPS = PROTOCOL_SETS["basic"]
PROTOCOL_REST_SECONDS = 4.0
PROTOCOL_START_DELAY = 8.0   # sau khi kinh ket noi: thoi gian de dua tay vao vi tri


class Protocol:
    """Dieu phoi kich ban thu du lieu theo thoi gian, co tieng bip bao hieu."""

    def __init__(self, repeats, start_time, steps=None):
        steps = steps or PROTOCOL_STEPS
        self.steps = [s for _ in range(max(1, repeats)) for s in steps]
        self.t0 = start_time + PROTOCOL_START_DELAY
        self._last_phase = None

    def phase(self, t):
        """Tra ve (trang_thai, buoc, giay_con_lai) voi trang_thai la
        'wait' / 'rest' / 'step' / 'done'."""
        if t < self.t0:
            return "wait", self.steps[0], self.t0 - t
        elapsed = t - self.t0
        for i, step in enumerate(self.steps):
            if i > 0:
                if elapsed < PROTOCOL_REST_SECONDS:
                    return "rest", step, PROTOCOL_REST_SECONDS - elapsed
                elapsed -= PROTOCOL_REST_SECONDS
            if elapsed < step[2]:
                return "step", step, step[2] - elapsed
            elapsed -= step[2]
        return "done", None, 0.0

    def beep_on_change(self, state, step_idx_key):
        key = (state, step_idx_key)
        if key == self._last_phase:
            return
        self._last_phase = key
        pattern = {"step": [880], "rest": [660, 660], "done": [990, 990, 990]}.get(state)
        if pattern:
            threading.Thread(target=_beep, args=(pattern,), daemon=True).start()


def _beep(freqs):
    try:
        import winsound
        for f in freqs:
            winsound.Beep(f, 160)
            time.sleep(0.08)
    except Exception:
        print("\a", end="", flush=True)
SEARCH_STEP_RATIO = 0.5       # buoc nhay = 50% canh o -> cac o chong lan nhau


def build_search_grid(w, h):
    """Cac o phu kin TOAN khung hinh o NHIEU CO khac nhau, co chong lan de tay
    nam vat giua 2 o van duoc bat.

    Cac o duoc XEN KE theo co (co vua truoc, roi to, roi nho) thay vi quet het
    co nay moi sang co khac -- vi moi khung hinh ta chi thu vai o, xen ke giup
    thu du cac co som hon thay vi phai doi quet xong ca mot co."""
    per_scale = []
    for scale in SEARCH_BOX_SCALES:
        side = scale * min(w, h)
        step = max(SEARCH_STEP_RATIO * side, 1.0)
        boxes = []
        y = 0.0
        while y < h:
            x = 0.0
            while x < w:
                bx1 = min(x, max(0.0, w - side))
                by1 = min(y, max(0.0, h - side))
                boxes.append(np.array([bx1, by1, bx1 + side, by1 + side]))
                x += step
            y += step
        per_scale.append(boxes)

    # Xen ke: lay lan luot 1 o tu moi co cho den khi het.
    interleaved = []
    for i in range(max(len(b) for b in per_scale)):
        for boxes in per_scale:
            if i < len(boxes):
                interleaved.append(boxes[i])
    return interleaved


def expand_box(box, factor, w, h, pad=0):
    """Nong rong mot khung quanh tam cua no -- dung de tim lai tay ngay quanh
    vi tri cu (tay thuong chi vua dich di mot chut). Khung duoc phep tran ra
    ngoai anh toi `pad` pixel (xem EDGE_PAD_RATIO)."""
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    half_w = (box[2] - box[0]) * 0.5 * factor
    half_h = (box[3] - box[1]) * 0.5 * factor
    return np.array([
        max(-pad, cx - half_w), max(-pad, cy - half_h),
        min(float(w + pad), cx + half_w), min(float(h + pad), cy + half_h),
    ])


class ImageWriter:
    """Ghi anh --record o LUONG PHU: imwrite JPEG q95 mat ~19 ms/anh, truoc day nam ngay trong vong
    lap chinh nen moi anh Unity phai cho them chung ay. OpenCV nha GIL khi nen anh -> chay song song that."""

    def __init__(self, max_pending=240):
        import queue
        self._q = queue.Queue(maxsize=max_pending)
        self.dropped = 0
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def put(self, path, img):
        try:
            self._q.put_nowait((path, img))
        except Exception:  # hang doi day (dia qua cham) -> bo anh nay, khong lam cham vong lap chinh
            self.dropped += 1

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            cv2.imwrite(item[0], item[1], [cv2.IMWRITE_JPEG_QUALITY, 95])

    def close(self):
        self._q.put(None)
        self._t.join(timeout=30)
        if self.dropped:
            print(f"[RECORD] Bo {self.dropped} anh vi dia ghi khong kip")


class LatestFrameReader:
    """Doc khung hinh tu kinh, LUON tra ve khung MOI NHAT va vut bo khung cu.

    Vi sao can: kinh gui ~20 khung/giay nhung may tinh chi xu ly kip ~12
    khung/giay. TCP KHONG BAO GIO lam mat du lieu -- no XEP HANG. Nen moi
    giay lai du ra vai khung, don lai trong bo dem, va may tinh luon xu ly
    anh CU. Do tre vi the TANG DAN theo thoi gian chay (sau 10 giay co the
    da tre vai giay), chu khong phai mot hang so.

    Vut bo cac khung ton dong la cach duy nhat giu do tre thap: ta chi quan
    tam tay THAT dang o dau NGAY BAY GIO, khung cu khong con gia tri gi."""

    def __init__(self, conn, timeout=5.0):
        self.conn = conn
        self.buf = bytearray()
        self.timeout = timeout
        self.last_meta = None  # {"fid": so thu tu anh, "hint": (x, y, kich_thuoc) hoac None}

    def _extract_frames(self):
        """Tach cac khung HOAN CHINH dang co trong bo dem, giu lai phan du."""
        frames = []
        while len(self.buf) >= 4:
            length = struct.unpack(">I", self.buf[:4])[0]
            if len(self.buf) < 4 + length:
                break
            frames.append(bytes(self.buf[4:4 + length]))
            del self.buf[:4 + length]
        return frames

    def read_latest(self):
        """Tra ve (anh_moi_nhat, so_khung_da_vut_bo)."""
        # 1. Cho cho den khi co it nhat 1 khung hoan chinh.
        frames = self._extract_frames()
        while not frames:
            chunk = self.conn.recv(65536)
            if not chunk:
                raise ConnectionError("Ket noi voi kinh bi dong.")
            self.buf.extend(chunk)
            frames = self._extract_frames()

        # 2. Hut sach nhung gi DA co san trong hang doi (khong cho doi), de
        # biet co khung nao moi hon dang ton dong hay khong.
        self.conn.setblocking(False)
        try:
            while True:
                try:
                    chunk = self.conn.recv(65536)
                except BlockingIOError:
                    break  # het du lieu san co -- day la truong hop binh thuong
                if not chunk:
                    raise ConnectionError("Ket noi voi kinh bi dong.")
                self.buf.extend(chunk)
        finally:
            self.conn.settimeout(self.timeout)  # tra ve che do chan co timeout

        frames.extend(self._extract_frames())

        dropped = len(frames) - 1
        jpg_bytes = frames[-1]  # CHI lay khung moi nhat

        # Phan dau tuy chon (GloveLiveStreamer ban moi): "GLV1" + so thu tu anh +
        # vi tri/kich thuoc ban tay tren anh (chieu tu co tay Quest bam duoc).
        # Kinh ban cu khong gui -> ca goi tin la JPEG nhu truoc.
        self.last_meta = None
        if jpg_bytes[:4] == HINT_MAGIC and len(jpg_bytes) > HINT_HEADER.size:
            _, fid, hx, hy, hsize, valid = HINT_HEADER.unpack(jpg_bytes[:HINT_HEADER.size])
            self.last_meta = {"fid": fid, "hint": (hx, hy, hsize) if valid and hsize > 1 else None}
            jpg_bytes = jpg_bytes[HINT_HEADER.size:]
        frame = cv2.imdecode(np.frombuffer(jpg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            # Anh tu kinh dang bi nguoc tren-duoi -- lat lai o day (nhanh hon
            # phai sua GloveLiveStreamer.cs._imageIsBottomUp roi build lai APK).
            frame = cv2.flip(frame, 0)
        return frame, dropped


def main():
    global CHECKPOINT_FILE
    parser = argparse.ArgumentParser(description="Quest camera -> Glove tracker -> Unity UDP bridge")
    parser.add_argument("--port", type=int, default=5007, help="TCP port to listen for the Quest's camera stream")
    parser.add_argument("--discovery-port", type=int, default=5008,
                        help="UDP port used to broadcast this PC's presence so the Quest finds it without a hardcoded IP")
    parser.add_argument("--udp-port", type=int, default=5005, help="Target UDP port on the Quest (must match FingerUDPReceiver's _port)")
    parser.add_argument("--blur-thr", type=float, default=BLUR_THRESHOLD,
                        help="Anh nhoe (quay dau nhanh) -> bo qua, khong chay model. Do net = phuong sai Laplacian cua ca "
                             "anh thu nho 320x240. 0 = tat.")
    parser.add_argument("--conf-thr", type=float, default=0.15,
                        help="Min core-joint confidence to accept a detection. Lowered from 0.26: the model has never "
                             "seen this glove so its confidence sits near the threshold, and a cluttered background "
                             "(e.g. a monitor) pushes it under. Shape checks below reject bad poses independently.")
    parser.add_argument(
        "--original",
        action="store_true",
        help="Force the ORIGINAL (non-fine-tuned) checkpoint -- found to track thumb/index on the "
             "glove more stably than the fine-tuned one; middle/ring/pinky are ignored downstream.",
    )
    if torch.cuda.is_available():
        _default_device = "cuda"
    elif torch.backends.mps.is_available():
        _default_device = "mps"
    else:
        _default_device = "cpu"
    parser.add_argument("--device", type=str, default=_default_device, help="Device: 'cuda', 'mps' or 'cpu'")
    parser.add_argument("--checkpoint", default="",
                        help="Duong dan checkpoint rieng (vd checkpoints/rtmpose_glove_pinch.pth tu finetune_pinch.py). "
                             "Uu tien hon --original.")
    parser.add_argument(
        "--record",
        action="store_true",
        help="Luu ANH GOC cua tung khung + ket qua nhan dang (tim thay tay? vi sao bi loai?) vao "
             "recordings/<thoi gian>/ -- de phan tich sau vi sao mat dau tay, va thu cach sua tren "
             "chinh nhung khung do ma khong can deo kinh lai.",
    )
    parser.add_argument(
        "--flip", action="store_true",
        help="Bat flip test cua mmpose (chay them anh lat ngang roi lay trung binh). Mac dinh TAT tu 02/10: "
             "model 47 -> 25 ms/anh tren GPU, diem lech trung vi 0.009 long ban tay.")
    parser.add_argument(
        "--dorsal", action="store_true",
        help="Tim cum luc giac tren mu ban tay (18 ms/anh) va gui cho Unity. Mac dinh TAT tu 02/10: "
             "Unity da tat dung tin hieu nay (ImageHandSolver._useDorsalTiles).")
    parser.add_argument(
        "--protocol",
        action="store_true",
        help="Thu du lieu theo KICH BAN co san (xoe tay mu/long, lat tay, nam, pinch...), co tieng bip bao "
             "bat dau/ket thuc tung dong tac. CHI ghi luc dang lam dong tac, moi khung ghi kem ten dong tac "
             "va 21 diem model doan. Tu bat --record. Xem PROTOCOL_STEPS.",
    )
    parser.add_argument("--protocol-repeats", type=int, default=2,
                        help="So lan lap lai ca kich ban (mac dinh 2).")
    parser.add_argument("--protocol-set", choices=sorted(PROTOCOL_SETS), default="basic",
                        help="Bo dong tac: 'basic' (xoe/lat/nam/pinch) hoac 'fist' (nam tay o nhieu goc).")
    args = parser.parse_args()
    if args.protocol:
        args.record = True
        steps = PROTOCOL_SETS[args.protocol_set]
        total = args.protocol_repeats * (sum(s[2] for s in steps) + PROTOCOL_REST_SECONDS * len(steps))
        print(f"[PROTOCOL] Bo '{args.protocol_set}': {args.protocol_repeats} lan x {len(steps)} dong tac, "
              f"khoang {total / 60:.1f} phut. Bat dau {PROTOCOL_START_DELAY:.0f} s sau khi kinh ket noi.")
        print("  1 bip = BAT DAU lam | 2 bip = NGHI, chuan bi dong tac sau | 3 bip = XONG")
        for i, (_, text, sec) in enumerate(steps, 1):
            print(f"  {i}. {text} ({sec} s)")

    if args.original:
        CHECKPOINT_FILE = _ORIGINAL_CKPT_URL
    if args.checkpoint:
        CHECKPOINT_FILE = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(_BASE_DIR, args.checkpoint)

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    ckpt_source = ("ORIGINAL (bare-hand only)" if CHECKPOINT_FILE == _ORIGINAL_CKPT_URL
                   else os.path.basename(CHECKPOINT_FILE))
    print(f"Loading RTMPose-m on {args.device.upper()}... [{ckpt_source}]")
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=args.device)
    if not args.flip:
        model.test_cfg["flip_test"] = False  # config bat san; tat = nhanh gap doi (xem --flip)
    print(f"Flip test: {'BAT' if args.flip else 'tat'} | Cum luc giac mu tay: {'BAT' if args.dorsal else 'tat'}")

    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    dummy_box = np.array([[400, 200, 880, 680]])
    _ = inference_topdown(model, dummy_img, bboxes=dummy_box)
    if args.device == "mps":
        torch.mps.synchronize()

    hostname = socket.gethostname()
    try:
        local_ips = socket.gethostbyname_ex(hostname)[2]
    except Exception:
        local_ips = ["(khong lay duoc tu dong, dung 'ipconfig' trong PowerShell)"]

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(1)

    start_discovery_beacon(args.port, args.discovery_port)

    print(f"\n[READY] Dang lang nghe stream tu kinh tren cong {args.port}.")
    print(f"[AUTO-DISCOVERY] Dang phat tin hieu tren cong UDP {args.discovery_port} moi giay "
          f"-- kinh se TU TIM THAY may nay, khong can dien IP.")
    print(f"IP may nay (chi can neu tat auto-discovery, dien tay vao Unity): {local_ips}")
    print("Nhan Ctrl+C de dung.\n")

    win_name = "Quest Camera -> Unity Bridge"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            print("Cho kinh ket noi...")
            conn, addr = srv.accept()
            quest_ip = addr[0]
            conn.settimeout(5.0)
            print(f"[CONNECTED] Kinh ({quest_ip}) da ket noi. Gui ket qua UDP nguoc lai {quest_ip}:{args.udp_port}")

            filter_kpts = None
            state = "SEARCHING"
            tracked_box = None
            lost_frames = 0
            prev_time = time.time()
            fps = 0.0
            pinch_amount = 0.0
            prev_kpts = None          # khung truoc da duoc chap nhan (de bat cu nhay dot ngot)
            fallback_reason = ""      # ly do dang fallback, hien len HUD
            last_good_msg = None      # goi tin tot cuoi cung, dung cho an han
            reader = LatestFrameReader(conn)

            # Ghi lai (xem --record): moi lan kinh ket noi la 1 thu muc moi.
            rec_dir, rec_log, rec_kpts, rec_idx = None, None, None, 0
            rec_writer = None
            protocol = (Protocol(args.protocol_repeats, time.time(), PROTOCOL_SETS[args.protocol_set])
                        if args.protocol else None)
            if args.record:
                suffix = f"_protocol_{args.protocol_set}" if protocol else ""
                rec_dir = os.path.join(_BASE_DIR, "recordings", time.strftime("%Y%m%d_%H%M%S") + suffix)
                os.makedirs(rec_dir, exist_ok=True)
                rec_log = open(os.path.join(rec_dir, "log.csv"), "w", encoding="utf-8", buffering=1)
                rec_log.write("frame,t,pose,found,state,best_conf,sent,reject_reason,fallback_reason,fid,hint\n")
                # 21 diem model doan cho tung khung da ghi -- de phan tich va de
                # dien san khi gan nhan (khong can chay lai model).
                rec_kpts = open(os.path.join(rec_dir, "keypoints.jsonl"), "w", encoding="utf-8", buffering=1)
                rec_writer = ImageWriter()
                print(f"[RECORD] Dang luu khung hinh vao {rec_dir}")
            search_grid = None        # luoi o phu kin khung hinh (tao khi biet kich thuoc)
            search_idx = 0            # o dang quet toi (luan phien qua tung khung)
            last_known_box = None     # vi tri cuoi cung thay tay -- de tim lai cho nhanh
            dropped_total = 0         # tong so khung cu da vut bo (theo doi do tre)
            dropped_recent = 0.0      # trung binh truot, hien len HUD

            try:
                while True:
                    # Luon lay khung MOI NHAT, vut bo khung cu dang ton dong --
                    # neu khong, do tre se tang dan vo han (xem LatestFrameReader).
                    frame, dropped = reader.read_latest()
                    dropped_total += dropped
                    dropped_recent = 0.9 * dropped_recent + 0.1 * dropped
                    if frame is None:
                        continue
                    h, w = frame.shape[:2]

                    # Kich ban (--protocol): chi ghi luc dang lam dong tac.
                    pose = "free"
                    record_this = rec_dir is not None
                    if protocol is not None:
                        p_state, p_step, p_left = protocol.phase(time.time())
                        protocol.beep_on_change(p_state, p_step[0] if p_step else None)
                        record_this = record_this and p_state == "step"
                        pose = p_step[0] if p_state == "step" else p_state

                    # Luu anh GOC truoc khi ve khung xuong / chu len.
                    if record_this:
                        rec_idx += 1
                        # Ghi o luong phu (19 ms/anh) -- frame se bi ve len nen gui ban sao
                        rec_writer.put(os.path.join(rec_dir, f"{rec_idx:05d}.jpg"), frame.copy())

                    curr_time = time.time()
                    fps = 0.90 * fps + 0.10 * (1.0 / max(curr_time - prev_time, 1e-5))
                    prev_time = curr_time

                    if search_grid is None:
                        search_grid = build_search_grid(w, h)

                    # Anh da dem vien (xem EDGE_PAD_RATIO). Toa do khung bao va
                    # diem van tinh theo anh GOC; chi cong/tru `pad` khi goi model.
                    pad = int(EDGE_PAD_RATIO * max(w, h))
                    padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                                                value=(114, 114, 114))

                    # Danh sach khung thu theo thu tu uu tien. Vong lap ben duoi
                    # DUNG NGAY khi mot khung cho ket qua tot, nen khi dang bam
                    # on dinh thi chi ton dung 1 lan chay model -- cac phuong an
                    # du phong phia sau khong ton gi ca.
                    candidate_boxes = []

                    # 1. Khung dang theo doi (duong nhanh, dung cho hau het khung hinh).
                    if state == "TRACKING" and tracked_box is not None:
                        candidate_boxes.append(tracked_box)

                    # 2. LUON kem san phuong an du phong NGAY TRONG CUNG KHUNG HINH.
                    # Truoc day khi dang TRACKING ta chi thu dung 1 khung: neu tay
                    # vua dich manh hoac bi che, khung cu khong con om dung tay,
                    # model nhan vao vung lech va cho ra rac -- roi cu lap lai
                    # dung khung hong do suot 12 khung (~1 giay) moi chiu chuyen
                    # sang tim kiem. Gio that bai la tim lai ngay lap tuc.
                    # 1b. Khung quanh vi tri ban tay kinh GOI Y (chieu tu co tay Quest
                    # bam duoc). Khi chua bam, day la khung thu DAU TIEN -- khong phai
                    # do mo ca khung hinh (nguyen nhan cua ~58% so lan mat dau).
                    meta = reader.last_meta or {}
                    frame_fid = meta.get("fid", 0)
                    hint = meta.get("hint")
                    if hint is not None:
                        hx, hy, hsize = hint
                        half = max(hsize * HINT_BOX_SCALE, 140.0) * 0.5
                        if -pad < hx < w + pad and -pad < hy < h + pad:
                            candidate_boxes.append(np.array([
                                max(-pad, hx - half), max(-pad, hy - half),
                                min(w + pad, hx + half), min(h + pad, hy + half)]))

                    if last_known_box is not None:
                        candidate_boxes.append(expand_box(last_known_box, REACQUIRE_EXPAND, w, h, pad))

                    # 3. Quet dan TOAN khung hinh, moi khung vai o, luan phien.
                    for k in range(SEARCH_BOXES_PER_FRAME):
                        candidate_boxes.append(search_grid[(search_idx + k) % len(search_grid)])
                    search_idx = (search_idx + SEARCH_BOXES_PER_FRAME) % len(search_grid)

                    found_hand = False
                    best_kpts = None
                    best_scores = None
                    best_seen_conf = 0.0
                    # Xoa moi khung, neu khong se hien ly do CU cua khung truoc.
                    reject_reason = ""
                    fallback_reason = ""

                    # Anh nhoe vi quay dau nhanh: model van "thay tay" o cho khong co tay -> bo qua ca khung
                    sharpness = image_sharpness(frame)
                    if args.blur_thr > 0 and sharpness < args.blur_thr:
                        candidate_boxes = []
                        reject_reason = f"anh nhoe (do net {sharpness:.0f} < {args.blur_thr:.0f})"

                    for c_box in candidate_boxes:
                        bx1 = max(-pad, min(w + pad - 60, int(c_box[0])))
                        by1 = max(-pad, min(h + pad - 60, int(c_box[1])))
                        bx2 = max(bx1 + 60, min(w + pad, int(c_box[2])))
                        by2 = max(by1 + 60, min(h + pad, int(c_box[3])))
                        eval_box = np.array([[bx1 + pad, by1 + pad, bx2 + pad, by2 + pad]])

                        results = inference_topdown(model, padded, bboxes=eval_box)
                        if args.device == "mps":
                            torch.mps.synchronize()

                        if len(results) > 0 and hasattr(results[0], "pred_instances"):
                            inst = results[0].pred_instances
                            kpts = inst.keypoints[0] - pad
                            scores = inst.keypoint_scores[0]
                            valid, core_conf, why = is_valid_hand(
                                kpts, scores, conf_thr=args.conf_thr, frame_shape=frame.shape)
                            # Tay tim duoc phai nam DUNG CHO Quest bao tay gang dang o.
                            # Khong co buoc nay, model bam nham TAY TRAI (tay tran) va
                            # tay gang ao cu dong theo ngon tay trai (ban ghi
                            # 20260927_144842: 87 khung "tot" lech > 1.5 lan kich thuoc tay,
                            # trong khi khung dung chi lech <= 0.36).
                            if valid and hint is not None:
                                off = hint_offset(kpts, hint)
                                # Quest nhin gang moi hay lech xa (p90 ~1.1 kich thuoc tay) -> truoc day
                                # loai oan khung DUNG dung luc can anh nhat. Thay bang keo cam/xanh o dau
                                # ngon thi chac chan la tay gang (tay trai tran khong co) -> van nhan.
                                if off > HINT_MAX_OFFSET and not (off <= TAPE_HINT_MAX_OFFSET and tape_near_tips(frame, kpts)):
                                    valid, why = False, f"khong trung vi tri tay Quest ({off:.1f}x)"
                            if not valid and core_conf >= best_seen_conf:
                                reject_reason = why  # ly do cua lan doan TOT NHAT
                            # Nho lai do tin cay CAO NHAT gap trong khung nay, ke
                            # ca khi bi loai -- de hien len HUD, biet dang cach
                            # nguong bao xa thay vi chi biet "that bai".
                            best_seen_conf = max(best_seen_conf, core_conf)
                            if valid:
                                best_kpts = kpts
                                best_scores = scores
                                found_hand = True
                                break

                    thumb_joints = [0.0, 0.0, 0.0]
                    index_joints = [0.0, 0.0, 0.0]
                    bad_points = set()
                    spread_amount, spread_deg = 0.0, 0.0

                    # Dang tay co hop ly khong? Neu khong, coi nhu khong tim thay
                    # tay: Unity se chuyen muot ve tu the nghi thay vi hien dang
                    # cong venh/lat nguoc.
                    if found_hand and best_kpts is not None:
                        bad_points = set()
                        plausible, reason = is_pose_plausible(best_kpts, best_scores, prev_kpts, bad_points)
                        if not plausible:
                            found_hand = False
                            fallback_reason = reason
                        else:
                            fallback_reason = ""
                            prev_kpts = best_kpts.copy()

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
                        palm_len = palm_length(smoothed_kpts)
                        box_side = max((max_x - min_x) * 1.35, (max_y - min_y) * 1.35, palm_len * 2.8, 140.0)
                        half = box_side / 2.0
                        new_box = np.array([max(-pad, cx - half), max(-pad, cy - half),
                                            min(w + pad, cx + half), min(h + pad, cy + half)])
                        tracked_box = 0.70 * new_box + 0.30 * tracked_box if tracked_box is not None else new_box
                        # Nho lai vi tri nay de lan sau mat tay con biet cho ma
                        # tim lai truoc, thay vi quet lai tu dau ca khung hinh.
                        last_known_box = tracked_box.copy()

                        # Cum luc giac tren mu ban tay -- tim TRUOC khi ve skeleton len frame
                        if args.dorsal:
                            dorsal_ok, dorsal_x, dorsal_y, dorsal_area = detect_dorsal_tiles(frame, smoothed_kpts)
                        else:
                            dorsal_ok, dorsal_x, dorsal_y, dorsal_area = False, 0.0, 0.0, 0.0

                        # Goc that tai tung khop (goc/giua/dau) cho rieng ngon cai(1..4)
                        # va ngon tro(5..8), tinh THANG tu hinh dang 4 diem da detect --
                        # thay the calculate_finger_flex (chi dung khoang cach dau ngon-co
                        # tay, khong phan anh dung hinh dang tung khop).
                        thumb_joints = calculate_finger_joint_bends(smoothed_kpts, 1)
                        index_joints = calculate_finger_joint_bends(smoothed_kpts, 5)
                        spread_amount, spread_deg = calculate_thumb_index_spread(smoothed_kpts)

                        pinch_dist = np.linalg.norm(smoothed_kpts[4] - smoothed_kpts[8])
                        pinch_ratio = pinch_dist / max(palm_len, 1e-4)
                        raw_pinch = (PINCH_FAR_RATIO - pinch_ratio) / (PINCH_FAR_RATIO - PINCH_CLOSE_RATIO)
                        raw_pinch = float(np.clip(raw_pinch, 0.0, 1.0))
                        pinch_amount = PINCH_SMOOTH_ALPHA * raw_pinch + (1 - PINCH_SMOOTH_ALPHA) * pinch_amount

                        # Goi tin cho Unity (FingerUDPReceiver / ImageHandSolver): 9 diem theo ti le anh + do tin
                        # cay (diem co dot sai chieu dai gui 0 -> Unity bo qua), cum luc giac tren mu ban tay.
                        send_scores = np.array(best_scores, dtype=float)
                        for i in bad_points:
                            send_scores[i] = 0.0
                        pv, pc = build_pixel_payload(smoothed_kpts, send_scores, w, h)
                        msg = f"valid:1,fid:{frame_fid},pv:{pv},pc:{pc}"
                        if dorsal_ok:
                            msg += f",dv:{dorsal_x / w:.4f}|{dorsal_y / h:.4f},da:{dorsal_area:.3f}"
                        else:
                            msg += ",da:0"
                        last_good_msg = msg.encode("utf-8")
                        udp_sock.sendto(last_good_msg, (quest_ip, args.udp_port))

                        # Ve SAU khi gui -- khong de Unity cho phan hien thi
                        draw_skeleton(frame, smoothed_kpts, best_scores)
                        if dorsal_ok:
                            cv2.circle(frame, (int(dorsal_x), int(dorsal_y)), 10, (255, 0, 255), -1)

                        bx1, by1, bx2, by2 = [int(v) for v in tracked_box]
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    else:
                        lost_frames += 1
                        if lost_frames > 12:
                            state = "SEARCHING"
                            tracked_box = None
                            filter_kpts = None
                            prev_kpts = None

                        # AN HAN: giu nguyen dang TOT CUOI CUNG them vai khung
                        # truoc khi bao "khong hop le".
                        #
                        # Ly do: tren nen roi, do tin cay cua model dao dong
                        # manh -- rat hay co 1-2 khung tut xuong duoi nguong roi
                        # len lai ngay. Neu bao "khong hop le" tuc thi, tay ao
                        # se giat lien tuc giua dang that va tu the nghi, trong
                        # con te hon la giu nguyen dang cu trong tich tac.
                        #
                        # Chi giu trong thoi gian ngan: neu mat that (tay ra khoi
                        # tam nhin) thi van phai ve tu the nghi, khong duoc giu
                        # mot dang cu sai mai.
                        if last_good_msg is not None and lost_frames <= HOLD_LAST_GOOD_FRAMES:
                            udp_sock.sendto(last_good_msg, (quest_ip, args.udp_port))
                        else:
                            # VAN PHAI GUI tin hieu, khong duoc im lang: neu khong,
                            # Unity khong biet gi va se DONG BANG o dang sai cuoi
                            # cung. "valid:0" bao Unity chuyen muot ve tu the nghi.
                            udp_sock.sendto(b"valid:0", (quest_ip, args.udp_port))

                    if record_this:
                        if found_hand:
                            sent = "good"
                        elif last_good_msg is not None and lost_frames <= HOLD_LAST_GOOD_FRAMES:
                            sent = "hold"
                        else:
                            sent = "invalid"
                        clean = lambda s: s.replace(",", ";")
                        rec_log.write(f"{rec_idx},{curr_time:.3f},{pose},{int(found_hand)},{state},{best_seen_conf:.3f},"
                                      f"{sent},{clean(reject_reason)},{clean(fallback_reason)},{frame_fid},{int(hint is not None)}\n")
                        # Diem THO cua lan doan duoc chon (ke ca khi bi loai o buoc
                        # kiem tra hinh dang) -- "found" cho biet co duoc dung khong.
                        rec_kpts.write(json.dumps({
                            "frame": rec_idx, "pose": pose, "found": bool(found_hand),
                            "kpts": None if best_kpts is None else np.round(np.asarray(best_kpts), 1).tolist(),
                            "scores": None if best_scores is None else np.round(np.asarray(best_scores), 3).tolist(),
                            "hint": None if hint is None else [round(float(x), 1) for x in hint],
                            "dorsal": [round(dorsal_x, 1), round(dorsal_y, 1), round(dorsal_area, 3)] if found_hand and dorsal_ok else None,
                        }) + "\n")

                    draw_hud(frame, thumb_joints, index_joints, pinch_amount, fps, found_hand,
                             spread_amount, spread_deg)

                    # Ly do khung hinh bi loai -- de biet CHINH XAC bo loc nao
                    # chan, thay vi chi thay "HAND NOT FOUND".
                    if not found_hand and reject_reason:
                        cv2.putText(frame, f"LOAI (kiem tra co ban): {reject_reason}",
                                    (20, h - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2, cv2.LINE_AA)
                    if fallback_reason:
                        cv2.putText(frame, f"FALLBACK (tay ao ve tu the nghi): {fallback_reason}",
                                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2, cv2.LINE_AA)

                    header_color = (0, 255, 0) if state == "TRACKING" else (0, 200, 255)
                    cv2.putText(frame, f"FPS: {fps:.1f} ({args.device.upper()}) | State: {state} <- {quest_ip}",
                                (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, header_color, 2, cv2.LINE_AA)
                    # So khung cu vut bo moi vong lap: >0 nghia la kinh gui nhanh
                    # hon may tinh xu ly (binh thuong). Mien la con vut duoc thi
                    # do tre van thap -- van de chi xay ra neu KHONG vut.
                    cv2.putText(frame, f"Bo khung cu: {dropped_recent:.1f}/vong (tong {dropped_total})",
                                (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
                    # Do tin cay cao nhat do duoc so voi nguong -- cho biet khi
                    # mat tracking thi dang "hut" bao nhieu, hay khong thay gi ca.
                    conf_color = (140, 255, 140) if best_seen_conf >= args.conf_thr else (0, 140, 255)
                    cv2.putText(frame, f"Do tin cay: {best_seen_conf:.2f} / nguong {args.conf_thr:.2f}",
                                (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.45, conf_color, 1, cv2.LINE_AA)

                    tile_txt = "THAY (cham tim)" if found_hand and dorsal_ok else "khong thay"
                    cv2.putText(frame, f"Cum luc giac mu ban tay: {tile_txt}",
                                (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 0, 255) if found_hand and dorsal_ok else (180, 180, 180), 1, cv2.LINE_AA)

                    if protocol is not None:
                        draw_protocol(frame, p_state, p_step, p_left, rec_idx)

                    cv2.imshow(win_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        return
            except (ConnectionError, socket.timeout, OSError) as e:
                print(f"[DISCONNECTED] Mat ket noi voi kinh ({quest_ip}): {e}. Cho ket noi lai...")
            finally:
                conn.close()
                if rec_log is not None:
                    rec_log.close()
                    rec_kpts.close()
                if rec_writer is not None:
                    rec_writer.close()
                    print(f"[RECORD] Da luu {rec_idx} khung vao {rec_dir}")
    except KeyboardInterrupt:
        print("\nDung lai.")
    finally:
        srv.close()
        cv2.destroyAllWindows()
        udp_sock.close()


if __name__ == "__main__":
    main()
