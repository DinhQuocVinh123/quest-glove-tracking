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
CHECKPOINT_FILE = _FINETUNED_CKPT if os.path.exists(_FINETUNED_CKPT) else _ORIGINAL_CKPT_URL

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

# Fixed bend sent for middle/ring/pinky -- not tracked live (see
# run_glove_to_unity.py docstring for why).
FIXED_MRP_BEND = 1.0

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


# --- Ep 3 ngon giua/ap ut/ut nam lai khi dang pinch --------------------------
# Khi pinch, 3 ngon nay bi chinh ban tay che khuat nen model doan rat bay
# (thieu du lieu fine-tune cho tu the do). Nhung o cac tu the mo (xoe tay, nam
# tay nhin tu mu ban tay) thi chung lo ro va model bam kha tot.
#
# Nen: cang pinch chat thi cang ep chung ve dang nam (bo qua model), con khi
# khong pinch thi de chung bam theo model nhu binh thuong. Chuyen dan giua 2
# che do de khong bi giat.
MRP_FORCE_PINCH_LOW = 0.35    # duoi muc nay: hoan toan tin model
MRP_FORCE_PINCH_HIGH = 0.60   # tren muc nay: hoan toan ep nam
MRP_MIN_CONF = 0.35           # do tin cay trung binh toi thieu cua 3 ngon do
MRP_SMOOTH_ALPHA = 0.25


def compute_mrp_close_amount(scores, pinch_amount, prev_amount):
    """Muc do ep 3 ngon giua/ap ut/ut ve dang nam (0 = tin model, 1 = ep nam)."""
    # 1. Cang pinch chat cang ep nam.
    span = max(MRP_FORCE_PINCH_HIGH - MRP_FORCE_PINCH_LOW, 1e-4)
    from_pinch = float(np.clip((pinch_amount - MRP_FORCE_PINCH_LOW) / span, 0.0, 1.0))

    # 2. Du khong pinch, neu model khong nhin ro 3 ngon do thi cung ep nam --
    # con hon de chung nhay lung tung.
    mrp_conf = float(np.mean(scores[9:21]))
    from_conf = 1.0 if mrp_conf < MRP_MIN_CONF else 0.0

    target = max(from_pinch, from_conf)
    return MRP_SMOOTH_ALPHA * target + (1.0 - MRP_SMOOTH_ALPHA) * prev_amount


def apply_mrp_closed_pose(kpts, close_amount):
    """Tra ve ban sao kpts trong do 3 ngon giua/ap ut/ut duoc gap ve phia long
    ban tay theo muc close_amount.

    Cach dung: gap ngon ve huong CO TAY. Trong anh 2D nhin tu mu ban tay, mot
    ngon dang nam se nam ep xuong long ban tay va huong nguoc ve phia co tay --
    nen day la xap xi don gian ma rat on dinh (khong can biet chieu long ban
    tay hay tay trai/phai)."""
    if close_amount <= 1e-3:
        return kpts

    out = kpts.copy()
    wrist = kpts[0]
    palm_len = max(float(np.linalg.norm(kpts[9] - wrist)), 1e-4)

    # Ty le chieu dai tung dot khi gap (so voi long ban tay).
    seg_ratios = (0.34, 0.26, 0.20)

    for base in (9, 13, 17):  # giua, ap ut, ut
        mcp = kpts[base]      # khop goc -- diem nay model van thay ro ke ca khi pinch
        to_wrist = wrist - mcp
        n = float(np.linalg.norm(to_wrist))
        if n < 1e-4:
            continue
        to_wrist = to_wrist / n

        pos = mcp.copy()
        for j, ratio in enumerate(seg_ratios):
            pos = pos + to_wrist * (ratio * palm_len)
            # Tron dan giua vi tri that (model) va vi tri gap, de khong bi giat.
            out[base + 1 + j] = (1.0 - close_amount) * kpts[base + 1 + j] + close_amount * pos

    return out


def build_points_payload(kpts):
    """Goi 21 diem THO (co tay + 4 diem moi ngon) de Unity tu xoay tung dot
    xuong chia dung huong doan backbone tuong ung.

    Day la cach duy nhat tai tao dung HINH DANG ngon tay: xoay tung khop
    quanh 1 truc co dinh (cach cu) khong the tao ra huong bat ky trong mat
    phang, nen hinh dang khong bao gio khop duoc du goc tinh co chuan.

    Toa do duoc chuan hoa: goc toa do tai CO TAY, chia cho chieu dai long
    ban tay (nen khong phu thuoc tay o gan/xa camera), va DOI DAU truc y
    (anh co y tang xuong duoi, Unity co +y huong LEN TREN).
    Dinh dang: "x|y;x|y;..." (9 diem, khong chua dau phay de khong pha
    dinh dang key:value phan cach bang dau phay cua goi UDP)."""
    wrist = kpts[0]
    palm_len = max(float(np.linalg.norm(kpts[9] - wrist)), 1e-4)
    parts = []
    for i in range(21):  # 0=co tay, 1-4=cai, 5-8=tro, 9-12=giua, 13-16=ap ut, 17-20=ut
        dx = (kpts[i][0] - wrist[0]) / palm_len
        dy = -(kpts[i][1] - wrist[1]) / palm_len
        parts.append(f"{dx:.3f}|{dy:.3f}")
    return ";".join(parts)


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
FINGER_POINT_CONF_THR = 0.18    # do tin cay toi thieu cua tung diem tren ngon cai/tro
SEG_MIN_RATIO = 0.06            # 1 dot ngan nhat = 6% chieu dai long ban tay
SEG_MAX_RATIO = 1.00            # 1 dot dai nhat = 100% chieu dai long ban tay
FINGER_MIN_RATIO = 0.35         # ca ngon ngan nhat = 35% long ban tay
FINGER_MAX_RATIO = 2.00         # ca ngon dai nhat = 200% long ban tay
MAX_JUMP_RATIO = 1.20           # 1 diem khong the nhay qua 120% long ban tay trong 1 khung


def is_pose_plausible(kpts, scores, prev_kpts=None):
    """Kiem tra rieng dang NGON CAI + NGON TRO co hop ly ve giai phau khong.

    Khac voi is_valid_hand (chi xet co tay + cac khop GOC), ham nay xet chinh
    cac diem tao nen hinh dang ngon tay -- ke ca dau ngon. Do la lo hong khien
    khung hinh co khop goc ro nhung dau ngon doan bay van lot qua.

    Tra ve (hop_ly, ly_do)."""
    wrist = kpts[0]
    palm_len = float(np.linalg.norm(kpts[9] - wrist))
    if palm_len < 1e-4:
        return False, "long ban tay ~ 0"

    # 1. Do tin cay cua TUNG diem tren 2 ngon (ke ca dau ngon).
    for i in (1, 2, 3, 4, 5, 6, 7, 8):
        if scores[i] < FINGER_POINT_CONF_THR:
            return False, f"diem {i} do tin cay thap ({scores[i]:.2f})"

    # 2. Tung dot xuong phai co chieu dai hop ly so voi long ban tay.
    for base, name in ((1, "cai"), (5, "tro")):
        total = 0.0
        for j in range(3):
            seg = float(np.linalg.norm(kpts[base + j + 1] - kpts[base + j]))
            total += seg
            if seg < SEG_MIN_RATIO * palm_len or seg > SEG_MAX_RATIO * palm_len:
                return False, f"dot {j} ngon {name} dai bat thuong"
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


def is_valid_hand(kpts, scores, conf_thr=0.25):
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
    core_scores = scores[[0, 1, 5, 9, 13, 17]]
    core_conf = float(np.mean(core_scores))
    if core_conf < conf_thr:
        return False, core_conf, f"do tin cay {core_conf:.2f} < {conf_thr:.2f}"

    palm_len = np.linalg.norm(kpts[9] - wrist)
    if palm_len < 15.0:  # truoc: 30.0
        return False, core_conf, f"long ban tay qua nho ({palm_len:.0f}px)"

    knuckle_span = np.linalg.norm(kpts[17] - kpts[5])
    if knuckle_span < 8.0:  # truoc: 20.0 -- nhin nghieng thi cac dot chum lai
        return False, core_conf, f"be ngang ban tay qua nho ({knuckle_span:.0f}px)"
    if knuckle_span > palm_len * 2.6:  # truoc: 2.2
        return False, core_conf, f"be ngang ban tay qua lon ({knuckle_span / palm_len:.1f}x)"

    thumb_tip_dist = np.linalg.norm(kpts[4] - kpts[1])
    if thumb_tip_dist > palm_len * 2.6:  # truoc: 2.0
        return False, core_conf, f"ngon cai qua dai ({thumb_tip_dist / palm_len:.1f}x)"

    return True, core_conf, ""


def draw_skeleton(img, kpts, scores, score_thr=0.20, thickness=3):
    for bone_idx, (start, end) in enumerate(HAND_SKELETON):
        if scores[start] > score_thr and scores[end] > score_thr:
            pt1 = (int(kpts[start][0]), int(kpts[start][1]))
            pt2 = (int(kpts[end][0]), int(kpts[end][1]))
            color = FINGER_COLORS[min(bone_idx // 4, 4)]
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
    for i, (x, y) in enumerate(kpts):
        if scores[i] > score_thr:
            pt = (int(x), int(y))
            color = (255, 255, 255) if i == 0 else FINGER_COLORS[min((i - 1) // 4, 4)]
            cv2.circle(img, pt, max(3, thickness + 2), color, -1, cv2.LINE_AA)
            cv2.circle(img, pt, max(4, thickness + 3), (20, 20, 20), 1, cv2.LINE_AA)


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
SEARCH_BOX_SCALE = 0.55       # canh o = 55% canh ngan cua khung hinh
SEARCH_STEP_SCALE = 0.27      # buoc nhay giua cac o (nho hon canh o -> chong lan)
REACQUIRE_EXPAND = 1.8        # he so nong rong khung cu khi vua moi mat tay


def build_search_grid(w, h):
    """Danh sach cac o phu kin TOAN khung hinh, co chong lan de tay nam vat
    giua 2 o van duoc bat."""
    side = SEARCH_BOX_SCALE * min(w, h)
    step = SEARCH_STEP_SCALE * min(w, h)
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
    return boxes


def expand_box(box, factor, w, h):
    """Nong rong mot khung quanh tam cua no -- dung de tim lai tay ngay quanh
    vi tri cu (tay thuong chi vua dich di mot chut)."""
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    half_w = (box[2] - box[0]) * 0.5 * factor
    half_h = (box[3] - box[1]) * 0.5 * factor
    return np.array([
        max(0.0, cx - half_w), max(0.0, cy - half_h),
        min(float(w), cx + half_w), min(float(h), cy + half_h),
    ])


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
    args = parser.parse_args()

    if args.original:
        CHECKPOINT_FILE = _ORIGINAL_CKPT_URL

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    ckpt_source = "FINE-TUNED (glove)" if CHECKPOINT_FILE == _FINETUNED_CKPT else "ORIGINAL (bare-hand only)"
    print(f"Loading RTMPose-m on {args.device.upper()}... [{ckpt_source}]")
    model = init_model(CONFIG_FILE, CHECKPOINT_FILE, device=args.device)

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
            reader = LatestFrameReader(conn)
            mrp_close = 0.0           # muc ep 3 ngon giua/ap ut/ut ve dang nam
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

                    curr_time = time.time()
                    fps = 0.90 * fps + 0.10 * (1.0 / max(curr_time - prev_time, 1e-5))
                    prev_time = curr_time

                    if search_grid is None:
                        search_grid = build_search_grid(w, h)

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
                    if last_known_box is not None:
                        candidate_boxes.append(expand_box(last_known_box, REACQUIRE_EXPAND, w, h))

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

                    for c_box in candidate_boxes:
                        bx1 = max(0, min(w - 60, int(c_box[0])))
                        by1 = max(0, min(h - 60, int(c_box[1])))
                        bx2 = max(bx1 + 60, min(w, int(c_box[2])))
                        by2 = max(by1 + 60, min(h, int(c_box[3])))
                        eval_box = np.array([[bx1, by1, bx2, by2]])

                        results = inference_topdown(model, frame, bboxes=eval_box)
                        if args.device == "mps":
                            torch.mps.synchronize()

                        if len(results) > 0 and hasattr(results[0], "pred_instances"):
                            inst = results[0].pred_instances
                            kpts = inst.keypoints[0]
                            scores = inst.keypoint_scores[0]
                            valid, core_conf, why = is_valid_hand(kpts, scores, conf_thr=args.conf_thr)
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
                    spread_amount, spread_deg = 0.0, 0.0

                    # Dang tay co hop ly khong? Neu khong, coi nhu khong tim thay
                    # tay: Unity se chuyen muot ve tu the nghi thay vi hien dang
                    # cong venh/lat nguoc.
                    if found_hand and best_kpts is not None:
                        plausible, reason = is_pose_plausible(best_kpts, best_scores, prev_kpts)
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
                        palm_len = np.linalg.norm(smoothed_kpts[9] - smoothed_kpts[0])
                        box_side = max((max_x - min_x) * 1.35, (max_y - min_y) * 1.35, palm_len * 2.8, 140.0)
                        half = box_side / 2.0
                        new_box = np.array([max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)])
                        tracked_box = 0.70 * new_box + 0.30 * tracked_box if tracked_box is not None else new_box
                        # Nho lai vi tri nay de lan sau mat tay con biet cho ma
                        # tim lai truoc, thay vi quet lai tu dau ca khung hinh.
                        last_known_box = tracked_box.copy()

                        draw_skeleton(frame, smoothed_kpts, best_scores)

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

                        # Khi pinch (hoac khi model khong nhin ro), ep 3 ngon
                        # giua/ap ut/ut ve dang nam thay vi tin du doan cua model.
                        # Lam ngay tren toa do diem truoc khi gui -- Unity khong
                        # can biet gi, cu bam backbone nhu binh thuong.
                        mrp_close = compute_mrp_close_amount(best_scores, pinch_amount, mrp_close)
                        send_kpts = apply_mrp_closed_pose(smoothed_kpts, mrp_close)

                        msg = (
                            f"valid:1,"
                            f"thumb:{thumb_joints[1]:.3f},index:{index_joints[1]:.3f},"
                            f"thumb0:{thumb_joints[0]:.3f},thumb1:{thumb_joints[1]:.3f},thumb2:{thumb_joints[2]:.3f},"
                            f"index0:{index_joints[0]:.3f},index1:{index_joints[1]:.3f},index2:{index_joints[2]:.3f},"
                            f"spread:{spread_amount:.3f},"
                            f"pts:{build_points_payload(send_kpts)},"
                            f"middle:{FIXED_MRP_BEND:.3f},ring:{FIXED_MRP_BEND:.3f},pinky:{FIXED_MRP_BEND:.3f},"
                            f"pinch:{pinch_amount:.3f}"
                        )
                        udp_sock.sendto(msg.encode("utf-8"), (quest_ip, args.udp_port))

                        bx1, by1, bx2, by2 = [int(v) for v in tracked_box]
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    else:
                        lost_frames += 1
                        if lost_frames > 12:
                            state = "SEARCHING"
                            tracked_box = None
                            filter_kpts = None
                            prev_kpts = None

                        # VAN PHAI GUI tin hieu, khong duoc im lang: neu khong,
                        # Unity khong biet gi va se DONG BANG o dang sai cuoi
                        # cung. "valid:0" bao Unity chuyen muot ve tu the nghi.
                        udp_sock.sendto(b"valid:0", (quest_ip, args.udp_port))

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

                    mrp_txt = "EP NAM" if mrp_close > 0.5 else "bam theo model"
                    cv2.putText(frame, f"3 ngon giua/ap ut/ut: {mrp_txt} ({mrp_close:.2f})",
                                (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 200, 255) if mrp_close > 0.5 else (140, 255, 140), 1, cv2.LINE_AA)

                    cv2.imshow(win_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        return
            except (ConnectionError, socket.timeout, OSError) as e:
                print(f"[DISCONNECTED] Mat ket noi voi kinh ({quest_ip}): {e}. Cho ket noi lai...")
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("\nDung lai.")
    finally:
        srv.close()
        cv2.destroyAllWindows()
        udp_sock.close()


if __name__ == "__main__":
    main()
