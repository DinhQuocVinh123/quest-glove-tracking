#!/usr/bin/env python3
"""
Nhận ảnh + nhãn (21 khớp tay trần) gửi trực tiếp từ app Unity trên Quest
qua WiFi (TCP), lưu vào máy tính này -- thay cho việc phải rút USB/vào
File Explorer thủ công.

Giao thức đơn giản, mỗi lần Unity ghi 1 mẫu thì mở 1 kết nối TCP ngắn,
gửi theo thứ tự:
  [4 byte big-endian: độ dài JSON][JSON bytes]
  [4 byte big-endian: độ dài JPEG][JPEG bytes]
rồi đóng kết nối.

Chạy: python dataset_receiver.py --port 5006
Rồi trong Unity (GloveDatasetCollector), điền đúng IP máy này + port vào
_pcIpAddress / _pcPort.
"""

import argparse
import json
import os
import socket
import struct
import threading
import time

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_BASE_DIR, "dataset_from_quest")
IMG_DIR = os.path.join(OUT_DIR, "images")
ANNOTATIONS_PATH = os.path.join(OUT_DIR, "annotations.jsonl")

_lock = threading.Lock()
_count = 0


def _recv_exact(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Ket noi dong som, khong nhan du du lieu.")
        buf += chunk
    return buf


def handle_client(conn, addr):
    global _count
    try:
        json_len = struct.unpack(">I", _recv_exact(conn, 4))[0]
        json_bytes = _recv_exact(conn, json_len)
        jpg_len = struct.unpack(">I", _recv_exact(conn, 4))[0]
        jpg_bytes = _recv_exact(conn, jpg_len)

        meta = json.loads(json_bytes.decode("utf-8"))

        with _lock:
            idx = _count
            _count += 1

        img_name = f"quest_{idx:06d}.jpg"
        with open(os.path.join(IMG_DIR, img_name), "wb") as f:
            f.write(jpg_bytes)

        record = dict(meta)
        record["image_file"] = img_name
        with open(ANNOTATIONS_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"[{idx}] Nhan tu {addr[0]}: {img_name} ({len(jpg_bytes)} bytes), "
              f"{len(record.get('bare_hand_kpts', []))} kpts")

        try:
            conn.sendall(b"OK")
        except Exception:
            pass
    except Exception as e:
        print(f"[ERROR] {addr}: {e}")
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Nhan dataset tu Quest qua WiFi")
    parser.add_argument("--port", type=int, default=5006)
    args = parser.parse_args()

    os.makedirs(IMG_DIR, exist_ok=True)
    if not os.path.exists(ANNOTATIONS_PATH):
        open(ANNOTATIONS_PATH, "w").close()

    global _count
    if os.path.exists(ANNOTATIONS_PATH):
        with open(ANNOTATIONS_PATH) as f:
            _count = sum(1 for _ in f)

    # In ra tat ca dia chi IP cua may nay de de dien vao Unity.
    hostname = socket.gethostname()
    try:
        local_ips = socket.gethostbyname_ex(hostname)[2]
    except Exception:
        local_ips = ["(khong lay duoc tu dong, dung 'ipconfig' trong PowerShell)"]

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", args.port))
    srv.listen(5)

    print(f"[READY] Dang lang nghe tren cong {args.port}. Da co san {_count} mau.")
    print(f"Dia chi IP may nay (dien vao Unity _pcIpAddress): {local_ips}")
    print("Nhan Ctrl+C de dung.\n")

    try:
        while True:
            conn, addr = srv.accept()
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        print("\nDung lai.")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
