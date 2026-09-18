# Theo dõi găng tay haptic bằng camera của Quest 3

Bộ công cụ **dùng lại được**: cắm vào bất kỳ dự án Unity nào cho Quest 3 để có ngay
khả năng theo dõi bàn tay đeo găng haptic — không cần webcam hay camera rời.

Camera passthrough của kính gửi hình về máy tính, máy tính chạy mạng nhận diện khớp
tay (RTMPose), rồi gửi ngược 21 điểm khớp về kính để điều khiển bàn tay ảo.

## Luồng hoạt động

```
        KÍNH QUEST 3                          MÁY TÍNH (cần GPU NVIDIA)
   ┌─────────────────────┐                  ┌──────────────────────────┐
   │ GloveLiveStreamer   │  ảnh JPEG, TCP   │ run_glove_quest_stream   │
   │ (camera passthrough)├─────────────────>│  • tìm bàn tay           │
   │                     │   cổng 5007      │  • chạy RTMPose          │
   │                     │                  │  • lọc kết quả sai       │
   │ FingerUDPReceiver   │  21 điểm, UDP    │                          │
   │ (xoay xương tay ảo) │<─────────────────┤                          │
   └─────────────────────┘   cổng 5005      └──────────────────────────┘
                                   ▲
                    kính TỰ TÌM máy tính qua tín hiệu phát
                    trên cổng UDP 5008 — không cần điền IP
```

Việc tự tìm này có chủ đích: đổi mạng, đổi hotspot, IP thay đổi — không phải sửa
gì và không phải build lại. Nó chỉ chạy lúc kết nối, sau đó nằm ngoài đường truyền
dữ liệu nên **không thêm độ trễ** cho mỗi khung hình.

---

# Cắm vào một dự án Unity mới

### Điều kiện cần

- **Quest 3 hoặc 3S**, Horizon OS **v74 trở lên** (API camera passthrough chỉ có từ đây)
- Dự án Unity đã có **Meta XR SDK** và bật **hand tracking**
- Một **model bàn tay 3D** có xương đặt tên theo chuẩn `XRHand_*`
  (ví dụ `XRHand_IndexProximal`, `XRHand_ThumbMetacarpal`) — model tay sẵn có của
  Meta XR SDK đã đúng chuẩn này
- Máy tính có **GPU NVIDIA**, cùng mạng với kính

### Bước 1 — Chép 3 file C#

Chép từ `unity/` vào `Assets/Scripts/` của dự án mới:

| File | Việc |
|---|---|
| `GloveLiveStreamer.cs` | Gửi hình từ camera passthrough về máy tính |
| `FingerUDPReceiver.cs` | Nhận 21 điểm, xoay xương bàn tay ảo |
| `HandFingerRig.cs` | Tự tìm các xương ngón theo tên `XRHand_*` |

(`GloveDatasetCollector.cs` chỉ cần nếu bạn muốn thu thêm dữ liệu huấn luyện.)

### Bước 2 — Xin quyền dùng camera

Thêm vào `Assets/Plugins/Android/AndroidManifest.xml`:

```xml
<manifest ... xmlns:horizonos="http://schemas.horizonos/sdk">
  <horizonos:uses-horizonos-sdk horizonos:minSdkVersion="74" ... />
  <uses-permission android:name="horizonos.permission.HEADSET_CAMERA" />
```

Thiếu bước này thì camera sẽ không bao giờ khởi động được.

### Bước 3 — Dựng scene

**a) Object gửi hình:** tạo một GameObject trống, gắn `GloveLiveStreamer`, rồi kéo
component `PassthroughCameraAccess` (đang bật, đang chạy) vào ô `Passthrough Camera`.

Để nguyên `Use Auto Discovery` đã tích sẵn — khi đó ô IP bên dưới không cần điền.

**b) Bàn tay ảo:** trên GameObject chứa model bàn tay, gắn `FingerUDPReceiver`
(`HandFingerRig` sẽ tự được thêm vào). Nếu object có component `Hand Visual`, kéo nó
vào ô cùng tên để tránh tranh chấp với hand tracking gốc của Quest.

> **Quan trọng:** object bàn tay nên là **con của `CenterEyeAnchor`**. Camera gắn trên
> đầu nên di chuyển theo đầu, vì vậy toạ độ phải tính tương đối với đầu.

`HandFingerRig` tự tìm xương theo tên, thường không phải kéo thả gì thêm.

### Bước 4 — Cài môi trường Python

Xem `requirements.txt` — **phải cài đúng thứ tự trong đó**, không chạy
`pip install -r` một phát được. Bộ thư viện OpenMMLab rất kén phiên bản.

Rồi clone mmpose vào **ngay cạnh** các file `.py`:

```bash
git clone https://github.com/open-mmlab/mmpose.git
```

### Bước 5 — Chạy

```bash
# Máy tính chạy TRƯỚC, rồi mới mở app trên kính:
python run_glove_quest_stream.py --original
```

Cửa sổ debug sẽ hiện backbone bàn tay kèm số liệu chẩn đoán (FPS, độ tin cậy, lý do
loại khung hình...). Model gốc **tự tải về** lần chạy đầu, không cần chuẩn bị gì.

**Nếu không kết nối được:** kính và máy tính phải cùng mạng, và mạng đó không được
chặn thiết bị nói chuyện trực tiếp với nhau. Mạng công ty hay chặn kiểu này —
hotspot điện thoại là phương án chắc ăn nhất.

---

## Giao thức (nếu cần thay thế một nửa)

**Kính → máy tính, TCP cổng 5007:** lặp lại `[4 byte độ dài, big-endian][ảnh JPEG]`

**Máy tính → kính, UDP cổng 5005:** chuỗi văn bản `khóa:giá_trị` cách nhau bằng dấu phẩy

| Khóa | Ý nghĩa |
|---|---|
| `valid` | `1` = dữ liệu tin được, `0` = tay ảo nên về tư thế nghỉ |
| `pts` | 21 điểm `x\|y;x\|y;...`, gốc toạ độ tại cổ tay, chia theo chiều dài lòng bàn tay, `+y` hướng lên |
| `pinch` | 0..1, khoảng cách 2 đầu ngón cái–trỏ |
| `thumb0/1/2`, `index0/1/2` | Góc cong từng khớp (gốc/giữa/đầu) |

**Thứ tự 21 điểm:** `0` cổ tay, `1-4` ngón cái, `5-8` trỏ, `9-12` giữa, `13-16` áp út,
`17-20` út. Mỗi ngón đi từ khớp gốc ra đầu ngón.

---

## Các file Python

**Đường chạy chính:**

| File | Việc |
|---|---|
| `run_glove_quest_stream.py` | **Script chính.** Camera kính → máy tính → Unity |
| `run_glove_to_unity.py` | Bản dùng webcam thay cho camera kính |
| `run_white_haptics_glove.py` | Xem thử trên máy tính, không cần đeo kính |
| `dataset_receiver.py` | Nhận mẫu huấn luyện từ `GloveDatasetCollector.cs` |

**Công cụ làm dữ liệu / huấn luyện:**
`manual_label_quest.py` (dán nhãn tay), `review_quest_import.py`, `review_dataset.py`,
`import_quest_dataset.py`, `run_split_screen_finetune.py`, `finetune_glove.py`

Các file `run_*.py` còn lại là thử nghiệm cũ, giữ để tham khảo, **không thuộc đường
chạy hiện tại**.

> Tất cả file `.py` phải nằm **cùng một cấp thư mục** — chúng tìm `mmpose/` và
> `dataset/` theo đường dẫn tương đối so với chính nó.

---

## Vì sao có cờ `--original`

Cờ này bắt dùng model **gốc** thay vì bản đã fine-tune. Nghe ngược đời, nhưng khi so
sánh trực tiếp, bản gốc bám ngón cái/trỏ **ổn định hơn hẳn** bản fine-tune trên 170
mẫu tự thu. Nguyên nhân nhiều khả năng là bộ dữ liệu còn quá nhỏ và thiếu đa dạng,
khiến fine-tune làm hỏng khả năng khái quát vốn có của model gốc.

Bỏ cờ đi thì script dùng `checkpoints/rtmpose_glove_finetuned.pth` nếu file đó tồn tại.

## Vài điều đã học được (để khỏi giẫm lại)

- **Kính gửi nhanh hơn máy tính xử lý.** TCP không vứt dữ liệu mà xếp hàng, nên khung
  hình cũ dồn lại và độ trễ **tăng dần** tới 1–2 giây. `LatestFrameReader` luôn vứt
  khung cũ, chỉ xử lý khung mới nhất.

- **Model cần được chỉ chỗ để nhìn.** RTMPose thuộc loại "top-down": nó không tự dò cả
  ảnh mà phải được đưa sẵn một khung. Nếu chỉ đưa vài vùng cố định thì đưa tay ra rìa
  khung hình là mất dấu hoàn toàn. Hiện dùng lưới quét phủ kín toàn khung, luân phiên
  từng phần qua các khung hình để không tụt FPS.

- **Lọc theo hình dạng, đừng chỉ tin điểm số.** Model chưa từng học chiếc găng này nên
  độ tin cậy luôn sát ngưỡng; nền rối là tụt xuống dưới. Cách đáng tin hơn là kiểm tra
  hình dạng có giống bàn tay không (chiều dài từng đốt, tổng chiều dài ngón, có nhảy
  đột ngột giữa 2 khung không).

- **Đừng đặt ngưỡng bằng pixel tuyệt đối.** Camera gắn trên đầu nên tay có thể sát ngay
  trước mặt và nhìn nghiêng, lúc đó các khớp chồng lên nhau khi chiếu xuống ảnh 2D.
  Ngưỡng pixel viết cho webcam để bàn sẽ loại oan chính những khung hình tốt.

- **Xoay xương theo hướng backbone, đừng xoay quanh một trục cố định.** Ngón tay thật
  cong trong mặt phẳng bất kỳ. Xoay mỗi khớp quanh một trục chọn sẵn thì dù tính góc
  chuẩn đến đâu, hình dạng cũng không bao giờ khớp.

- **Muốn pinch khép được thì 2 ngón phải cùng mặt phẳng.** Nếu giữ nguyên chiều sâu tư
  thế nghỉ, ngón cái luôn chĩa về phía người nhìn nên không bao giờ chạm được ngón trỏ
  (`Preserve Rest Depth` phải để TẮT).

- **3 ngón giữa/áp út/út bị chính bàn tay che khi pinch**, model đoán rất bậy. Python
  tự ép chúng về dáng nắm khi đang pinch, thay vì tin model.
