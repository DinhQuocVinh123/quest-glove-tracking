using System;
using System.Net.Sockets;
using System.Text;
using Meta.XR;
using Unity.Collections;
using UnityEngine;

/// <summary>
/// Thu thập dữ liệu training cho việc fine-tune model nhận diện tay đeo
/// găng, dùng CAMERA THẬT của Quest (Passthrough Camera API) thay vì
/// webcam, và dùng NATIVE HAND TRACKING của Quest cho tay trần (rất đáng
/// tin cậy) làm "nhãn chuẩn" -- không cần chạy model AI nào trên kính cả.
///
/// Thiết kế: chỉ lấy 21 điểm khớp 3D của TAY TRẦN (qua OVRSkeleton.Bones,
/// đã có sẵn từ native tracking), chiếu sang tọa độ 2D trên ảnh camera
/// thật (qua PassthroughCameraAccess.WorldToViewportPoint). KHÔNG cố lấy
/// tracking của tay găng trên kính (vì đó chính là thứ Quest cũng khó
/// nhận diện, y như model Python của mình) -- việc tìm/cắt vùng tay găng
/// trong ảnh vẫn để code Python (đã có, đã test) xử lý sau, y hệt cách
/// đang làm với webcam.
///
/// Mỗi mẫu ghi được GỬI THẲNG QUA WIFI (TCP) về máy tính đang chạy
/// dataset_receiver.py -- KHÔNG lưu file trên kính, không cần rút USB/vào
/// File Explorer thủ công.
///
/// Cách dùng:
/// 1. Trên máy tính, chạy trước: python dataset_receiver.py --port 5006
/// 2. Gắn script này vào 1 GameObject bất kỳ trong scene.
/// 3. Kéo PassthroughCameraAccess (đã bật, đang chạy) vào field _passthroughCamera.
/// 4. Kéo object tay TRẦN (vd "[BuildingBlock] Hand Tracking left") vào CẢ
///    HAI field _bareHandOVR và _bareSkeleton (nó có cả 2 component đó).
/// 5. Điền IP của máy tính (vd 10.42.0.166) vào _pcIpAddress.
/// 6. Bấm nút A trên tay cầm để ghi lại 1 mẫu (ảnh + nhãn), gửi ngay lập
///    tức qua WiFi. Giữ nút B để bật/tắt tự động ghi liên tục.
/// </summary>
public class GloveDatasetCollector : MonoBehaviour
{
    [Header("Nguồn dữ liệu")]
    [Tooltip("Component PassthroughCameraAccess đang chạy (đã Enable), cung cấp ảnh + phép chiếu 3D->2D.")]
    [SerializeField] private PassthroughCameraAccess _passthroughCamera;
    [Tooltip("Component 'OVR Hand (Script)' trên object tay TRẦN (vd: [BuildingBlock] Hand Tracking left) -- dùng để kiểm tra độ tin cậy tracking.")]
    [SerializeField] private OVRHand _bareHandOVR;
    [Tooltip("Component 'OVR Skeleton (Script)' trên CÙNG object với _bareHandOVR -- nguồn tọa độ 21 khớp.")]
    [SerializeField] private OVRSkeleton _bareSkeleton;

    [Header("Gửi qua WiFi về máy tính (thay cho lưu file + USB)")]
    [Tooltip("Địa chỉ IP của máy tính đang chạy dataset_receiver.py (xem IP hiện ra khi chạy script đó).")]
    [SerializeField] private string _pcIpAddress = "10.42.0.166";
    [SerializeField] private int _pcPort = 5006;

    [Header("Điều khiển ghi")]
    [SerializeField] private OVRInput.Button _captureButton = OVRInput.Button.One; // nút A
    [SerializeField] private OVRInput.Button _autoToggleButton = OVRInput.Button.Two; // nút B
    [Tooltip("Khi bật auto-capture (nút B), khoảng cách giữa 2 lần ghi tự động (giây).")]
    [SerializeField] private float _autoCaptureIntervalSeconds = 0.3f;

    [Header("Feedback (tuỳ chọn -- kéo 1 TextMeshPro vào đây để thấy số đã gửi ngay trong kính)")]
    [SerializeField] private TMPro.TextMeshPro _statusText;

    [Header("Gỡ lỗi -- KHÔNG THỂ TỰ KIỂM TRA (không có kính để test)")]
    [Tooltip("Dữ liệu ảnh thô từ RenderTexture của Unity đôi khi bị ngược trên-dưới so với file JPEG chuẩn. " +
             "Nếu sau khi xem thử ảnh đã nhận trên máy tính, bạn thấy ảnh bị lộn ngược HOẶC các điểm nhãn " +
             "(kpts) không khớp đúng vị trí ngón tay theo chiều dọc, hãy đảo giá trị này rồi thử ghi lại vài mẫu.")]
    [SerializeField] private bool _imageIsBottomUp = true;

    // Đúng thứ tự 21 điểm dùng xuyên suốt các script Python:
    // 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
    // (mỗi ngón: base/MCP -> giữa -> gần đầu -> đầu ngón).
    // Dùng bộ tên "XRHand_*" (không phải "Hand_*") vì Skeleton Type trên
    // OVR Skeleton component đang để "OpenXR Hand".
    private static readonly OVRSkeleton.BoneId[] JointOrder = new[]
    {
        OVRSkeleton.BoneId.XRHand_Wrist,
        OVRSkeleton.BoneId.XRHand_ThumbMetacarpal, OVRSkeleton.BoneId.XRHand_ThumbProximal, OVRSkeleton.BoneId.XRHand_ThumbDistal, OVRSkeleton.BoneId.XRHand_ThumbTip,
        OVRSkeleton.BoneId.XRHand_IndexProximal, OVRSkeleton.BoneId.XRHand_IndexIntermediate, OVRSkeleton.BoneId.XRHand_IndexDistal, OVRSkeleton.BoneId.XRHand_IndexTip,
        OVRSkeleton.BoneId.XRHand_MiddleProximal, OVRSkeleton.BoneId.XRHand_MiddleIntermediate, OVRSkeleton.BoneId.XRHand_MiddleDistal, OVRSkeleton.BoneId.XRHand_MiddleTip,
        OVRSkeleton.BoneId.XRHand_RingProximal, OVRSkeleton.BoneId.XRHand_RingIntermediate, OVRSkeleton.BoneId.XRHand_RingDistal, OVRSkeleton.BoneId.XRHand_RingTip,
        OVRSkeleton.BoneId.XRHand_LittleProximal, OVRSkeleton.BoneId.XRHand_LittleIntermediate, OVRSkeleton.BoneId.XRHand_LittleDistal, OVRSkeleton.BoneId.XRHand_LittleTip,
    };

    private int _sentCount;
    private int _failCount;
    private float _lastAutoCaptureTime;
    private bool _autoCapturing;
    private string _lastStatus = "";

    private void Update()
    {
        if (_passthroughCamera == null || _bareHandOVR == null || _bareSkeleton == null)
        {
            if (_statusText != null)
            {
                _statusText.text = "GloveDatasetCollector:\nThiếu _passthroughCamera / _bareHandOVR / _bareSkeleton!";
            }
            return;
        }

        if (OVRInput.GetDown(_autoToggleButton))
        {
            _autoCapturing = !_autoCapturing;
        }

        bool manualPress = OVRInput.GetDown(_captureButton);
        bool autoDue = _autoCapturing && (Time.time - _lastAutoCaptureTime) >= _autoCaptureIntervalSeconds;

        if (manualPress || autoDue)
        {
            _lastAutoCaptureTime = Time.time;
            TryCaptureSample();
        }

        UpdateStatusText();
    }

    private void TryCaptureSample()
    {
        if (!_passthroughCamera.IsPlaying)
        {
            Debug.LogWarning("[GloveDatasetCollector] Passthrough camera chưa sẵn sàng, bỏ qua khung này.");
            return;
        }
        if (!_bareHandOVR.IsDataValid || !_bareHandOVR.IsDataHighConfidence)
        {
            // Không track được tay trần đủ tin cậy lúc này -- không ghi
            // (giống quy tắc "chỉ ghi khi khung xương nhìn đúng" bên Python).
            return;
        }

        var bones = _bareSkeleton.Bones;
        if (bones == null || bones.Count == 0)
        {
            return;
        }

        // 1. Chiếu 21 điểm khớp tay trần (world space, từ native tracking)
        // sang tọa độ pixel trên ảnh camera thật.
        var pixelPoints = new Vector2[JointOrder.Length];
        bool allValid = true;
        for (int i = 0; i < JointOrder.Length; i++)
        {
            Transform boneTransform = null;
            for (int b = 0; b < bones.Count; b++)
            {
                if (bones[b].Id == JointOrder[i])
                {
                    boneTransform = bones[b].Transform;
                    break;
                }
            }
            if (boneTransform == null)
            {
                allValid = false;
                break;
            }
            Vector2 viewport = _passthroughCamera.WorldToViewportPoint(boneTransform.position);
            // Viewport: (0,0) = góc dưới-trái, (1,1) = góc trên-phải.
            // Ảnh lưu ra (giống OpenCV/JPEG): (0,0) = góc trên-trái.
            // Nên phải LẬT trục Y khi quy đổi.
            float px = viewport.x * _passthroughCamera.CurrentResolution.x;
            float py = (1f - viewport.y) * _passthroughCamera.CurrentResolution.y;
            pixelPoints[i] = new Vector2(px, py);
        }
        if (!allValid)
        {
            return;
        }

        // Nếu toàn bộ 21 điểm đều rơi ra ngoài khung ảnh (ví dụ tay trần
        // đang ở ngoài tầm nhìn camera dù vẫn "high confidence" theo
        // native tracking khác, hoặc tay đang ở góc quá xa) -- bỏ qua,
        // không gửi mẫu vô nghĩa.
        var res = _passthroughCamera.CurrentResolution;
        bool anyInFrame = false;
        foreach (var p in pixelPoints)
        {
            if (p.x >= 0 && p.x <= res.x && p.y >= 0 && p.y <= res.y) { anyInFrame = true; break; }
        }
        if (!anyInFrame)
        {
            return;
        }

        // 2. Lấy ảnh camera thật hiện tại (CPU-side, dạng Color32[]).
        var colors = _passthroughCamera.GetColors();
        if (colors.Length == 0)
        {
            return;
        }

        Color32[] pixelArray = colors.ToArray();
        if (_imageIsBottomUp)
        {
            pixelArray = FlipRows(pixelArray, res.x, res.y);
        }
        var tex = new Texture2D(res.x, res.y, TextureFormat.RGBA32, false);
        tex.SetPixels32(pixelArray);
        tex.Apply();
        byte[] jpgBytes = tex.EncodeToJPG(90);
        Destroy(tex);

        // 3. Đóng gói nhãn thành JSON, GỬI THẲNG qua TCP về máy tính (không
        // lưu file trên kính). Giao thức khớp với dataset_receiver.py:
        //   [4 byte big-endian: do dai JSON][JSON bytes]
        //   [4 byte big-endian: do dai JPEG][JPEG bytes]
        var sb = new StringBuilder();
        sb.Append("{\"image_width\":").Append(res.x).Append(',');
        sb.Append("\"image_height\":").Append(res.y).Append(',');
        sb.Append("\"timestamp\":").Append(DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()).Append(',');
        sb.Append("\"bare_hand_kpts\":[");
        for (int i = 0; i < pixelPoints.Length; i++)
        {
            if (i > 0) sb.Append(',');
            sb.Append('[').Append(pixelPoints[i].x.ToString("F2", System.Globalization.CultureInfo.InvariantCulture))
              .Append(',').Append(pixelPoints[i].y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture)).Append(']');
        }
        sb.Append("]}");
        byte[] jsonBytes = Encoding.UTF8.GetBytes(sb.ToString());

        SendSampleOverNetwork(jsonBytes, jpgBytes);
    }

    private void SendSampleOverNetwork(byte[] jsonBytes, byte[] jpgBytes)
    {
        try
        {
            using (var client = new TcpClient())
            {
                client.SendTimeout = 3000;
                client.ReceiveTimeout = 3000;
                client.Connect(_pcIpAddress, _pcPort);
                using (var stream = client.GetStream())
                {
                    WriteFramed(stream, jsonBytes);
                    WriteFramed(stream, jpgBytes);

                    var ack = new byte[2];
                    int read = stream.Read(ack, 0, ack.Length);
                    if (read <= 0)
                    {
                        throw new Exception("Khong nhan duoc phan hoi tu may tinh.");
                    }
                }
            }
            _sentCount++;
            _lastStatus = "OK";
        }
        catch (Exception e)
        {
            _failCount++;
            _lastStatus = "LOI: " + e.Message;
            Debug.LogWarning($"[GloveDatasetCollector] Gui mau that bai ({_pcIpAddress}:{_pcPort}): {e.Message}");
        }
    }

    private static void WriteFramed(NetworkStream stream, byte[] data)
    {
        byte[] lenPrefix = BitConverter.GetBytes(data.Length);
        if (BitConverter.IsLittleEndian)
        {
            Array.Reverse(lenPrefix); // big-endian, khớp struct.unpack('>I', ...) bên Python
        }
        stream.Write(lenPrefix, 0, lenPrefix.Length);
        stream.Write(data, 0, data.Length);
    }

    private static Color32[] FlipRows(Color32[] src, int width, int height)
    {
        var dst = new Color32[src.Length];
        for (int y = 0; y < height; y++)
        {
            int srcRow = y * width;
            int dstRow = (height - 1 - y) * width;
            Array.Copy(src, srcRow, dst, dstRow, width);
        }
        return dst;
    }

    private void UpdateStatusText()
    {
        if (_statusText == null) return;
        _statusText.text = $"Glove Dataset Collector\nSent: {_sentCount}  Failed: {_failCount}\n" +
                            $"Auto: {(_autoCapturing ? "ON" : "off")}\nLast: {_lastStatus}\n" +
                            $"-> {_pcIpAddress}:{_pcPort}\n[A] capture 1  [B] toggle auto";
    }
}
