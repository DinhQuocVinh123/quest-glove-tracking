using System;
using System.Net.Sockets;
using Meta.XR;
using Unity.Collections;
using UnityEngine;

/// <summary>
/// Biến thể "live" của GloveDatasetCollector: thay vì ghi từng mẫu training
/// khi bấm nút, script này LIÊN TỤC gửi khung hình từ camera Passthrough
/// thật của Quest sang máy tính (qua 1 kết nối TCP giữ mở), để
/// run_glove_quest_stream.py chạy model AI nhận diện tay đeo găng và gửi
/// ngược lại dữ liệu ngón tay/pinch qua UDP vào FingerUDPReceiver.
///
/// Không cần điện thoại/webcam rời -- camera duy nhất là của kính Quest.
/// Máy tính vẫn là nơi chạy model AI (kính không đủ mạnh để chạy real-time).
///
/// Cách dùng:
/// 1. Trên máy tính, chạy trước: python run_glove_quest_stream.py --original
///    (script sẽ in ra IP máy này và cổng đang lắng nghe, mặc định 5007).
/// 2. Gắn script này vào 1 GameObject bất kỳ trong scene.
/// 3. Kéo PassthroughCameraAccess (đã bật, đang chạy) vào field _passthroughCamera.
/// 4. Điền đúng IP của máy tính vào _pcIpAddress (cổng nhận hình mặc định 5007).
/// 5. FingerUDPReceiver bên kính sẽ tự nhận dữ liệu ngón tay/pinch --
///    run_glove_quest_stream.py tự phát hiện IP của kính từ kết nối TCP,
///    không cần điền IP kính vào máy tính.
/// </summary>
public class GloveLiveStreamer : MonoBehaviour
{
    [Header("Nguồn camera")]
    [Tooltip("Component PassthroughCameraAccess đang chạy (đã Enable).")]
    [SerializeField] private PassthroughCameraAccess _passthroughCamera;

    [Header("Tự tìm máy tính (khỏi phải điền IP + build lại mỗi lần đổi mạng)")]
    [Tooltip("BAT: kinh tu lang nghe tin hieu ma may tinh phat ra moi giay, tu biet IP -- doi mang bao nhieu lan cung khong can sua gi.\n\n" +
             "Viec do tim CHI xay ra luc ket noi, sau do nam ngoai duong truyen du lieu: khong them do tre cho moi khung hinh. " +
             "Chi mat toi da ~1 giay cho luc moi mo app.\n\n" +
             "TAT: dung IP dien tay o o _pcIpAddress ben duoi (dung khi mang chan goi tin broadcast, vd mot so mang cong ty).")]
    [SerializeField] private bool _useAutoDiscovery = true;
    [Tooltip("Cong UDP de nghe tin hieu tu may tinh -- phai khop --discovery-port ben run_glove_quest_stream.py (mac dinh 5008).")]
    [SerializeField] private int _discoveryPort = 5008;

    [Header("Gửi liên tục về máy tính (chạy run_glove_quest_stream.py)")]
    [Tooltip("Địa chỉ IP của máy tính đang chạy run_glove_quest_stream.py. Chỉ dùng khi TẮT auto-discovery.")]
    [SerializeField] private string _pcIpAddress = "10.42.0.166";
    [SerializeField] private int _pcPort = 5007;

    [Header("Tốc độ gửi")]
    [Tooltip("Khoảng cách tối thiểu giữa 2 lần gửi khung hình (giây). 0.05 = tối đa ~20 FPS -- " +
             "không cần cao hơn tốc độ xử lý thực tế của máy tính, WiFi/GPU mới là điểm nghẽn.")]
    [SerializeField] private float _sendIntervalSeconds = 0.05f;
    [Tooltip("Chất lượng nén JPEG (0-100). Thấp hơn = gửi nhanh hơn, đỡ nghẽn WiFi, nhưng model có thể kém chính xác hơn chút.")]
    [SerializeField] private int _jpegQuality = 80;

    [Header("Gỡ lỗi -- dùng CHUNG giá trị đã xác nhận đúng ở GloveDatasetCollector")]
    [Tooltip("Nếu ảnh nhận được trên máy tính bị lộn ngược trên-dưới, đảo giá trị này. " +
             "Dùng đúng giá trị bạn đã xác nhận hoạt động đúng với GloveDatasetCollector trên cùng kính này.")]
    [SerializeField] private bool _imageIsBottomUp = true;

    [Header("Feedback (tuỳ chọn -- kéo 1 TextMeshPro vào đây để xem trạng thái ngay trong kính)")]
    [SerializeField] private TMPro.TextMeshPro _statusText;

    private TcpClient _client;
    private NetworkStream _stream;
    private System.Net.Sockets.UdpClient _discoveryClient;
    private string _discoveredIp;
    private int _discoveredPort;
    private float _lastSendTime;
    private float _lastConnectAttemptTime;
    private int _sentCount;
    private int _failCount;
    private string _lastStatus = "Chua ket noi";

    private void OnDestroy()
    {
        CloseConnection();
        CloseDiscovery();
    }

    private void OnApplicationQuit()
    {
        CloseConnection();
        CloseDiscovery();
    }

    private void CloseDiscovery()
    {
        try { _discoveryClient?.Close(); } catch { /* bo qua */ }
        _discoveryClient = null;
    }

    private void Update()
    {
        // Nghe tin hieu tu may tinh moi khung hinh (khong chan, chi doc khi
        // that su co goi tin trong hang doi).
        if (_useAutoDiscovery) PollDiscovery();

        if (_passthroughCamera == null)
        {
            _lastStatus = "Thieu _passthroughCamera!";
            UpdateStatusText();
            return;
        }
        if (!_passthroughCamera.IsPlaying)
        {
            _lastStatus = "Cho passthrough camera san sang...";
            UpdateStatusText();
            return;
        }
        if (Time.time - _lastSendTime < _sendIntervalSeconds)
        {
            return;
        }
        _lastSendTime = Time.time;
        TrySendFrame();
        UpdateStatusText();
    }

    /// <summary>Doc tin hieu "may chu o day" ma may tinh phat ra (dinh dang
    /// "GLOVE_SERVER|&lt;cong TCP&gt;"). IP cua may tinh lay THANG tu dia chi
    /// nguoi gui goi tin, nen khong can dien tay bao gio.
    ///
    /// Khong chan luong chinh: chi doc khi UdpClient.Available > 0, tuc la da
    /// co san goi tin trong hang doi -- Receive() tra ve ngay lap tuc.</summary>
    private void PollDiscovery()
    {
        try
        {
            if (_discoveryClient == null)
            {
                _discoveryClient = new System.Net.Sockets.UdpClient();
                _discoveryClient.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
                _discoveryClient.Client.Bind(new System.Net.IPEndPoint(System.Net.IPAddress.Any, _discoveryPort));
            }

            while (_discoveryClient.Available > 0)
            {
                var remote = new System.Net.IPEndPoint(System.Net.IPAddress.Any, 0);
                byte[] data = _discoveryClient.Receive(ref remote);
                string msg = System.Text.Encoding.UTF8.GetString(data);
                if (!msg.StartsWith("GLOVE_SERVER|")) continue;

                string portText = msg.Substring("GLOVE_SERVER|".Length);
                if (!int.TryParse(portText, out int tcpPort)) continue;

                string ip = remote.Address.ToString();
                // Neu may tinh doi IP (doi mang), tu ngat ket noi cu de lan
                // gui ke tiep noi lai vao dia chi moi.
                if (_discoveredIp != ip || _discoveredPort != tcpPort)
                {
                    _discoveredIp = ip;
                    _discoveredPort = tcpPort;
                    CloseConnection();
                }
            }
        }
        catch (Exception e)
        {
            // Dong han socket hong roi de null -- lan Update sau se tao lai tu
            // dau. Neu giu lai socket chua bind duoc, moi lan goi .Available
            // se nem loi lien tuc va khong bao gio tu phuc hoi.
            _lastStatus = "Loi do tim: " + e.Message;
            CloseDiscovery();
        }
    }

    private void TrySendFrame()
    {
        if (_client == null || !_client.Connected)
        {
            // Đừng thử kết nối lại mỗi frame nếu đang thất bại liên tục --
            // giới hạn 1 lần mỗi giây để không làm nghẽn Update().
            if (Time.time - _lastConnectAttemptTime < 1.0f)
            {
                return;
            }
            _lastConnectAttemptTime = Time.time;
            if (!TryConnect())
            {
                return;
            }
        }

        try
        {
            var res = _passthroughCamera.CurrentResolution;
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
            byte[] jpgBytes = tex.EncodeToJPG(_jpegQuality);
            Destroy(tex);

            WriteFramed(_stream, jpgBytes);
            _sentCount++;
            _lastStatus = "OK";
        }
        catch (Exception e)
        {
            _failCount++;
            _lastStatus = "Mat ket noi: " + e.Message;
            CloseConnection();
        }
    }

    private bool TryConnect()
    {
        // Uu tien dia chi tu do tim duoc; neu chua nhan duoc tin hieu nao thi
        // quay ve IP dien tay (van chay duoc tren mang chan broadcast).
        bool useDiscovered = _useAutoDiscovery && !string.IsNullOrEmpty(_discoveredIp);
        string targetIp = useDiscovered ? _discoveredIp : _pcIpAddress;
        int targetPort = useDiscovered ? _discoveredPort : _pcPort;

        if (string.IsNullOrEmpty(targetIp))
        {
            _lastStatus = "Dang cho tin hieu tu may tinh...";
            return false;
        }

        try
        {
            _client = new TcpClient();
            _client.NoDelay = true; // gửi ngay từng khung, không gộp batch -- ưu tiên độ trễ thấp
            _client.SendTimeout = 2000;
            _client.Connect(targetIp, targetPort);
            _stream = _client.GetStream();
            _lastStatus = $"Da ket noi {targetIp}:{targetPort}" + (useDiscovered ? " (tu tim)" : " (IP tay)");
            return true;
        }
        catch (Exception e)
        {
            _lastStatus = $"Khong ket noi duoc {targetIp}:{targetPort}: {e.Message}";
            CloseConnection();
            return false;
        }
    }

    private void CloseConnection()
    {
        try { _stream?.Close(); } catch { /* bo qua */ }
        try { _client?.Close(); } catch { /* bo qua */ }
        _stream = null;
        _client = null;
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
        string target = (_useAutoDiscovery && !string.IsNullOrEmpty(_discoveredIp))
            ? $"{_discoveredIp}:{_discoveredPort} (tu tim)"
            : (_useAutoDiscovery ? "dang do tim..." : $"{_pcIpAddress}:{_pcPort}");
        _statusText.text = $"Glove Live Streamer\nSent: {_sentCount}  Failed: {_failCount}\n" +
                            $"-> {target}\nLast: {_lastStatus}";
    }
}
