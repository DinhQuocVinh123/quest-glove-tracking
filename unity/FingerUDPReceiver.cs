using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using Oculus.Interaction;
using UnityEngine;

/// <summary>
/// Nhận dữ liệu độ cong ngón tay (0..1 mỗi ngón) qua UDP từ script Python
/// dò màu (live_finger_tracker.py), rồi xoay đúng các xương trong
/// HandFingerRig -- thay thế FingerSineTest bằng dữ liệu thật từ camera.
///
/// Gắn cùng GameObject với HandFingerRig (StaticHandModel_Right).
/// Tắt hoặc gỡ FingerSineTest trước khi dùng script này (2 cái cùng
/// xoay 1 xương sẽ đá nhau).
///
/// QUAN TRỌNG: object này PHẢI là con của <c>CenterEyeAnchor</c> (bám
/// theo đầu), KHÔNG phải TrackingSpace -- vì webcam gắn trên đầu kính
/// VR, di chuyển cùng đầu, nên vị trí/góc xoay tay phải tính theo
/// local space (tương đối với đầu), không phải world space.
/// </summary>
[RequireComponent(typeof(HandFingerRig))]
public class FingerUDPReceiver : MonoBehaviour
{
    private enum Axis { X, Y, Z }

    [Header("Mạng")]
    [SerializeField] private int _port = 5005;

    [Header("Chống tranh chấp với native tracking -- kéo component 'Hand Visual' cùng object vào đây")]
    [Tooltip("De code cong ngon chay DUNG NGAY SAU khi HandVisual ghi xong khop (qua event WhenHandVisualUpdated), thay vi doan mo Update/LateUpdate. Bo trong van chay duoc (fallback LateUpdate) nhung khong dam bao thang native tracking.")]
    [SerializeField] private HandVisual _handVisual;

    [Header("BÁM THEO BACKBONE (khuyên dùng) -- xoay từng đốt xương chĩa đúng hướng đoạn backbone của model")]
    [Tooltip("BAT: nhan thang 21 diem (co tay + 4 diem moi ngon) tu Python, roi xoay MOI DOT XUONG sao cho no chia dung huong doan backbone tuong ung -- tai tao dung HINH DANG ngon tay.\n\n" +
             "TAT: quay ve cach cu (xoay moi khop quanh 1 truc co dinh theo do cong) -- cach do KHONG THE tai tao dung hinh dang, vi ngon tay that cong trong mat phang bat ky, khong trung voi truc do.\n\n" +
             "Camera gan tren dau kinh nen huong tren anh 2D ~ huong ban nhin thay: chia dung huong 2D thi tay ao trong dung dang tu goc nhin cua ban. Cach nay TU TIM RA truc xoay dung cho tung xuong, khong can doan X/Y/Z.")]
    [SerializeField] private bool _useBackboneRetarget = true;
    [Tooltip("Transform dung lam he quy chieu 'huong nhin' (phai/tren cua man hinh). De trong = dung parent (CenterEyeAnchor) -- dung cho hau het truong hop.")]
    [SerializeField] private Transform _viewReference;
    [Tooltip("TAT (mac dinh, CAN THIET CHO PINCH): ep moi dot xuong nam trong mat phang nhin -- ngon cai va ngon tro cung mot mat phang nen dau ngon GAP DUOC NHAU khi pinch. Doi lai, nhin nghieng tu canh ben se thay tay hoi bep.\n\n" +
             "BAT: giu nguyen do 'chia vao/ra man hinh' cua tu the nghi (trong day dan hon khi nhin nghieng), NHUNG pha vo pinch: ngon cai o tu the nghi von chia ngang qua long ban tay tuc la HUONG VE PHIA MAT BAN, nen no bi day ra truoc so voi ngon tro va khong bao gio cham duoc nhau. " +
             "Chua ke xuong chia ve phia mat se bi PHOI CANH lam ngan lai tren man hinh, khien dau ngon voi khong toi.")]
    [SerializeField] private bool _preserveRestDepth = false;
    [Tooltip("BAT (khuyen dung): dung lai dang 3D cua ngon tay -- tim GOC KHOP sao cho ngon ao nhin tu camera trung voi cac diem 2D, " +
             "ngon chi duoc gap theo cach khop that cho phep (xem FingerChainFitter). Dung ca khi ngon chia vao/ra camera (vd nam tay nhin tu tren xuong).\n\n" +
             "TAT: cach cu -- xoay tung dot cho chia dung huong 2D va ep nam phang trong mat phang nhin (nam tay se thanh ngon tro dung thang).")]
    [SerializeField] private bool _useAnatomicalFit = true;
    [Tooltip("Tay trai hay phai -- de biet phia nao la long ban tay khi tu tim truc gap ngon.")]
    [SerializeField] private bool _leftHand = false;

    [Header("Fallback -- về tư thế nghỉ khi dữ liệu không đáng tin")]
    [Tooltip("Model doi khi doan sai (tay chua vao tu the san sang, bi che khuat...) tao ra dang tay cong venh/lat nguoc rat ky quac. " +
             "Khi Python bao du lieu khong hop ly (valid:0) hoac khi mat tin hieu, tay ao chuyen MUOT ve tu the nghi binh thuong thay vi giu dang sai.\n\n" +
             "Day la thoi gian (giay) de chuyen muot giua 2 trang thai. Nho hon = phan ung nhanh nhung dot ngot; lon hon = muot nhung tre.")]
    [SerializeField] private float _fallbackBlendSeconds = 0.15f;
    [Tooltip("Neu khong nhan duoc goi tin nao trong so giay nay (Python tat, mat mang...), tu dong ve tu the nghi.")]
    [SerializeField] private float _dataTimeoutSeconds = 0.5f;

    [Header("4 ngón (trỏ/giữa/áp út/út) -- trục cong (chỉ dùng khi TẮT bám backbone)")]
    [SerializeField] private Axis _curlAxis = Axis.X;
    [SerializeField] private bool _invert = false;

    [Header("Ngón cái riêng")]
    [SerializeField] private Axis _thumbCurlAxis = Axis.X;
    [SerializeField] private bool _thumbInvert = false;

    [Header("Biên độ cong tối đa (độ)")]
    [SerializeField] private float _amplitudeDegrees = 70f;

    [Header("ĐỘ XÒE ngón cái (tạo khoảng hở cái<->trỏ, hình chữ 'C')")]
    [Tooltip("Do cong (curl) va do XOE (spread) la HAI bac tu do KHAC NHAU. Tu the 'chu C' (2 ngon deu THANG nhung cach nhau vai cm) duoc tao ra hoan toan boi do XOE nay -- neu khong ap no, tay ao khong the tao ra tu the do du curl co chuan den dau. " +
             "Truc nay la truc lam ngon cai DANG RA XA ngon tro (thuong KHAC voi truc curl). Neu chua biet truc nao dung: trong Scene view, chon XRHand_ThumbMetacarpal roi thu xoay tung truc X/Y/Z xem truc nao mo rong khoang ho cai<->tro.")]
    [SerializeField] private Axis _thumbSpreadAxis = Axis.Y;
    [SerializeField] private bool _thumbSpreadInvert = false;
    [Tooltip("Goc xoe toi da (do) khi spread=1.0 (2 ngon xoe rong het co). Tang/giam de khop voi khoang ho that.")]
    [SerializeField] private float _thumbSpreadAmplitudeDegrees = 45f;
    [Tooltip("Bat/tat viec ap do xoe -- tat di de so sanh voi hanh vi cu.")]
    [SerializeField] private bool _applyThumbSpread = true;

    [Header("Tỷ lệ cong từng khớp (gốc/giữa/đầu) -- nắm tay thật cong khớp GỐC nhiều hơn khớp ĐẦU, không đều nhau. Áp cùng 1 góc cho cả 3 khớp tạo cảm giác 'móng vuốt' cơ học.")]
    [SerializeField] private float _proximalCurlRatio = 1.0f;
    [SerializeField] private float _intermediateCurlRatio = 0.85f;
    [SerializeField] private float _distalCurlRatio = 0.6f;

    [Header("Dáng PINCH (ngón cái+trỏ chạm nhau) -- Python gửi 'pinch' LIÊN TỤC 0..1 (tỷ lệ khoảng cách 2 đầu ngón thật), tự trộn dần sang dáng này")]
    [Tooltip("Khi dang pinch, ngon TRO dung goc nay THAY VI logic bend/curl thong thuong (x=khop goc, y=khop giua, z=khop dau, don vi do). Can chinh truc tiep trong Inspector luc Play de khop dung dang pinch that -- day chi la gia tri khoi diem chua qua tinh chinh thuc te.")]
    [SerializeField] private Vector3 _pinchIndexAngles = new Vector3(30f, 25f, 15f);
    [Tooltip("Tuong tu cho ngon CAI (x=khop goc/metacarpal, y=khop giua/proximal, z=khop dau/distal).")]
    [SerializeField] private Vector3 _pinchThumbAngles = new Vector3(20f, 35f, 10f);
    [Tooltip("Lam muot rieng cho trang thai pinch (0 = doi tu the ngay lap tuc, cao hon = tu tu chuyen dang).")]
    [Range(0f, 0.95f)]
    [SerializeField] private float _pinchSmoothing = 0.3f;
    [Tooltip("CHI bat dau tron sang dang pinch co dinh khi _currentPinch vuot qua nguong nay (vd 0.75 = chi khi 2 dau ngon THAT SU gan cham nhau). Duoi nguong nay, dung 100% du lieu cong khop THAT, khong bi keo ve dang co dinh -- tranh dang pinch (chua tinh chinh chuan) lan at du lieu that o cac tu the binh thuong (vd pinch~0.3 khi 2 ngon con cach nhau vai cm).")]
    [Range(0f, 0.95f)]
    [SerializeField] private float _pinchBlendStartThreshold = 0.75f;

    [Header("Làm mượt (0 = không mượt, giá trị càng cao càng mượt nhưng trễ hơn)")]
    [Range(0f, 0.95f)]
    [SerializeField] private float _smoothing = 0.5f;

    [Header("Vùng chết (bỏ qua bend nhỏ hơn giá trị này, coi như đang mở)")]
    [Range(0f, 0.6f)]
    [SerializeField] private float _deadzone = 0.05f;

    [Header("Góc xoay bàn tay (từ solvePnP) -- BẬT/TẮT và hiệu chỉnh trục")]
    [SerializeField] private bool _applyRotation = false;
    [Tooltip("Nhân dấu/hệ số cho từng trục nhận được từ Python, dùng để đảo chiều khi hiệu chỉnh (vd: -1 để đảo trục đó).")]
    [SerializeField] private Vector3 _rotationSign = new Vector3(1f, 1f, 1f);
    [Range(0f, 0.95f)]
    [SerializeField] private float _rotationSmoothing = 0.6f;

    [Header("Vị trí bàn tay (từ solvePnP, đơn vị cm từ Python) -- BẬT/TẮT và hiệu chỉnh")]
    [SerializeField] private bool _applyPosition = false;
    [Tooltip("Nhân dấu cho từng trục (đảo chiều khi hiệu chỉnh). Trục Python: X=phải, Y=xuống, Z=ra xa camera.")]
    [SerializeField] private Vector3 _positionSign = new Vector3(1f, -1f, 1f);
    [Tooltip("Hệ số quy đổi cm (Python) sang mét (Unity) và làm giảm/tăng độ nhạy di chuyển.")]
    [SerializeField] private float _positionScale = 0.01f;
    [Range(0f, 0.95f)]
    [SerializeField] private float _positionSmoothing = 0.6f;

    // Hieu chuan camera-to-HEAD (khong phai camera-to-world nua!) -- vi
    // webcam gan TREN DAU KINH VR, di chuyen CUNG voi dau, nen do lech
    // giua webcam va dau la MOT HANG SO CO DINH, khong doi theo vi
    // tri/huong dau trong phong. Do 1 lan bang cach dat tay cam Touch
    // Plus sat webcam, doc so tu ControllerPosDisplay.cs (da tinh san
    // theo he truc CUA DAU), dien vao day, build lai 1 lan.
    [Header("Hiệu chuẩn camera-to-HEAD -- độ lệch vị trí webcam so với đầu (mét, local)")]
    [SerializeField] private float _camHeadPosX = 0.044f;
    [SerializeField] private float _camHeadPosY = -0.036f;
    [SerializeField] private float _camHeadPosZ = 0.113f;

    [Header("Hiệu chuẩn camera-to-HEAD -- độ lệch hướng webcam so với đầu (độ, local)")]
    [SerializeField] private float _camHeadRotX = 283.7f;
    [SerializeField] private float _camHeadRotY = 14.4f;
    [SerializeField] private float _camHeadRotZ = 339.9f;

    private Vector3 _camHeadPosition => new Vector3(_camHeadPosX, _camHeadPosY, _camHeadPosZ);
    private Vector3 _camHeadEulerRotation => new Vector3(_camHeadRotX, _camHeadRotY, _camHeadRotZ);

    private Quaternion _baseLocalRotation;
    private Vector3 _baseLocalPosition;
    private Vector3 _latestRotEuler;
    private Vector3 _currentRotEuler;
    private Vector3 _latestTranslation;
    private Vector3 _currentTranslation;

    private HandFingerRig _rig;
    private Transform[][] _fingerJoints; // [0]=thumb, [1]=index, [2]=middle, [3]=ring, [4]=pinky
    private float[][] _baseAngleOnAxis;
    private readonly string[] _fingerNames = { "thumb", "index", "middle", "ring", "pinky" };
    private readonly float[] _currentBend = new float[5]; // đã làm mượt, dùng trong Update
    private readonly float[] _latestBend = new float[5];  // giá trị thô mới nhất từ UDP

    // Góc cong THẬT tại từng khớp (gốc/giữa/đầu), riêng cho ngón cái(0)/trỏ(1)
    // -- tính thẳng từ hình dạng 8 điểm model detect được (xem
    // run_glove_quest_stream.py: calculate_finger_joint_bends), gửi qua UDP
    // với khóa "thumb0/1/2" và "index0/1/2". Khác với _latestBend/_currentBend
    // (1 giá trị chung rồi NHÂN TỶ LỆ đều cho cả 3 khớp) -- đây là 3 giá trị
    // ĐỘC LẬP, mỗi khớp phản ánh đúng góc thật của khớp đó.
    private readonly float[][] _latestJointBend = { new float[3], new float[3] };  // [0]=thumb,[1]=index
    private readonly float[][] _currentJointBend = { new float[3], new float[3] };
    // Ca 2 gia tri deu LIEN TUC 0..1 (Python gui ty le khoang cach 2 dau
    // ngon cai-tro, khong con nhi phan pinch/khong-pinch nua) -- nho vay
    // dang "chuan bi pinch" (2 ngon dang tien lai gan) tu nhien hien ra
    // o muc giua khi Lerp, khong can dinh nghia rieng 1 pose thu 3.
    private float _latestPinch;  // gia tri tho moi nhat tu UDP (pinch:0.0..1.0)
    private float _currentPinch; // da lam muot, dung de Lerp giua dang binh thuong va dang pinch

    // Do XOE ngon cai so voi ngon tro (0..1) -- bac tu do RIENG, khong lien
    // quan den do cong. Xem calculate_thumb_index_spread() ben Python.
    private float _latestSpread;
    private float _currentSpread;
    // Goc XOE o tu the nghi cua rig, doc 1 lan luc Awake -- do xoe nhan duoc
    // se cong THEM vao goc nghi nay (khong ghi de tuyet doi), de spread=0
    // giu nguyen dang tay tu nhien cua model (da xac nhan la dung).
    private float _thumbSpreadRestAngle;

    // 21 diem backbone tu model (0=co tay, 1-4=cai, 5-8=tro, 9-12=giua,
    // 13-16=ap ut, 17-20=ut), toa do chuan hoa quanh co tay theo chieu dai
    // long ban tay, +y huong LEN TREN.
    private const int NumBackbonePoints = 21;
    // Chi so diem GOC cua tung ngon trong mang tren, theo thu tu _fingerJoints
    // ([0]=cai, [1]=tro, [2]=giua, [3]=ap ut, [4]=ut).
    private static readonly int[] FingerBasePointIndex = { 1, 5, 9, 13, 17 };
    private readonly Vector2[] _latestPoints = new Vector2[NumBackbonePoints];
    private readonly Vector2[] _currentPoints = new Vector2[NumBackbonePoints];
    private volatile bool _hasPoints;
    // Huong "doc theo than xuong" trong HE TOA DO RIENG cua tung xuong, do 1
    // lan luc Awake bang cach nhin xem khop con nam ve phia nao. Nho vay khong
    // can doan truc X/Y/Z nao la truc cong -- tu rig suy ra duoc.
    private Vector3[][] _boneLocalDir;
    // Goc xoay o TU THE NGHI cua tung dot. Moi khung hinh ta dat lai ve day
    // TRUOC khi chinh huong -- neu khong, phep chinh huong se chong len ket
    // qua cua khung truoc, lam do XOAN quanh truc xuong tich luy dan va tay
    // bi van veo (dung trieu chung dang gap).
    private Quaternion[][] _boneRestLocalRotation;
    private FingerChainFitter[] _fitters; // moi ngon 1 bo giai goc khop (xem _useAnatomicalFit)
    private Transform _wrist;
    private int _lastFitFrame = -1;
    // 1 = bam hoan toan theo tay that, 0 = ve han tu the nghi. Chuyen dan giua
    // 2 gia tri nay de khong bi giat khi du lieu chap chon.
    private float _poseBlend;
    private volatile bool _dataValid;
    // Dat tu luong nhan UDP; doc/xoa o Update. KHONG goi Time.time trong luong
    // do duoc -- Unity API chi dung duoc tu luong chinh.
    private volatile bool _packetArrived;
    private float _lastDataTime = -999f;

    private UdpClient _udpClient;
    private Thread _receiveThread;
    private volatile bool _running;
    private readonly object _lock = new object();

    private void Awake()
    {
        _baseLocalRotation = transform.localRotation; // luu lai goc da calib thu cong truoc do
        _baseLocalPosition = transform.localPosition;  // luu lai vi tri da calib thu cong truoc do

        _rig = GetComponent<HandFingerRig>();
        _fingerJoints = new[]
        {
            new[] { _rig.thumbMetacarpal, _rig.thumbProximal, _rig.thumbDistal },
            new[] { _rig.indexProximal, _rig.indexIntermediate, _rig.indexDistal },
            new[] { _rig.middleProximal, _rig.middleIntermediate, _rig.middleDistal },
            new[] { _rig.ringProximal, _rig.ringIntermediate, _rig.ringDistal },
            new[] { _rig.pinkyProximal, _rig.pinkyIntermediate, _rig.pinkyDistal },
        };

        // Ep tat ca khop ve chung 1 moc 0 do -- KHONG doc tu the goc co san
        // cua tung khop, vi model co the co tu the nghi khong dong nhat
        // giua cac ngon (ngon nay hoi cong san, ngon kia thang), khien
        // bend=0 (mo hoan toan) khong dong nhat neu dung tu the goc rieng
        // cua tung khop lam moc.
        _baseAngleOnAxis = new float[_fingerJoints.Length][];
        for (int f = 0; f < _fingerJoints.Length; f++)
        {
            _baseAngleOnAxis[f] = new float[_fingerJoints[f].Length];
            for (int j = 0; j < _fingerJoints[f].Length; j++)
            {
                _baseAngleOnAxis[f][j] = 0f;
            }
        }

        // Truc XOE thi NGUOC LAI: giu nguyen goc nghi tu nhien cua rig lam
        // moc (spread=0 -> tay giu dung dang goc, da xac nhan la dung), roi
        // cong them goc xoe nhan duoc. Khong ep ve 0 nhu truc curl, vi goc
        // nghi cua ngon cai tren truc nay la mot phan cua dang tay tu nhien.
        Transform thumbBase = _fingerJoints[0][0];
        _thumbSpreadRestAngle = thumbBase != null
            ? GetAxis(thumbBase.localEulerAngles, _thumbSpreadAxis)
            : 0f;

        CacheBoneLocalDirections();
        CreateFitters();
    }

    /// <summary>Tao bo giai goc khop cho tung ngon, tu tu the nghi cua rig.</summary>
    private void CreateFitters()
    {
        _wrist = _rig.indexProximal != null ? _rig.indexProximal.parent : null;
        while (_wrist != null && !_wrist.name.Contains("Wrist")) _wrist = _wrist.parent;
        if (_wrist == null || _rig.middleProximal == null || _rig.pinkyProximal == null) return;

        Vector3 palmTarget = FingerChainFitter.PalmTarget(
            _wrist, _rig.indexProximal, _rig.middleProximal, _rig.pinkyProximal, _leftHand);

        _fitters = new FingerChainFitter[_fingerJoints.Length];
        for (int f = 0; f < _fingerJoints.Length; f++)
        {
            Transform[] chain = _fingerJoints[f];
            if (System.Array.IndexOf(chain, null) >= 0 || chain[2].childCount == 0) continue;
            _fitters[f] = new FingerChainFitter(chain, chain[2].GetChild(0), f == 0, palmTarget);
        }
    }

    /// <summary>Do huong "doc theo than xuong" cho tung dot, trong he toa do
    /// rieng cua chinh dot do. Xac dinh bang cach nhin xem khop KE TIEP nam ve
    /// phia nao -- nho vay khong can biet truoc rig dung truc X, Y hay Z lam
    /// truc xuong (moi model 3D mot kieu).</summary>
    private void CacheBoneLocalDirections()
    {
        _boneLocalDir = new Vector3[_fingerJoints.Length][];
        _boneRestLocalRotation = new Quaternion[_fingerJoints.Length][];
        for (int f = 0; f < _fingerJoints.Length; f++)
        {
            _boneLocalDir[f] = new Vector3[_fingerJoints[f].Length];
            _boneRestLocalRotation[f] = new Quaternion[_fingerJoints[f].Length];
            for (int j = 0; j < _fingerJoints[f].Length; j++)
            {
                Transform bone = _fingerJoints[f][j];
                if (bone == null) continue;
                _boneRestLocalRotation[f][j] = bone.localRotation;

                // Khop ke tiep: dot sau trong cung ngon, hoac neu la dot cuoi
                // (dot dau ngon) thi lay con dau tien cua no (thuong la diem
                // dau ngon trong rig).
                Transform next = (j + 1 < _fingerJoints[f].Length)
                    ? _fingerJoints[f][j + 1]
                    : (bone.childCount > 0 ? bone.GetChild(0) : null);

                Vector3 dirWorld = next != null
                    ? (next.position - bone.position)
                    : bone.forward;

                if (dirWorld.sqrMagnitude < 1e-10f) dirWorld = bone.forward;
                _boneLocalDir[f][j] = Quaternion.Inverse(bone.rotation) * dirWorld.normalized;
            }
        }
    }

    private void OnEnable()
    {
        _running = true;
        _receiveThread = new Thread(ReceiveLoop) { IsBackground = true };
        _receiveThread.Start();

        if (_handVisual != null)
        {
            _handVisual.WhenHandVisualUpdated += ApplyFingerCurl;
        }
    }

    private void OnDisable()
    {
        _running = false;
        _udpClient?.Close();
        _receiveThread?.Join(200);

        if (_handVisual != null)
        {
            _handVisual.WhenHandVisualUpdated -= ApplyFingerCurl;
        }
    }

    private void ReceiveLoop()
    {
        try
        {
            _udpClient = new UdpClient(_port);
            var remoteEP = new IPEndPoint(IPAddress.Any, _port);
            while (_running)
            {
                byte[] data = _udpClient.Receive(ref remoteEP); // chặn tới khi có gói tin
                string msg = System.Text.Encoding.UTF8.GetString(data);
                ParseAndStore(msg);
            }
        }
        catch (SocketException)
        {
            // Xảy ra bình thường khi đóng socket lúc OnDisable -- bỏ qua.
        }
    }

    private void ParseAndStore(string msg)
    {
        // Định dạng: "thumb:0.234,index:0.876,...,rx:12.3,ry:-4.5,rz:0.8"
        _packetArrived = true;
        var parts = msg.Split(',');
        lock (_lock)
        {
            foreach (var part in parts)
            {
                var kv = part.Split(':');
                if (kv.Length != 2) continue;
                string key = kv[0].Trim();

                // "pts" mang gia tri dang "x|y;x|y;..." (9 diem backbone), khong
                // phai 1 so don -- xu ly truoc khi thu parse thanh float.
                if (key == "pts") { ParsePoints(kv[1]); continue; }

                if (!float.TryParse(kv[1], System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture, out float v))
                    continue;

                if (key == "rx") { _latestRotEuler.x = v; continue; }
                if (key == "ry") { _latestRotEuler.y = v; continue; }
                if (key == "rz") { _latestRotEuler.z = v; continue; }
                if (key == "tx") { _latestTranslation.x = v; continue; }
                if (key == "ty") { _latestTranslation.y = v; continue; }
                if (key == "tz") { _latestTranslation.z = v; continue; }
                if (key == "pinch") { _latestPinch = Mathf.Clamp01(v); continue; }
                if (key == "spread") { _latestSpread = Mathf.Clamp01(v); continue; }
                // Python bao dang tay co hop ly khong (0 = doan sai, ve tu the nghi).
                if (key == "valid") { _dataValid = v > 0.5f; continue; }

                // "thumb0/1/2" va "index0/1/2" = goc that tung khop (goc/giua/dau),
                // rieng cho 2 ngon nay -- xem calculate_finger_joint_bends() ben Python.
                if (key.Length == 6 && key.StartsWith("thumb") && key[5] >= '0' && key[5] <= '2')
                {
                    _latestJointBend[0][key[5] - '0'] = Mathf.Clamp01(v);
                    continue;
                }
                if (key.Length == 6 && key.StartsWith("index") && key[5] >= '0' && key[5] <= '2')
                {
                    _latestJointBend[1][key[5] - '0'] = Mathf.Clamp01(v);
                    continue;
                }

                int idx = Array.IndexOf(_fingerNames, key);
                if (idx >= 0) _latestBend[idx] = Mathf.Clamp01(v);
            }
        }
    }

    /// <summary>Doc chuoi "x|y;x|y;..." (9 diem backbone) vao _latestPoints.
    /// Goi tu luong nhan UDP, da nam trong lock cua ParseAndStore.</summary>
    private void ParsePoints(string value)
    {
        var items = value.Split(';');
        if (items.Length < NumBackbonePoints) return;

        for (int i = 0; i < NumBackbonePoints; i++)
        {
            var xy = items[i].Split('|');
            if (xy.Length != 2) return;
            if (!float.TryParse(xy[0], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float px)) return;
            if (!float.TryParse(xy[1], System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out float py)) return;
            _latestPoints[i] = new Vector2(px, py);
        }
        _hasPoints = true;
    }

    private void Update()
    {
        Vector3 latestRot;
        Vector3 latestTrans;
        lock (_lock)
        {
            for (int f = 0; f < 5; f++)
                _currentBend[f] = Mathf.Lerp(_latestBend[f], _currentBend[f], _smoothing);
            for (int f = 0; f < 2; f++)
                for (int j = 0; j < 3; j++)
                    _currentJointBend[f][j] = Mathf.Lerp(_latestJointBend[f][j], _currentJointBend[f][j], _smoothing);
            _currentPinch = Mathf.Lerp(_latestPinch, _currentPinch, _pinchSmoothing);
            _currentSpread = Mathf.Lerp(_latestSpread, _currentSpread, _smoothing);
            if (_hasPoints)
            {
                for (int i = 0; i < NumBackbonePoints; i++)
                    _currentPoints[i] = Vector2.Lerp(_latestPoints[i], _currentPoints[i], _smoothing);
            }
            latestRot = _latestRotEuler;
            latestTrans = _latestTranslation;
        }

        // Fallback: chi bam theo tay that khi Python bao du lieu hop ly VA goi
        // tin con moi. Neu khong, chuyen dan ve tu the nghi (blend -> 0).
        if (_packetArrived)
        {
            _packetArrived = false;
            _lastDataTime = Time.time;
        }
        bool dataFresh = (Time.time - _lastDataTime) <= _dataTimeoutSeconds;
        float blendTarget = (_dataValid && dataFresh && _hasPoints) ? 1f : 0f;
        float blendStep = Time.deltaTime / Mathf.Max(_fallbackBlendSeconds, 1e-4f);
        _poseBlend = Mathf.MoveTowards(_poseBlend, blendTarget, blendStep);

        // Quy doi tu he truc cua webcam sang khong gian CUA DAU (local,
        // vi object nay la con cua CenterEyeAnchor) thong qua hieu chuan
        // camera-to-HEAD co dinh (_camHeadPosition/_camHeadEulerRotation).
        // Webcam gan tren dau kinh, di chuyen cung dau -- nen dung local
        // space (transform.localPosition/localRotation), KHONG dung world
        // space nua.
        Quaternion camToHead = Quaternion.Euler(_camHeadEulerRotation);

        if (_applyRotation)
        {
            _currentRotEuler = Vector3.Lerp(latestRot, _currentRotEuler, _rotationSmoothing);
            Vector3 signed = Vector3.Scale(_currentRotEuler, _rotationSign);
            Quaternion handRelativeToCam = Quaternion.Euler(signed);
            transform.localRotation = camToHead * handRelativeToCam * _baseLocalRotation;
        }

        if (_applyPosition)
        {
            _currentTranslation = Vector3.Lerp(latestTrans, _currentTranslation, _positionSmoothing);
            Vector3 camSpaceOffset = Vector3.Scale(_currentTranslation, _positionSign) * _positionScale;
            Vector3 headSpaceOffset = camToHead * camSpaceOffset;
            transform.localPosition = _camHeadPosition + headSpaceOffset + _baseLocalPosition;
        }
    }

    // LUON chay moi khung hinh, du co gan _handVisual hay khong -- day la
    // "duong bao dam" chinh. Neu KHONG lam vay ma chi dua vao event, va
    // event do vi ly do nao do khong ban ra deu (vd Hand.WhenHandUpdated
    // it/khong kich hoat), ngon tay se dung yen mai o tu the cu, dung
    // hoan toan im lang -- da tung xay ra dung nhu vay.
    private void LateUpdate()
    {
        ApplyFingerCurl();
    }

    // Goi dung 1 lan, NGAY SAU KHI HandVisual ghi xong toan bo khop cho
    // khung hinh hien tai (qua event WhenHandVisualUpdated) -- dam bao
    // du lieu ngon tay tu Python luon la gia tri "thang cuoi cung",
    // khong con phu thuoc doan mo thu tu Update/LateUpdate nua.
    // Model fine-tune hien tai CHI nhan dien on dinh ngon cai(0) va tro(1)
    // -- giua/ap ut/ut(2,3,4) van con nhay/sai nhieu do glove chua co du
    // lieu train da dang. Gioi han vong lap chi con 2 ngon nay de 3 ngon
    // kia KHONG BAO GIO bi script nay dong vao, du UDP gui gi di nua --
    // an toan hon so voi chi dua vao Python luon gui gia tri co dinh.
    private const int NumLiveTrackedFingers = 2;

    /// <summary>Xoay tung dot xuong sao cho no CHIA DUNG HUONG doan backbone
    /// tuong ung ma model detect duoc.
    ///
    /// Vi sao cach nay dung hon xoay quanh 1 truc co dinh: ngon tay that cong
    /// trong mat phang bat ky, khong trung voi truc curl da chon san -- nen du
    /// goc cong tinh chuan den may, xoay quanh sai truc thi hinh dang van khong
    /// bao gio khop. O day ta khong quan tam truc nao ca: chi don gian xoay
    /// xuong tu huong hien tai sang huong can den.
    ///
    /// Vi camera gan TREN DAU KINH nen mat phang anh ~ mat phang ban dang nhin:
    /// chia dung huong 2D thi tay ao trong dung dang tu goc nhin cua ban. Do
    /// sau (z) khong co trong du lieu 2D nen cac dot nam trong mat phang nhin.
    ///
    /// Thu tu quan trong: xoay dot GOC truoc, vi xoay cha se keo theo ca con.</summary>
    private void ApplyBackboneRetarget()
    {
        Transform view = _viewReference != null ? _viewReference : transform.parent;
        if (view == null || _boneLocalDir == null) return;

        // Dieu khien CA 5 NGON. Rieng giua/ap ut/ut: khi dang pinch chung bi
        // chinh ban tay che khuat nen model doan bay -- nhung viec ep chung ve
        // dang nam da duoc xu ly BEN PYTHON (xem apply_mrp_closed_pose), toa do
        // gui sang day da la toa do dung roi. Unity chi viec bam theo.
        for (int f = 0; f < _fingerJoints.Length; f++)
        {
            int baseIdx = FingerBasePointIndex[f];

            for (int j = 0; j < _fingerJoints[f].Length; j++)
            {
                Transform bone = _fingerJoints[f][j];
                if (bone == null) continue;

                // Dat lai ve tu the NGHI truoc khi chinh huong. Neu bo buoc
                // nay, phep chinh huong cua khung hinh nay se chong len ket
                // qua khung truoc, khien do xoan quanh truc xuong tich luy
                // dan qua tung khung -> tay van veo.
                bone.localRotation = _boneRestLocalRotation[f][j];

                Vector2 segment = _currentPoints[baseIdx + j + 1] - _currentPoints[baseIdx + j];
                if (segment.sqrMagnitude < 1e-8f) continue;
                segment.Normalize();

                Vector3 restWorld = (bone.rotation * _boneLocalDir[f][j]).normalized;
                if (restWorld.sqrMagnitude < 1e-8f) continue;

                Vector3 planarDir = view.right * segment.x + view.up * segment.y;
                Vector3 targetWorld;
                if (_preserveRestDepth)
                {
                    // Giu nguyen phan "chia vao/ra man hinh" cua tu the nghi,
                    // chi xoay lai phan nam TRONG mat phang nhin cho khop
                    // backbone. Nho vay ngon tay khong bi ep det thanh mat
                    // phang (du lieu 2D khong co chieu sau that).
                    float depth = Vector3.Dot(restWorld, view.forward);
                    float planarScale = Mathf.Sqrt(Mathf.Max(0f, 1f - depth * depth));
                    targetWorld = planarDir * planarScale + view.forward * depth;
                }
                else
                {
                    targetWorld = planarDir;
                }

                if (targetWorld.sqrMagnitude < 1e-8f) continue;
                bone.rotation = Quaternion.FromToRotation(restWorld, targetWorld.normalized) * bone.rotation;

                // Tron ve tu the nghi theo _poseBlend: 1 = bam hoan toan theo
                // tay that, 0 = ve han dang nghi binh thuong (khi Python bao
                // du lieu khong hop ly, hoac mat tin hieu).
                if (_poseBlend < 1f)
                {
                    bone.localRotation = Quaternion.Slerp(
                        _boneRestLocalRotation[f][j], bone.localRotation, _poseBlend);
                }
            }
        }
    }

    /// <summary>Dung lai dang 3D cua ca 5 ngon tu 21 diem 2D (xem FingerChainFitter).</summary>
    private void ApplyAnatomicalFit()
    {
        Transform view = _viewReference != null ? _viewReference : transform.parent;
        if (view == null) return;

        // Diem tu Python da chia cho chieu dai long ban tay TREN ANH. Lam tuong
        // tu voi tay ao: chieu long ban tay ao (huong lay tu Quest) len mat
        // phang nhin -- long ban tay nghieng thi ca 2 ben cung ngan lai nhu nhau.
        Vector3 palm = _rig.middleProximal.position - _wrist.position;
        Vector2 palmOnImage = new Vector2(Vector3.Dot(palm, view.right), Vector3.Dot(palm, view.up));
        // Long ban tay gan nhu vuong goc voi mat phang anh -> chieu dai tren anh
        // qua ngan, chia cho no se khuech dai nhieu; chan duoi o 35%.
        float palm2D = Mathf.Max(palmOnImage.magnitude, 0.35f * palm.magnitude);

        // Ham nay chay 2 lan moi khung (event cua HandVisual + LateUpdate):
        // chi GIAI 1 lan, lan sau chi ghi lai goc da co vao xuong.
        bool solve = _poseBlend > 0f && Time.frameCount != _lastFitFrame;
        if (solve) _lastFitFrame = Time.frameCount;

        for (int f = 0; f < _fitters.Length; f++)
        {
            FingerChainFitter fitter = _fitters[f];
            if (fitter == null) continue;

            if (solve)
            {
                int b = FingerBasePointIndex[f];
                Vector2 root = _currentPoints[b];
                fitter.Solve(_currentPoints[b + 1] - root, _currentPoints[b + 2] - root, _currentPoints[b + 3] - root,
                             view.right, view.up, palm2D);
            }
            // Tron ve tu the nghi khi du lieu khong dang tin (giong cach cu)
            fitter.Apply(_poseBlend);
        }
    }

    private void ApplyFingerCurl()
    {
        if (_useBackboneRetarget)
        {
            if (!_hasPoints) return;
            if (_useAnatomicalFit && _fitters != null) ApplyAnatomicalFit();
            else ApplyBackboneRetarget();
            return;
        }

        for (int f = 0; f < NumLiveTrackedFingers; f++)
        {
            bool isThumb = f == 0;
            Axis axis = isThumb ? _thumbCurlAxis : _curlAxis;
            bool invert = isThumb ? _thumbInvert : _invert;

            // Dang PINCH: CHI ap dung rieng cho ngon cai (f=0) va ngon
            // tro (f=1) -- 3 ngon con lai khong lien quan gi pinch, van
            // dung logic bend/curl thong thuong nhu cu du dang pinch
            // hay khong.
            Vector3? pinchAngles = f == 0 ? _pinchThumbAngles : f == 1 ? _pinchIndexAngles : null;

            for (int j = 0; j < _fingerJoints[f].Length; j++)
            {
                Transform joint = _fingerJoints[f][j];
                if (joint == null) continue;

                // Goc cong THAT cua rieng KHOP NAY (gia tri doc lap 0..1,
                // tinh tu hinh dang that cua 8 diem model detect duoc --
                // xem run_glove_quest_stream.py: calculate_finger_joint_bends),
                // KHONG con la 1 gia tri chung nhan ty le co dinh nhu truoc.
                float raw = _currentJointBend[f][j];
                // Vung chet: bo qua nhieu nho khi khop gan nhu thang.
                float shaped = raw <= _deadzone ? 0f : (raw - _deadzone) / (1f - _deadzone);
                float signedAmplitude = shaped * _amplitudeDegrees * (invert ? -1f : 1f);

                // jointRatio van giu lai lam he so tinh chinh them cho tung
                // khop (vd neu khop dau nhin "qua cong" so voi khop giua du
                // du lieu that dung, co the giam _distalCurlRatio de bu).
                float jointRatio = j switch
                {
                    0 => _proximalCurlRatio,
                    1 => _intermediateCurlRatio,
                    _ => _distalCurlRatio,
                };
                float offset = signedAmplitude * jointRatio;

                // Tron dan sang dang PINCH (rieng cho khop nay cua ngon
                // cai/tro) -- CHI khi pinch vuot qua nguong (gan cham hoan
                // toan), khong phai tu pinch>0 nhu truoc. Duoi nguong, ty
                // le tron = 0 -- offset la du lieu cong khop THAT 100%,
                // khong bi keo ve dang co dinh (chua tinh chinh) lam sai
                // lech cac tu the binh thuong (vd chu "C" co pinch~0.3).
                float pinchBlend = Mathf.Clamp01((_currentPinch - _pinchBlendStartThreshold) / (1f - _pinchBlendStartThreshold));
                if (pinchAngles.HasValue && pinchBlend > 0f)
                {
                    float pinchTarget = j switch
                    {
                        0 => pinchAngles.Value.x,
                        1 => pinchAngles.Value.y,
                        _ => pinchAngles.Value.z,
                    };
                    offset = Mathf.Lerp(offset, pinchTarget, pinchBlend);
                }

                Vector3 e = joint.localEulerAngles;
                SetAxis(ref e, _baseAngleOnAxis[f][j] + offset, axis);

                // DO XOE: chi ap cho xuong GOC cua ngon cai (j==0), tren mot
                // truc KHAC voi truc curl. Day la bac tu do tao ra khoang ho
                // cai<->tro (hinh chu "C") -- thu ma rieng do cong khong the
                // tao ra du hieu chinh kieu gi.
                if (_applyThumbSpread && isThumb && j == 0 && _thumbSpreadAxis != axis)
                {
                    float spreadOffset = _currentSpread * _thumbSpreadAmplitudeDegrees
                                         * (_thumbSpreadInvert ? -1f : 1f);
                    SetAxis(ref e, _thumbSpreadRestAngle + spreadOffset, _thumbSpreadAxis);
                }

                joint.localEulerAngles = e;
            }
        }
    }

    private static float GetAxis(Vector3 e, Axis axis) => axis switch
    {
        Axis.X => e.x,
        Axis.Y => e.y,
        _ => e.z,
    };

    private static void SetAxis(ref Vector3 e, float value, Axis axis)
    {
        switch (axis)
        {
            case Axis.X: e.x = value; break;
            case Axis.Y: e.y = value; break;
            default: e.z = value; break;
        }
    }
}
