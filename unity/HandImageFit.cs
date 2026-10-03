using UnityEngine;

/// <summary>
/// Dung ban tay ao (co tay + ngon cai + ngon tro) TU ANH: moi diem model thay tren anh la 1 TIA tu
/// camera; tim vi tri/huong co tay va goc cac khop sao cho moi khop ao nam tren tia cua no.
///
/// Khac cach cu (FingerUDPReceiver.CorrectHandToImage + FingerChainFitter rieng tung ngon):
/// cach cu lay co tay QUEST (gang moi -> sai, rung) lam nen roi va 6 lop len tren, giai lai 60
/// lan/giay theo co tay dang rung. O day anh la nguon chinh, giai 1 lan cho MOI anh; Quest chi
/// gop phan anh khong cho biet (tay cach camera bao xa, long ban tay up hay ngua) qua rang buoc nhe.
///
/// Anh 2D khong co chieu sau, nen bu bang dieu biet chac: xuong co chieu dai co dinh (doan ngan tren
/// anh = dang chia vao/ra camera), khop chi gap 1 chieu trong gioi han, tay khong lat trong 0.1 s
/// (gan dap an anh truoc), chieu sau gan Quest (da lam muot).
///
/// Thuan C# (khong MonoBehaviour) de chay thu trong Editor tren ban ghi.
/// </summary>
public sealed class HandImageFit
{
    public const int Points = 9; // 0 co tay, 1-4 ngon cai (goc -> dau), 5-8 ngon tro
    private const int NP = 14;   // [0..2] dich co tay (cm), [3..5] xoay co tay (do, vector quay the gioi), [6..9] cai, [10..13] tro
    private const int NR = 2 * Points + 1 + 3 + 3 + 3 + 8 + 6 + 8 + 2 + 3 + 3; // ... + 6 chong be nguoc, 8 dang tay hoc tu tay tran, 2 cong an, 3 cum luc giac, 3 pinch

    // Trong so (don vi: sai so anh tinh bang CHIEU DAI LONG BAN TAY o chieu sau cua tay).
    // Sai 10 pixel ~ 0.05. Static de chinh thu tren ban ghi.
    public static float DepthPriorWeight = 0.1f;     // moi 1 long ban tay lech chieu sau so voi Quest
    public static float RotPriorWeight = 0.0015f;    // moi do lech huong Quest (Quest rung 30-50 do -> chi de chon up/ngua)
    public static float TemporalPosWeight = 0.4f;    // moi 1 long ban tay doi so voi anh truoc (gia lap nhieu 8 px: 0.05 -> nhay 56/145 anh, 0.4 -> 0)
    public static float TemporalRotWeight = 0.012f;  // moi do doi so voi anh truoc (ban ghi 01/10 16:58: 0.006 -> xoay >5 do/khung 102 lan, 0.012 -> 38)
    public static float TemporalAngleWeight = 0.006f;
    public static float PinchWeight = 2f;            // 2 dau ngon cach nhau hon PinchContactDistance khi anh thay dang chum
    public static float PinchContactDistance = 0.016f;
    public static float PinchNearOnImage = 0.2f;     // khoang cach 2 dau ngon tren anh / |co tay - goc ngon tro|
    public static float PinchFarOnImage = 0.45f;
    public static float SwitchRatio = 0.7f;          // doi sang cach hieu khac khi sai so thap hon it nhat 30%
    // Cum mieng luc giac mau be tren dot goc 3 ngon dang nam (Python tim theo mau): diem so 10. 9 diem
    // co tay/cai/tro gan nhu khong cho biet ban tay LAT quanh truc co tay -> goc ngon tro bao nhieu
    // (chi diem 1 lech khoi truc ~2 cm, lai la diem model dat kem nhat) -> tay ao hay nghieng canh sai.
    public static float DorsalWeight = 1.0f;         // sai so anh cua tam cum. Ban ghi 01/10 17:45 (611 anh thay cum): mat ban tay tren anh rong 0.11 -> 0.29 long ban tay (dung voi anh), sai so 9 diem 0.070 -> 0.074
    public static float DorsalHiddenWeight = 0.3f;   // khong thay cum ma mat cum lai quay ve camera
    public static float DorsalHiddenFacing = 0.35f;  // cos goc giua phap tuyen mat cum va huong ve camera, tren muc nay thi phai thay
    public static float DorsalOffset = 0.012f;       // mieng luc giac + vai gang nho len khoi xuong (met)
    // Diem CO TAY (diem 0) model dat tren anh nam o dau so voi khop co tay cua tay ao (he co tay, met).
    public static Vector3 WristPointLocal = Vector3.zero;
    // Diem GOC NGON TRO (diem 5) model dat lech bao nhieu so voi khop goc ngon tro tay ao (he co tay, met).
    // Model gang moi dat diem 5 gan diem 9 (giua cac khop goc), lech ve phia ngon ut ~2 cm (+X cua co tay
    // XRHand). Ban ghi 01/10 19:07 + 19:26: doi 2 cm -> do xoe ngon tro med -30 -> -4 do, ngon cai be nguoc
    // 23% -> 1% thoi gian, sai so anh 0.063/0.068 -> 0.061/0.065.
    public static Vector3 IndexMcpPointLocal = new Vector3(0.02f, 0f, 0f);
    private const float HyperextensionWeight = 0.006f;

    // DANG TAY HOC TU TAY TRAN (HandPoseRecorder: tay trai Quest theo doi, 02/10 11:31 + 12:15, 13959 khung, 1704
    // khung pinch chat; chuyen sang goc khop tay gang -- lech huong tung dot ~0.6-1.3 do). 2 kieu pinch: 11:31 khop
    // goc gap nhieu (47/37/18), 12:15 (cam bong ao) khop giua gap nhieu (19/47/33) -- ma tran tuong quan rong du cho ca 2. Thay cho cac rang buoc dat tay truoc day
    // (khop dau = 2/3 khop giua, khop giua >= khop goc, dang pinch 35/55/35): tay that khi pinch la ~47/37/18,
    // 54% thoi gian khop goc gap hon khop giua 10 do, khop dau ~0.55 khop giua.
    // Mo hinh: goc trung binh noi suy theo muc pinch (mo -> chum) + ma tran tuong quan 8 goc (cai, tro: xoe/goc/
    // giua/dau). Phat = khoang cach Mahalanobis -> cac khop di cung nhau nhu tay that.
    // 03/10: hoc lai sau khi FingerChainFitter.StraightenFingers (goc 0 = ngon THANG): khop dau ngon tro mo tay
    // 9.4 -> 1.0 do, pinch 32.9 -> 24.6 do (truoc do so voi dot cuoi venh nguoc ~8 do cua rig).
    // CONG AN cua ngon tro: phan gap 3D ma anh KHONG thay (ngon cong ve phia / ra xa camera trong anh van thang).
    // Mot camera khong phan biet duoc -> bo giai co the them cong an ma van khop anh; nhin tu mat (khac goc camera)
    // thanh ngon cong queo. Phat (gap 3D - gap tren anh) o khop giua + khop dau, moi do.
    // 03/10 (co cong StraightOnImage), ban ghi 13:44 tay phai->trai gan/xa/pinch: cong an DIP p90 10-13 -> 4-5 do,
    // gap DIP p90 28-32 -> 21 do, sai so anh khong doi, khe 2 dau ngon khi pinch 1.9 -> 2.0 cm. Synthetic: huong co tay
    // trung vi 2-6 -> 4-5 do (khong cong: 6 do). 0.032 chi tot them chut, giat hon o ban ghi voi xa.
    public static float HiddenBendWeight = 0.016f;
    // Chi phat khi ANH cho thay ngon thang: dot giua + dot cuoi (diem 6-7-8) gay it va KHONG bi co ngan (ngon cong
    // ve phia camera trong anh ngan di). Phat deu tay lam sai ngon cong that theo chieu sau: bai Synthetic (ngon cong
    // kieu pinch, co dap an) huong co tay lech 2 -> 6 do. Ban ghi 03/10: nhin ngang gay tren anh p90 7-9 do, dai >= 0.32
    // long ban tay; chia doc truc nhin dai 0.11-0.22.
    public static float StraightBendFull = 8f, StraightBendZero = 15f;      // gay tren anh (do) tai diem 7
    public static float StraightLenZero = 0.22f, StraightLenFull = 0.30f;   // (|6-7| + |7-8|) / |0-5| tren anh
    public static float PosePriorWeight = 0.02f;     // moi 1 do lech chuan (sai 2 do lech chuan ~ sai anh 0.04 long ban tay)
    private static readonly float[] PoseMeanOpen = { 0.8791f, 4.2680f, 27.0197f, 0.9191f, 2.7536f, 11.5433f, 18.6156f, 1.0288f };
    private static readonly float[] PoseMeanPinch = { 7.0680f, 11.1249f, 38.0017f, 15.8167f, 2.2528f, 18.5744f, 47.1248f, 24.5913f };
    private static readonly float[,] PoseWhiten = // nghich dao Cholesky cua ma tran hiep phuong sai (don vi: 1/do)
    {
        { 0.0670f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f },
        { 0.0236f, 0.1975f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f },
        { -0.0074f, 0.0392f, 0.1342f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f },
        { -0.0255f, -0.0233f, -0.1364f, 0.1442f, 0.0000f, 0.0000f, 0.0000f, 0.0000f },
        { 0.0238f, -0.0537f, 0.0245f, -0.0116f, 0.2629f, 0.0000f, 0.0000f, 0.0000f },
        { 0.0496f, -0.0119f, 0.0081f, 0.0123f, -0.0563f, 0.0913f, 0.0000f, 0.0000f },
        { 0.0355f, -0.0108f, -0.0217f, -0.0143f, 0.0412f, 0.0032f, 0.1173f, 0.0000f },
        { -0.0627f, -0.0315f, 0.0120f, -0.0200f, 0.0055f, 0.0025f, -0.1728f, 0.3121f },
    };
    private readonly float[] _poseDev = new float[8];
    private Vector3 _indexMcpJoint; // khop goc ngon tro THAT (diem 5 tren anh lech khoi no, xem IndexMcpPointLocal)

    private readonly FingerChainFitter _thumb, _index;
    private Vector3 _thumbRootL, _indexRootL;                  // goc ngon trong he co tay (met, theo Scale)
    private readonly Quaternion _thumbParentL, _indexParentL;  // huong xuong cha cua goc ngon, trong he co tay
    private readonly float[] _min = new float[8], _max = new float[8];
    private bool _hasDorsalModel;
    private Vector3 _dorsalL, _dorsalNL;                       // tam cum luc giac + phap tuyen mat cum, trong he co tay
    private Vector2 _dorsalTarget;
    private int _dorsalState = -1;                             // -1 khong biet, 0 khong thay, 1 thay

    /// <summary>Co tay -> goc ngon tro cua tay ao (met).</summary>
    public float PalmLength { get; private set; }
    /// <summary>Ti le kich thuoc tay hien tai so voi luc tao (HandVisual doi kich thuoc tay khi chay).</summary>
    public float Scale { get; private set; } = 1f;
    private float _palmLength0;
    private Vector3 _thumbRootL0, _indexRootL0, _dorsalL0;

    // Du lieu anh dang giai
    private Vector3 _camPos;
    private Quaternion _camRot, _camInv;
    private readonly Vector2[] _target = new Vector2[Points]; // toa do chuan hoa x/z, y/z trong he camera
    private readonly float[] _weight = new float[Points];
    private float _imgScale;
    private float _depthPrior;
    private bool _useDepthPrior;
    private Quaternion _rotPrior;
    private bool _useRotPrior;
    private bool _useTemporal;
    private Vector3 _prevP;
    private Quaternion _prevR;
    private readonly float[] _prevA = new float[8];
    private Vector3 _baseP;
    private Quaternion _baseR;

    /// <summary>Ket qua: co tay (the gioi).</summary>
    public Vector3 Position { get; private set; }
    public Quaternion Rotation { get; private set; } = Quaternion.identity;
    /// <summary>[0..3] ngon cai, [4..7] ngon tro: xoe, gap goc, gap giua, gap dau (do).</summary>
    public readonly float[] Angles = new float[8];
    public bool HasSolution { get; private set; }
    /// <summary>Sai so khop anh trung binh (don vi long ban tay; 0.05 ~ 4-5 mm).</summary>
    public float ImageError { get; private set; }
    public float LastCost { get; private set; }
    public float Pinch { get; private set; }
    public int Switches { get; private set; }
    /// <summary>Chieu sau goc ngon tro (truc nhin camera, met) cua ket qua.</summary>
    public float Depth { get; private set; }
    /// <summary>Cum luc giac dung trong lan giai vua roi: -1 khong biet, 0 khong thay, 1 thay.</summary>
    public int DorsalUsed => _dorsalState;
    /// <summary>0..1: anh cho thay ngon tro thang ro rang den dau (xem StraightBendFull) -- he so cua phat cong an.</summary>
    public float StraightOnImage { get; private set; }

    private int _nextSeed;
    private static readonly float[,] FingerSeeds =
    {
        { 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f },        // duoi thang
        { 0f, 15f, 30f, 20f, 0f, 30f, 40f, 25f },  // nua nam / chuan bi pinch
        { 0f, 30f, 50f, 40f, 0f, 60f, 80f, 50f },  // chum
    };
    // Huong mau (do): x = nghieng quanh truc ngang tren anh, y = lat quanh truc co tay -> goc ngon tro
    private static readonly Vector2[] TiltSeeds = { Vector2.zero, new Vector2(35f, 0f), new Vector2(-35f, 0f), new Vector2(0f, 70f), new Vector2(0f, -70f) };

    private readonly float[] _x = new float[NP], _trial = new float[NP], _a8 = new float[8], _bestA = new float[8];
    private readonly float[] _r = new float[NR], _rTry = new float[NR];
    private readonly float[,] _jac = new float[NR, NP];
    private readonly float[,] _normal = new float[NP, NP + 1], _sys = new float[NP, NP + 1];
    private readonly float[] _a4 = new float[4];
    private readonly Vector3[] _chain = new Vector3[4];
    private readonly Vector3[] _model = new Vector3[Points];
    private static readonly float[] Step = { 0.05f, 0.05f, 0.05f, 0.2f, 0.2f, 0.2f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f, 0.3f };
    private static readonly float[] MaxStep = { 5f, 5f, 5f, 20f, 20f, 20f, 25f, 25f, 25f, 25f, 25f, 25f, 25f, 25f };

    /// <summary>Doc hinh dang ban tay ao O TU THE NGHI (goi luc Awake, cung luc tao 2 FingerChainFitter).</summary>
    public HandImageFit(Transform wrist, FingerChainFitter thumb, FingerChainFitter index)
        : this(wrist, new[] { thumb, index }, Vector3.zero, Vector3.zero) { }

    /// <param name="fitters">5 ngon (cai, tro, giua, ap ut, ut); 3 ngon sau de tinh vi tri cum luc giac (co the null).</param>
    /// <param name="palmTarget">Diem phia LONG ban tay (FingerChainFitter.PalmTarget) -- de biet phia mu ban tay.</param>
    /// <param name="closedAngles">Goc nam cua 3 ngon giua/ap ut/ut (FingerUDPReceiver).</param>
    public HandImageFit(Transform wrist, FingerChainFitter[] fitters, Vector3 palmTarget, Vector3 closedAngles)
    {
        FingerChainFitter thumb = fitters[0], index = fitters[1];
        _thumb = thumb;
        _index = index;
        Quaternion inv = Quaternion.Inverse(wrist.rotation);
        Transform tr = thumb.RootBone, ir = index.RootBone;
        _thumbRootL = inv * (tr.position - wrist.position);
        _indexRootL = inv * (ir.position - wrist.position);
        _thumbParentL = inv * (tr.parent != null ? tr.parent.rotation : wrist.rotation);
        _indexParentL = inv * (ir.parent != null ? ir.parent.rotation : wrist.rotation);
        PalmLength = _palmLength0 = Mathf.Max(_indexRootL.magnitude, 0.03f);
        _thumbRootL0 = _thumbRootL;
        _indexRootL0 = _indexRootL;
        for (int p = 0; p < 4; p++)
        {
            _min[p] = thumb.MinAngle(p); _max[p] = thumb.MaxAngle(p);
            _min[4 + p] = index.MinAngle(p); _max[4 + p] = index.MaxAngle(p);
        }

        // Tam cum luc giac: giua dot goc (MCP -> PIP) cua 3 ngon dang nam, nho len phia mu ngon.
        // Mat cum quay ve phia TRUOC nam tay (dot goc gap ~75 do), khong quay len nhu mu ban tay.
        if (fitters.Length >= 5 && palmTarget != Vector3.zero)
        {
            Vector3 mid = Vector3.zero, along = Vector3.zero, palmCenter = wrist.position + index.RootBone.position;
            int n = 0, nc = 2;
            float[] a = { 0f, closedAngles.x, closedAngles.y, closedAngles.z };
            var joints = new Vector3[4];
            for (int f = 2; f < 5; f++)
            {
                FingerChainFitter fit = fitters[f];
                if (fit == null) continue;
                Transform root = fit.RootBone;
                palmCenter += root.position; nc++;
                fit.JointsAt(a, root.position, root.parent != null ? root.parent.rotation : wrist.rotation, joints);
                mid += (joints[0] + joints[1]) * 0.5f;
                along += (joints[1] - joints[0]).normalized;
                n++;
            }
            if (n > 0)
            {
                mid /= n;
                palmCenter /= nc;
                Vector3 back = (palmCenter - palmTarget).normalized; // huong mu ban tay
                Vector3 normal = Vector3.ProjectOnPlane(back, along.normalized).normalized;
                _dorsalL = _dorsalL0 = inv * (mid + normal * DorsalOffset - wrist.position);
                _dorsalNL = inv * normal;
                _hasDorsalModel = true;
            }
        }
    }

    /// <summary>Dat vi tri tam cum luc giac + phap tuyen mat cum (he co tay, met) thay cho uoc luong tu rig.</summary>
    public void SetDorsalModel(Vector3 pointLocal, Vector3 normalLocal)
    {
        _dorsalL0 = pointLocal;
        _dorsalL = pointLocal * Scale;
        _dorsalNL = normalLocal.normalized;
        _hasDorsalModel = normalLocal.sqrMagnitude > 1e-8f;
    }

    /// <summary>Dat ti le kich thuoc tay so voi luc tao. HandVisual (Quest) doi kich thuoc goc ban tay khi chay;
    /// xuong ngon (FingerChainFitter) doc kich thuoc HIEN TAI, nen long ban tay o day phai theo cung ti le --
    /// truoc day chup 1 lan luc khoi dong (0.9) trong khi tay ve ra ~1.0 -> bo giai dung ban tay nho hon ~10%,
    /// dat tay gan camera hon, nhin tu mat tay ao to va lech so voi tay that.</summary>
    public void SetScale(float k)
    {
        if (k <= 0f || Mathf.Abs(k - Scale) < 1e-4f) return;
        Scale = k;
        _thumbRootL = _thumbRootL0 * k;
        _indexRootL = _indexRootL0 * k;
        _dorsalL = _dorsalL0 * k;
        PalmLength = _palmLength0 * k;
    }

    /// <summary>Tam cum luc giac cua ket qua hien tai (the gioi) -- de ve/kiem tra.</summary>
    public bool TryGetDorsalPoint(out Vector3 point)
    {
        point = Position + Rotation * _dorsalL;
        return _hasDorsalModel;
    }

    public void Reset()
    {
        HasSolution = false;
    }

    /// <summary>Giai cho 1 anh.</summary>
    /// <param name="camPos">Vi tri camera LUC CHUP anh.</param>
    /// <param name="camRot">Huong camera luc chup.</param>
    /// <param name="rayDirs">Huong tia (the gioi) tu camera qua 9 diem anh.</param>
    /// <param name="weights">Do tin 0..1 tung diem (0 = bo qua). Can it nhat diem 0 va 5.</param>
    /// <param name="depthPrior">Chieu sau tay theo Quest (truc nhin camera, met); &lt;= 0 = khong co.</param>
    /// <param name="rotPrior">Huong co tay theo Quest (da lam muot); null = khong co.</param>
    /// <param name="warm">Dap an truoc con moi (vai tram ms) -> bat dau tu do va giu gan no.</param>
    /// <param name="dorsalState">Cum luc giac tren mu ban tay: -1 khong biet, 0 Python tim ma khong thay, 1 thay.</param>
    /// <param name="dorsalRayDir">Tia qua tam cum (khi dorsalState = 1).</param>
    public bool Solve(Vector3 camPos, Quaternion camRot, Vector3[] rayDirs, float[] weights,
                      float depthPrior, Quaternion? rotPrior, bool warm, int dorsalState = -1, Vector3 dorsalRayDir = default)
    {
        _camPos = camPos;
        _camRot = camRot;
        _camInv = Quaternion.Inverse(camRot);
        _dorsalState = _hasDorsalModel ? dorsalState : -1;
        if (_dorsalState == 1)
        {
            Vector3 dd = _camInv * dorsalRayDir;
            if (dd.z > 1e-3f) _dorsalTarget = new Vector2(dd.x / dd.z, dd.y / dd.z);
            else _dorsalState = -1;
        }
        for (int i = 0; i < Points; i++)
        {
            Vector3 d = _camInv * rayDirs[i];
            bool ok = d.z > 1e-3f && weights[i] > 0f;
            _target[i] = ok ? new Vector2(d.x / d.z, d.y / d.z) : Vector2.zero;
            _weight[i] = ok ? weights[i] : 0f;
        }
        if (_weight[0] <= 0f || _weight[5] <= 0f) return false;

        float palmOnImage = Vector2.Distance(_target[0], _target[5]);
        float gap = Vector2.Distance(_target[4], _target[8]) / Mathf.Max(palmOnImage, 1e-4f);
        Pinch = _weight[4] > 0f && _weight[8] > 0f
            ? Mathf.Clamp01((PinchFarOnImage - gap) / Mathf.Max(PinchFarOnImage - PinchNearOnImage, 1e-3f)) : 0f;
        StraightOnImage = StraightGate(palmOnImage);

        warm &= HasSolution;
        _useDepthPrior = depthPrior > 0.05f;
        _depthPrior = depthPrior;
        _useRotPrior = rotPrior.HasValue;
        _rotPrior = rotPrior ?? Quaternion.identity;
        _useTemporal = warm;
        if (warm)
        {
            _prevP = Position;
            _prevR = Rotation;
            System.Array.Copy(Angles, _prevA, 8);
        }
        float depth = _useDepthPrior ? depthPrior : HasSolution ? Depth : 0.4f;
        _imgScale = depth / PalmLength;

        Quaternion rotGuess = _useRotPrior ? _rotPrior : HasSolution ? Rotation : FacingCamera();
        float bestCost = float.PositiveInfinity;
        Vector3 bestP = Vector3.zero;
        Quaternion bestR = Quaternion.identity;

        if (warm)
        {
            bestCost = RefineFrom(Position, Rotation, Angles, out bestP, out bestR, _bestA);
            // Moi anh thu them 1 cach hieu khac (luan phien dang ngon x do nghieng): anh 2D hay co
            // 2 dang gan giong nhau; doi khi cach moi khop RO RANG tot hon. So sanh KHONG tinh phat "khac
            // anh truoc" -- tinh vao thi cach moi (khac han dap an dang dung) khong bao gio thang, tay ao
            // ket mai o dang sai (truoc day so lan doi = 0 o moi ban ghi).
            // Cong bang: ca 2 cach hieu deu giai KHONG co phat "khac anh truoc" (cach dang dung giai tiep
            // tu cho cua no) roi moi so. Dap an dang dung bi keo ve anh truoc nen sai so anh cua no luon
            // cao hon -- so thang voi no thi cach moi gan nhu luon thang, tay ao doi qua doi lai.
            int combos = FingerSeeds.GetLength(0) * TiltSeeds.Length;
            int c = _nextSeed;
            _nextSeed = (_nextSeed + 1) % combos;
            _useTemporal = false;
            float warmStatic = RefineFrom(bestP, bestR, _bestA, out _, out _, _a8);
            float cost = TrySeed(rotGuess, depth, c / TiltSeeds.Length, c % TiltSeeds.Length, out Vector3 p, out Quaternion r);
            _useTemporal = true;
            if (cost < warmStatic * SwitchRatio)
            {
                bestCost = cost; bestP = p; bestR = r;
                System.Array.Copy(_a8, _bestA, 8);
                Switches++;
            }
        }
        else
        {
            for (int s = 0; s < FingerSeeds.GetLength(0); s++)
                for (int t = 0; t < TiltSeeds.Length; t++)
                {
                    float cost = TrySeed(rotGuess, depth, s, t, out Vector3 p, out Quaternion r);
                    if (cost < bestCost)
                    {
                        bestCost = cost; bestP = p; bestR = r;
                        System.Array.Copy(_a8, _bestA, 8);
                    }
                }
        }

        Position = bestP;
        Rotation = bestR;
        System.Array.Copy(_bestA, Angles, 8);
        LastCost = bestCost;
        HasSolution = true;

        // Sai so khop anh rieng (khong tinh rang buoc) -- de chan doan
        _baseP = Position; _baseR = Rotation;
        for (int k = 0; k < 6; k++) _x[k] = 0f;
        System.Array.Copy(Angles, 0, _x, 6, 8);
        Evaluate(_x, _r);
        float sum = 0f, wsum = 0f;
        for (int i = 0; i < Points; i++)
        {
            if (_weight[i] <= 0f) continue;
            sum += (_r[2 * i] * _r[2 * i] + _r[2 * i + 1] * _r[2 * i + 1]) / (_weight[i] * _weight[i]);
            wsum += 1f;
        }
        ImageError = Mathf.Sqrt(sum / Mathf.Max(wsum, 1f));
        Depth = (_camInv * (_model[5] - _camPos)).z;
        return true;
    }

    /// <summary>Dat dap an hien tai (vd de chay thu: bat dau tu dang sai kinh da chon).</summary>
    public void SetSolution(Vector3 p, Quaternion r, float[] a)
    {
        Position = p;
        Rotation = r;
        System.Array.Copy(a, Angles, 8);
        HasSolution = true;
    }

    /// <summary>Vi tri 9 khop cua ket qua hien tai (the gioi) -- de ve/kiem tra.</summary>
    public void GetModelPoints(Vector3[] outPoints)
    {
        _baseP = Position; _baseR = Rotation;
        for (int k = 0; k < 6; k++) _x[k] = 0f;
        System.Array.Copy(Angles, 0, _x, 6, 8);
        ModelPoints(_x, out _, out _);
        System.Array.Copy(_model, outPoints, Points);
    }

    /// <summary>Vi tri 9 khop voi co tay (p, r) va goc ngon a[8] bat ky -- de chay thu.</summary>
    public void ComputePoints(Vector3 p, Quaternion r, float[] a, Vector3[] outPoints)
    {
        _baseP = p; _baseR = r;
        for (int k = 0; k < 6; k++) _x[k] = 0f;
        System.Array.Copy(a, 0, _x, 6, 8);
        ModelPoints(_x, out _, out _);
        System.Array.Copy(_model, outPoints, Points);
    }

    private Quaternion FacingCamera()
    {
        // Khong co goi y nao: ngon tro chia len tren anh, long ban tay quay ve camera
        return Quaternion.LookRotation(_camRot * Vector3.up, -(_camRot * Vector3.forward));
    }

    /// <summary>Dat tay ao theo 2 tia diem 0 va 5 + 1 dang ngon mau + 1 do nghieng mau, roi giai.</summary>
    private float TrySeed(Quaternion rotGuess, float depth, int fingerSeed, int tiltSeed, out Vector3 p, out Quaternion r)
    {
        Vector3 camFwd = _camRot * Vector3.forward;
        // Xoay trong mat phang anh cho huong co tay -> goc ngon tro trung voi anh
        Vector3 imgDir = _camRot * new Vector3(_target[5].x - _target[0].x, _target[5].y - _target[0].y, 0f);
        Vector3 palm = Vector3.ProjectOnPlane(rotGuess * _indexRootL, camFwd);
        Quaternion rot = rotGuess;
        if (palm.sqrMagnitude > 1e-8f && imgDir.sqrMagnitude > 1e-10f)
            rot = Quaternion.FromToRotation(palm, imgDir) * rot;
        // Huong mau: nghieng quanh truc ngang (vuong goc huong long ban tay tren anh), lat quanh
        // truc co tay -> goc ngon tro
        Vector3 across = Vector3.Cross(camFwd, imgDir);
        Vector2 seed = TiltSeeds[tiltSeed];
        if (across.sqrMagnitude > 1e-10f && seed.x != 0f)
            rot = Quaternion.AngleAxis(seed.x, across.normalized) * rot;
        Vector3 palmAxis = rot * _indexRootL;
        if (palmAxis.sqrMagnitude > 1e-10f && seed.y != 0f)
            rot = Quaternion.AngleAxis(seed.y, palmAxis.normalized) * rot;
        // Goc ngon tro nam tren tia diem 5 o chieu sau goi y
        Vector3 x5 = _camPos + _camRot * (new Vector3(_target[5].x, _target[5].y, 1f) * depth);
        Vector3 p0 = x5 - rot * _indexRootL;
        for (int k = 0; k < 8; k++) _a8[k] = FingerSeeds[fingerSeed, k];
        return RefineFrom(p0, rot, _a8, out p, out r, _a8);
    }

    private float RefineFrom(Vector3 p0, Quaternion r0, float[] a0, out Vector3 p, out Quaternion r, float[] aOut)
    {
        _baseP = p0;
        _baseR = r0;
        for (int k = 0; k < 6; k++) _x[k] = 0f;
        System.Array.Copy(a0, 0, _x, 6, 8);
        ClampAngles(_x);
        float cost = Refine(_x);
        Pose(_x, out p, out r);
        System.Array.Copy(_x, 6, aOut, 0, 8);
        return cost;
    }

    private void Pose(float[] x, out Vector3 p, out Quaternion r)
    {
        p = _baseP + new Vector3(x[0], x[1], x[2]) * 0.01f;
        Vector3 rv = new Vector3(x[3], x[4], x[5]);
        float ang = rv.magnitude;
        r = ang > 1e-6f ? Quaternion.AngleAxis(ang, rv / ang) * _baseR : _baseR;
    }

    private void ModelPoints(float[] x, out Vector3 p, out Quaternion r)
    {
        Pose(x, out p, out r);
        _model[0] = p + r * (WristPointLocal * Scale);
        for (int k = 0; k < 4; k++) _a4[k] = x[6 + k];
        _thumb.JointsAt(_a4, p + r * _thumbRootL, r * _thumbParentL, _chain);
        for (int k = 0; k < 4; k++) _model[1 + k] = _chain[k];
        for (int k = 0; k < 4; k++) _a4[k] = x[10 + k];
        _index.JointsAt(_a4, p + r * _indexRootL, r * _indexParentL, _chain);
        for (int k = 0; k < 4; k++) _model[5 + k] = _chain[k];
        _indexMcpJoint = _chain[0];
        _model[5] += r * (IndexMcpPointLocal * Scale);
    }

    private static Vector3 RotVec(Quaternion q)
    {
        if (q.w < 0f) { q.x = -q.x; q.y = -q.y; q.z = -q.z; q.w = -q.w; }
        Vector3 v = new Vector3(q.x, q.y, q.z);
        float s = v.magnitude;
        if (s < 1e-7f) return Vector3.zero;
        float ang = 2f * Mathf.Atan2(s, q.w) * Mathf.Rad2Deg;
        return v * (ang / s);
    }

    private float Evaluate(float[] x, float[] r)
    {
        ModelPoints(x, out Vector3 p, out Quaternion rot);
        int k = 0;
        for (int i = 0; i < Points; i++)
        {
            Vector3 c = _camInv * (_model[i] - _camPos);
            float z = Mathf.Max(c.z, 0.02f);
            Vector2 e = (new Vector2(c.x / z, c.y / z) - _target[i]) * (_imgScale * _weight[i]);
            r[k++] = e.x;
            r[k++] = e.y;
        }

        float zi = (_camInv * (_model[5] - _camPos)).z;
        r[k++] = _useDepthPrior ? DepthPriorWeight * (zi - _depthPrior) / PalmLength : 0f;

        Vector3 v = _useRotPrior ? RotVec(rot * Quaternion.Inverse(_rotPrior)) * RotPriorWeight : Vector3.zero;
        r[k++] = v.x; r[k++] = v.y; r[k++] = v.z;

        if (_useTemporal)
        {
            v = (p - _prevP) * (TemporalPosWeight / PalmLength);
            r[k++] = v.x; r[k++] = v.y; r[k++] = v.z;
            v = RotVec(rot * Quaternion.Inverse(_prevR)) * TemporalRotWeight;
            r[k++] = v.x; r[k++] = v.y; r[k++] = v.z;
            for (int a = 0; a < 8; a++) r[k++] = TemporalAngleWeight * (x[6 + a] - _prevA[a]);
        }
        else
        {
            for (int a = 0; a < 14; a++) r[k++] = 0f;
        }

        // Khong be nguoc khop (ngon cai, ngon tro)
        for (int f = 0; f < 2; f++)
            for (int a = 1; a < 4; a++) r[k++] = HyperextensionWeight * Mathf.Min(0f, x[6 + 4 * f + a]);
        // Dang tay hoc tu tay tran (chi quyet dinh khi anh khong phan biet duoc)
        for (int i = 0; i < 8; i++) _poseDev[i] = x[6 + i] - Mathf.Lerp(PoseMeanOpen[i], PoseMeanPinch[i], Pinch);
        for (int i = 0; i < 8; i++)
        {
            float w = 0f;
            for (int c = 0; c <= i; c++) w += PoseWhiten[i, c] * _poseDev[c];
            r[k++] = PosePriorWeight * w;
        }
        // Cong an cua ngon tro (khop giua, khop dau)
        float hw = HiddenBendWeight * StraightOnImage;
        r[k++] = hw > 0f ? hw * HiddenBend(_indexMcpJoint, _model[6], _model[7]) : 0f;
        r[k++] = hw > 0f ? hw * HiddenBend(_model[6], _model[7], _model[8]) : 0f;

        // Cum luc giac tren mu ban tay: thay -> tam cum ao trung tam cum tren anh (va mat cum khong
        // quay han ra sau); khong thay -> mat cum khong duoc quay ve camera
        r[k] = r[k + 1] = r[k + 2] = 0f;
        if (_dorsalState >= 0)
        {
            Vector3 dp = p + rot * _dorsalL;
            float facing = Vector3.Dot(rot * _dorsalNL, (_camPos - dp).normalized);
            if (_dorsalState == 1)
            {
                Vector3 c = _camInv * (dp - _camPos);
                float z = Mathf.Max(c.z, 0.02f);
                Vector2 e = (new Vector2(c.x / z, c.y / z) - _dorsalTarget) * (_imgScale * DorsalWeight);
                r[k] = e.x; r[k + 1] = e.y;
                r[k + 2] = DorsalHiddenWeight * Mathf.Max(0f, -facing);
            }
            else r[k + 2] = DorsalHiddenWeight * Mathf.Max(0f, facing - DorsalHiddenFacing);
        }
        k += 3;

        // Pinch: anh thay 2 dau ngon chum -> 2 dau ngon ao phai cham nhau (anh 2D khong cho biet
        // chieu sau tung ngon, dieu kien nay chu yeu sua chieu sau)
        Vector3 gap = _model[8] - _model[4];
        float d = gap.magnitude;
        v = Pinch > 0f && d > PinchContactDistance
            ? gap * ((1f - PinchContactDistance / d) * PinchWeight * Pinch / PalmLength) : Vector3.zero;
        r[k++] = v.x; r[k++] = v.y; r[k++] = v.z;

        float sum = 0f;
        for (int i = 0; i < NR; i++) sum += r[i] * r[i];
        return sum;
    }

    private float StraightGate(float palmOnImage)
    {
        if (_weight[6] <= 0f || _weight[7] <= 0f || _weight[8] <= 0f || palmOnImage < 1e-4f) return 0f;
        Vector2 u = _target[7] - _target[6], w = _target[8] - _target[7];
        float bend = Mathf.Abs(Mathf.Atan2(u.x * w.y - u.y * w.x, Vector2.Dot(u, w))) * Mathf.Rad2Deg;
        float len = (u.magnitude + w.magnitude) / palmOnImage;
        return Mathf.Clamp01((StraightBendZero - bend) / (StraightBendZero - StraightBendFull)) *
               Mathf.Clamp01((len - StraightLenZero) / (StraightLenFull - StraightLenZero));
    }

    /// <summary>Goc gap 3D tai b (do) tru goc gap cua hinh chieu len anh -- phan gap camera khong thay.</summary>
    private float HiddenBend(Vector3 a, Vector3 b, Vector3 c)
    {
        Vector3 u = b - a, w = c - b;
        float bend3 = Mathf.Atan2(Vector3.Cross(u, w).magnitude, Vector3.Dot(u, w)) * Mathf.Rad2Deg;
        Vector2 pa = Project(a), pb = Project(b), pc = Project(c);
        Vector2 u2 = pb - pa, w2 = pc - pb;
        float bend2 = Mathf.Abs(Mathf.Atan2(u2.x * w2.y - u2.y * w2.x, Vector2.Dot(u2, w2))) * Mathf.Rad2Deg;
        return Mathf.Max(0f, bend3 - bend2);
    }

    private Vector2 Project(Vector3 world)
    {
        Vector3 c = _camInv * (world - _camPos);
        float z = Mathf.Max(c.z, 0.02f);
        return new Vector2(c.x / z, c.y / z);
    }

    private void ClampAngles(float[] x)
    {
        for (int a = 0; a < 8; a++) x[6 + a] = Mathf.Clamp(x[6 + a], _min[a], _max[a]);
    }

    // Levenberg-Marquardt, dao ham bang sai phan (giong FingerChainFitter, 14 an)
    private float Refine(float[] x)
    {
        float cost = Evaluate(x, _r);
        float lambda = 1e-2f;
        for (int iter = 0; iter < 15; iter++)
        {
            for (int q = 0; q < NP; q++)
            {
                System.Array.Copy(x, _trial, NP);
                _trial[q] += Step[q];
                Evaluate(_trial, _rTry);
                float inv = 1f / Step[q];
                for (int i = 0; i < NR; i++) _jac[i, q] = (_rTry[i] - _r[i]) * inv;
            }
            for (int a = 0; a < NP; a++)
            {
                for (int b = a; b < NP; b++)
                {
                    float s = 0f;
                    for (int i = 0; i < NR; i++) s += _jac[i, a] * _jac[i, b];
                    _normal[a, b] = s;
                    _normal[b, a] = s;
                }
                float g = 0f;
                for (int i = 0; i < NR; i++) g += _jac[i, a] * _r[i];
                _normal[a, NP] = -g;
            }

            bool improved = false;
            for (int attempt = 0; attempt < 4 && !improved; attempt++)
            {
                System.Array.Copy(_normal, _sys, _normal.Length);
                for (int a = 0; a < NP; a++) _sys[a, a] += lambda * (_normal[a, a] + 1e-6f);
                if (!SolveLinear(_sys)) { lambda *= 10f; continue; }
                for (int a = 0; a < NP; a++) _trial[a] = x[a] + Mathf.Clamp(_sys[a, NP], -MaxStep[a], MaxStep[a]);
                ClampAngles(_trial);
                float trialCost = Evaluate(_trial, _rTry);
                if (trialCost < cost)
                {
                    bool converged = cost - trialCost < 1e-7f;
                    System.Array.Copy(_trial, x, NP);
                    System.Array.Copy(_rTry, _r, NR);
                    cost = trialCost;
                    lambda = Mathf.Max(lambda * 0.3f, 1e-4f);
                    improved = true;
                    if (converged) return cost;
                }
                else lambda *= 10f;
            }
            if (!improved) break;
        }
        return cost;
    }

    private static bool SolveLinear(float[,] m)
    {
        int n = NP;
        for (int c = 0; c < n; c++)
        {
            int pivot = c;
            for (int r = c + 1; r < n; r++) if (Mathf.Abs(m[r, c]) > Mathf.Abs(m[pivot, c])) pivot = r;
            if (Mathf.Abs(m[pivot, c]) < 1e-12f) return false;
            if (pivot != c)
                for (int k = c; k <= n; k++) (m[c, k], m[pivot, k]) = (m[pivot, k], m[c, k]);
            for (int r = 0; r < n; r++)
            {
                if (r == c) continue;
                float f = m[r, c] / m[c, c];
                if (f == 0f) continue;
                for (int k = c; k <= n; k++) m[r, k] -= f * m[c, k];
            }
        }
        for (int r = 0; r < n; r++) m[r, n] /= m[r, r];
        return true;
    }
}
