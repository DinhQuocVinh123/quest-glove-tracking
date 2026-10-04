using Oculus.Interaction;
using UnityEngine;

/// <summary>
/// Giu ngon tay AO khong xuyen vao trong vat bop duoc (SquishyPinchable).
///
/// Tay that khep lai duoc vi ngoai doi khong co vat that chan -- nhung tay
/// ao se dung lai dung tren be mat (ke ca khi be mat dang lun). Moi khung
/// hinh, SAU KHI tay ao da duoc dat dang (boi HandVisual / FingerUDPReceiver),
/// script kiem tra tung ngon: neu ngon lot vao trong vat thi CHOT dang ngon
/// (dang ngay truoc khi cham, duoi thang them) va chi xoay CA NGON quanh khop
/// goc cho toi khi nam vua tren be mat -- nhu ngon that bi vat chan lai. Uon
/// tung khop (cach cu) lam ngon gap/be nguoc cong venh khi bop tay that sau.
///
/// Chi sua HINH ANH tay ao. Du lieu tay (vi tri ngon that) giu nguyen, nen vat
/// van biet ban dang bop chat co nao -> van lun dung muc.
///
/// Gan vao object chua HandVisual cua ban tay (vd OVRHandVisualLeft, hoac
/// StaticHandModel_Right cho tay gang). Xuong duoc tim theo ten XRHand_*.
/// </summary>
[DefaultExecutionOrder(200)] // sau FingerUDPReceiver (0) va SquishyPinchable (100)
public class FingertipSurfaceConstraint : MonoBehaviour
{
    [Tooltip("HandVisual cua ban tay nay. Dung de chay NGAY SAU khi no ghi xong dang tay (tranh bi ghi de). De trong = tu tim tren cung object.")]
    [SerializeField] private HandVisual _handVisual;
    [Tooltip("Nhung vat ma ban tay nay KHONG DUOC xuyen vao. De trong = moi vat bop duoc trong scene.\n\n" +
             "Tay gan theo dau (tay gang) duoc do 'theo goc nhin', tay tran do bang vi tri 3D that -- vat tu phan biet, khong can chia danh sach theo tay.")]
    [SerializeField] private SquishyPinchable[] _objects;
    [Tooltip("Chi xu ly ngon cai + ngon tro (2 ngon dung de bop). Tat = xu ly ca 5 ngon.")]
    [SerializeField] private bool _thumbAndIndexOnly = false;
    [Tooltip("Transform dau ngon cai/tro ma vat DOC de biet tay nay dang cam (vd LeftThumbTip cua IsdkFingertipProxy). " +
             "De trong = chinh XRHand_ThumbTip/IndexTip cua ban tay nay (tay gang). Khi dang cam, 2 dau ngon nay duoc dat nam tren be mat vat.")]
    [SerializeField] private Transform _dataThumbTip;
    [SerializeField] private Transform _dataIndexTip;
    [Tooltip("Vat CUNG (PhysicsPinchGrabbable): kiem tra ca TUNG DOT ngon (khong chi dau ngon) -- ngon ao khong cam vao vat o bat ky dau. " +
             "Ban kinh = do day ngon tay (met).")]
    [SerializeField] private float _fingerRadius = 0.008f;
    [Tooltip("Thit ngon dai qua diem XRHand_*Tip bao nhieu (met, doc truc ngon). Mo hinh tay gang: mesh vuot ~9-10 mm qua " +
             "diem Tip, ban kinh dau ngon ~6-9 mm -> truc ngon keo them ~4 mm. 0 = chi xet toi diem Tip (dau ngon ao lut vao vat).")]
    [SerializeField] private float _tipExtension = 0.004f;
    [Tooltip("So diem kiem tra tren moi dot ngon (ke ca diem cuoi dot).")]
    [Range(1, 10)]
    [SerializeField] private int _samplesPerBone = 6;
    [Tooltip("So vong lap xoay khop goc de day ngon ra khoi vat.")]
    [Range(1, 20)]
    [SerializeField] private int _wholeFingerPasses = 10;

    // Moi ngon: cac dot xuong xoay duoc (tu goc ra ngoai) + diem dau ngon.
    private static readonly string[][] ChainNames =
    {
        new[] { "XRHand_ThumbMetacarpal", "XRHand_ThumbProximal", "XRHand_ThumbDistal", "XRHand_ThumbTip" },
        new[] { "XRHand_IndexProximal", "XRHand_IndexIntermediate", "XRHand_IndexDistal", "XRHand_IndexTip" },
        new[] { "XRHand_MiddleProximal", "XRHand_MiddleIntermediate", "XRHand_MiddleDistal", "XRHand_MiddleTip" },
        new[] { "XRHand_RingProximal", "XRHand_RingIntermediate", "XRHand_RingDistal", "XRHand_RingTip" },
        new[] { "XRHand_LittleProximal", "XRHand_LittleIntermediate", "XRHand_LittleDistal", "XRHand_LittleTip" },
    };

    private Transform[][] _chains;

    /// <summary>So ngon vua bi day ra khoi vat trong khung hinh gan nhat.</summary>
    public int ConstrainedFingerCount { get; private set; }

    private void Awake()
    {
        if (_handVisual == null) _handVisual = GetComponent<HandVisual>();
    }

    private void OnEnable()
    {
        if (_handVisual != null) _handVisual.WhenHandVisualUpdated += Apply;
    }

    private void OnDisable()
    {
        if (_handVisual != null) _handVisual.WhenHandVisualUpdated -= Apply;
    }

    // Luon chay them o LateUpdate, phong khi event cua HandVisual khong ban
    // ra (cung ly do voi FingerUDPReceiver). Chay 2 lan cung khong sao: lan
    // sau thay da ngon da nam tren be mat roi thi khong lam gi.
    private void LateUpdate() => Apply();

    [Header("Dang ngon khi cham vat (khong cong venh)")]
    [Tooltip("Luc vua cham vat, dang ngon (goc cac khop giua/dau) duoc GIU NGUYEN nhu ngay truoc khi cham, roi duoi thang them " +
             "theo ti le nay (0 = giu y dang cu, 1 = thang han). Sau do chi XOAY CA NGON quanh khop goc de nam tren be mat -- " +
             "bop tay that chat hon cung khong lam ngon ao gap/be nguoc.")]
    [Range(0f, 1f)]
    [SerializeField] private float _straightenOnContact = 0.6f;
    [Tooltip("Khop goc duoc xoay lech toi da bao nhieu do so voi tay that de day ngon ra khoi vat.")]
    [SerializeField] private float _maxRootCorrectionDeg = 70f;
    [Tooltip("Roi vat: ngon ao chuyen ve theo tay that trong khoang nay (giay).")]
    [SerializeField] private float _releaseBlendSeconds = 0.1f;
    [Tooltip("Het cham van coi la DANG CHAM them khoang nay (giay). Ngon that rung quanh mat vat -> truoc day cham/roi doi qua lai " +
             "moi vai khung, moi lan cham ngon nhay sang dang da chot (duoi thang 60%) -> ngon cai giat khi sap kep.")]
    [SerializeField] private float _contactHoldSeconds = 0.12f;
    [Tooltip("Dang cham: dang ngon ao truot toi dang da sua trong khoang nay (giay) thay vi nhay ngay trong 1 khung.")]
    [SerializeField] private float _contactSmoothSeconds = 0.05f;

    private const int Bones = 3; // moi ngon 3 dot xoay duoc (+ diem dau ngon)

    private sealed class FingerState
    {
        public readonly Quaternion[] Free = new Quaternion[Bones];   // dang tay that gan nhat KHI CHUA CHAM
        public readonly Quaternion[] Frozen = new Quaternion[Bones]; // dang giu trong luc cham
        public readonly Quaternion[] Out = new Quaternion[Bones];    // dang vua ghi ra
        public bool HasFree, HasOut, Contact, Blending;
        // Vat cung dang cham + mat da chot: phan ngon lot vao vat luon day ra MAT NAY (khong phai
        // mat gan nhat -- gan canh hop mat gan nhat doi qua lai lam ngon nhay).
        public PhysicsPinchGrabbable FaceBody;
        public int Face = -1;
        public bool Held; // khung truoc ngon nay dang CAM vat
        public readonly Quaternion[] Prev = new Quaternion[Bones];   // dang ghi ra LAN TRUOC (truoc khi xu ly lan nay)
        public bool HasPrev;
        public float LastContactTime = -999f;
    }

    private FingerState[] _state;

    // Vi tri dau ngon theo TAY THAT (ngay truoc khi bi sua) -- cho vat doc khi quyet dinh cam/tha.
    private static readonly System.Collections.Generic.Dictionary<Transform, Vector3> s_trackedTip =
        new System.Collections.Generic.Dictionary<Transform, Vector3>();

    /// <summary>Vi tri dau ngon theo tay that lan xu ly gan nhat. Dau ngon khong bi script nay
    /// sua (vd diem du lieu tay trai) thi tra ve vi tri hien tai.</summary>
    public static Vector3 TrackedPosition(Transform tip)
    {
        if (tip == null) return Vector3.zero;
        return s_trackedTip.TryGetValue(tip, out Vector3 p) ? p : tip.position;
    }
    private readonly Quaternion[] _scratch = new Quaternion[Bones];

    /// <summary>Tong thoi gian (ms) moi FingertipSurfaceConstraint chay trong KHUNG TRUOC -- ghi vao glove_diag.</summary>
    public static float FrameMs { get; private set; }
    private static int s_msFrame = -1;
    private static float s_msAcc;

    public void Apply()
    {
        long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
        ApplyCore();
        float ms = (System.Diagnostics.Stopwatch.GetTimestamp() - t0) * 1000f / System.Diagnostics.Stopwatch.Frequency;
        if (Time.frameCount != s_msFrame) { FrameMs = s_msAcc; s_msAcc = 0f; s_msFrame = Time.frameCount; }
        s_msAcc += ms;
    }

    private void ApplyCore()
    {
        ConstrainedFingerCount = 0;
        System.Collections.Generic.IReadOnlyList<SquishyPinchable> objects =
            _objects != null && _objects.Length > 0 ? _objects : SquishyPinchable.Active;
        if (objects.Count == 0 && PhysicsPinchGrabbable.Active.Count == 0) return;
        EnsureChains();
        if (_state == null || _state.Length != _chains.Length)
        {
            _state = new FingerState[_chains.Length];
            for (int i = 0; i < _state.Length; i++) _state[i] = new FingerState();
        }

        int count = _thumbAndIndexOnly ? 2 : _chains.Length;
        bool pinchFingersPosed = false;
        for (int f = 0; f < count; f++)
        {
            Transform[] chain = _chains[f];
            if (chain == null) continue;
            FingerState st = _state[f];

            // Ham nay chay 2 lan moi khung (event HandVisual + LateUpdate). Neu dang ngon van
            // la dang minh vua ghi (chua ai dat lai) thi da xu ly roi -- bo qua.
            if (st.HasOut && SameAsOut(chain, st)) continue;
            if (f < 2) pinchFingersPosed = true;
            st.HasPrev = st.HasOut;
            if (st.HasOut) System.Array.Copy(st.Out, st.Prev, Bones);

            Transform tip = chain[Bones];
            if (f < 2) s_trackedTip[tip] = tip.position; // dang ngon luc nay = tay that vua ghi
            // Ngon cai / tro cua tay DANG CAM: luon dat tren be mat (xuyen vao hay ho ra deu sua).
            Transform dataTip = f == 0 ? (_dataThumbTip != null ? _dataThumbTip : tip)
                              : f == 1 ? (_dataIndexTip != null ? _dataIndexTip : tip) : null;

            // 1. Tay THAT (dang vua duoc ghi) co cham vat nao khong?
            bool held = HeldTarget(chain, dataTip, objects, out _, out _);
            bool touching = held || Penetrates(chain, objects);
            float now = Time.time;
            if (touching) st.LastContactTime = now;
            // Tre tha cham: vua roi mat vat thi giu dang cham them _contactHoldSeconds (chong cham/roi doi qua lai)
            bool contact = touching || (st.Contact && now - st.LastContactTime < _contactHoldSeconds);

            if (!contact)
            {
                st.Contact = false;
                st.Held = false;
                st.FaceBody = null;
                st.Face = -1;
                for (int j = 0; j < Bones; j++) st.Free[j] = chain[j].localRotation;
                st.HasFree = true;
                if (st.Blending) BlendBack(chain, st); // vua roi vat: chuyen muot ve tay that
                SaveOut(chain, st);
                continue;
            }

            // 2. Vua cham: chot dang ngon (dang tay that ngay truoc khi cham, duoi thang them)
            if (!st.Contact)
            {
                st.Contact = true;
                FreezeShape(chain, st);
            }

            // 3. Giu dang da chot cho cac khop giua/dau; khop goc bat dau tu tay that roi
            //    xoay ca ngon ra khoi vat.
            //    Dang CAM lien tuc: bat dau tu tu the ngon khung truoc (khong phai tay that) -> ngon
            //    nam yen tren mat vat, chi xoay toi thieu theo diem cham; nhieu tracking ngon that
            //    (ngon cai hay dao dong) khong con lam ngon ao dong mo.
            Quaternion trackedRoot = chain[0].localRotation;
            bool keepPose = held && st.Held && st.HasOut;
            if (keepPose) chain[0].localRotation = st.Out[0];
            for (int j = 1; j < Bones; j++) chain[j].localRotation = st.Frozen[j];
            LockFace(chain, dataTip, st);
            SolveRoot(chain, dataTip, objects, st);
            if (!held) ClampRoot(chain, trackedRoot); // dang cam: diem dich nam tren mat vat, khong can gioi han
            st.Held = held;
            if (st.HasPrev) SmoothToward(chain, st.Prev, _contactSmoothSeconds);

            st.Blending = true;
            ConstrainedFingerCount++;
            SaveOut(chain, st);
        }
        if (pinchFingersPosed) ParallelPinch(objects);
    }

    [Header("Kep bong: 2 dau ngon song song")]
    [Tooltip("Dang kep bong bang ngon cai + ngon tro: xoay dot giua + dot cuoi cua 2 ngon cho 2 DOT CUOI song song nhau, " +
             "bung ngon doi dien qua tam bong (nhu kep vat tron that). 0 = tat (dang ngon theo tay that), 1 = song song han.")]
    [Range(0f, 1f)]
    [SerializeField] private float _parallelPinch = 0.85f;
    [Tooltip("Vua cam: chuyen dan sang dang song song trong khoang nay (giay) -- khong giat.")]
    [SerializeField] private float _parallelBlendSeconds = 0.15f;
    [Tooltip("Moi khop (dot giua, dot cuoi) chi xoay them toi da bay nhieu do de song song -- tranh ngon gap nguoc.")]
    [SerializeField] private float _parallelMaxDeg = 35f;

    private float _parallelWeight;
    private int _parallelFrame = -1;

    /// <summary>Kep bong khung vua roi: 0 = khong xet, 1 = ngon chua cham/cam, 2 = 2 ngon cham nhung khong cung cam 1 bong,
    /// 3 = dang cam (chua chuyen xong), 4 = 2 dot cuoi nguoc nhau, 5 = song song kieu cu, 6 = kep doi xung (3 khop).</summary>
    public int PinchStatus { get; private set; }

    /// <summary>Dang kep CUNG 1 qua bong bang ca ngon cai lan ngon tro: huong chung t = trung binh 2 dot cuoi chieu
    /// len mat phang vuong goc truc kep (noi 2 diem cham) -> xoay moi ngon cho dot cuoi theo t (nua o khop giua,
    /// nua o khop cuoi), roi xoay ca ngon quanh khop goc de bung ngon nam lai tren mat bong.</summary>
    private void ParallelPinch(System.Collections.Generic.IReadOnlyList<SquishyPinchable> objects)
    {
        Transform[] thumb = _chains[0], index = _chains[1];
        PinchStatus = 0;
        if (thumb == null || index == null) return;
        Transform dThumb = _dataThumbTip != null ? _dataThumbTip : thumb[Bones];
        Transform dIndex = _dataIndexTip != null ? _dataIndexTip : index[Bones];
        SquishyPinchable ball = null;
        if (_parallelPinch > 0f && _state[0].Held && _state[1].Held)
            foreach (var obj in objects)
                if (obj != null && obj.isActiveAndEnabled && !obj.MeasuresInViewPlane(dThumb) &&
                    obj.SnapsHeldTip(dThumb) && obj.SnapsHeldTip(dIndex)) { ball = obj; break; }

        if (Time.frameCount != _parallelFrame || !Application.isPlaying) // ngoai Play mode frameCount dung yen (chay thu trong Editor)
        {
            _parallelFrame = Time.frameCount;
            _parallelWeight = Mathf.MoveTowards(_parallelWeight, ball != null ? 1f : 0f,
                Time.deltaTime / Mathf.Max(_parallelBlendSeconds, 1e-3f));
        }
        PinchStatus = ball == null ? (_state[0].Held && _state[1].Held ? 2 : 1) : 3;
        if (ball == null || _parallelWeight <= 0f) return;
        // Truc kep: kep doi xung -> truc on dinh cua bong; khong thi noi 2 diem cham
        Vector3 grip;
        if (!ball.TryGetPinchTarget(dThumb, _fingerRadius, out _, out grip))
        {
            if (!SquishyHeldTarget(thumb, dThumb, ball, out _, out Vector3 cT) ||
                !SquishyHeldTarget(index, dIndex, ball, out _, out Vector3 cI)) return;
            grip = cI - cT;
            if (grip.sqrMagnitude < 1e-8f) return;
            grip.Normalize();
        }
        Vector3 t = Vector3.ProjectOnPlane(DistalDir(thumb), grip) + Vector3.ProjectOnPlane(DistalDir(index), grip);
        if (t.sqrMagnitude < 1e-4f) { PinchStatus = 4; return; } // 2 dot cuoi chi nguoc nhau -- khong co huong chung ro rang
        t.Normalize();
        PinchStatus = 5;

        float w = _parallelPinch * _parallelWeight;
        if (ball.TryGetPinchTarget(dThumb, _fingerRadius, out Vector3 tgT, out _) &&
            ball.TryGetPinchTarget(dIndex, _fingerRadius, out Vector3 tgI, out _))
        {
            // Kep doi xung: moi ngon tu giai 3 khop (gap theo truc gap) cho bung ngon nam DUNG day vet lun cua no,
            // dot cuoi song song voi ngon kia. Truoc day chi xoay khop goc -> bung ngon lech dich toi 19 mm,
            // 2 ngon chi doi nhau 161-164 do (mo phong 04/10).
            PinchStatus = 6;
            if (AdjustPinchShift(ball, thumb, index, tgT, tgI, grip))
            {
                ball.TryGetPinchTarget(dThumb, _fingerRadius, out tgT, out _);
                ball.TryGetPinchTarget(dIndex, _fingerRadius, out tgI, out _);
            }
            Vector3 c = ball.CenterWorld;
            PinchPose(thumb, true, tgT, t, w, c, -grip, _state[0]); // ngon cai o phia -truc kep
            PinchPose(index, false, tgI, t, w, c, grip, _state[1]);
        }
        else
        {
            for (int pass = 0; pass < 2; pass++)
            {
                AlignDistal(thumb, t, w);
                AlignDistal(index, t, w);
                SolveRoot(thumb, dThumb, objects, _state[0]);
                SolveRoot(index, dIndex, objects, _state[1]);
            }
        }
        SaveOut(thumb, _state[0]);
        SaveOut(index, _state[1]);
    }

    /// <summary>Truc gap (the gioi) cua 1 khop: rig XRHand gap 4 ngon quanh +X cuc bo, ngon cai quanh +Y.</summary>
    private static Vector3 FlexAxis(Transform joint, bool thumb) => joint.rotation * (thumb ? Vector3.up : Vector3.right);

    [Header("Bong truot de 2 ngon voi toi thoai mai")]
    [Tooltip("Tam voi thoai mai khi kep, theo phan tam voi khi duoi thang (khop goc -> bung ngon). Ngon nao phai voi xa hon " +
             "thi bong truot doc truc kep (SquishyPinchable.SetPinchShift) cho toi khi 2 ngon deu trong tam -- tay gang khong co " +
             "bong that nen khe 2 ngon that hep hon bong ao, dat bong o giua thi ngon tro phai duoi thang han (dang quap).")]
    [Range(0.6f, 1f)]
    [SerializeField] private float _comfortReach = 0.88f;
    [Tooltip("Bong truot toi da bay nhieu (met) doc truc kep.")]
    [SerializeField] private float _maxPinchShift = 0.02f;
    [Tooltip("Do truot bong bam theo nhu cau cham co nao (giay).")]
    [SerializeField] private float _pinchShiftSeconds = 0.15f;

    private float StraightReach(Transform[] chain) =>
        Vector3.Distance(chain[0].position, chain[1].position) + Vector3.Distance(chain[1].position, chain[2].position) +
        (Vector3.Distance(chain[2].position, chain[3].position) + _tipExtension) * _padAlongDistal;

    /// <summary>Tinh do truot bong doc truc kep cho 2 ngon khong voi qua tam thoai mai. Truot bong delta (met, + = ve ngon tro)
    /// thi tam voi ngon X doi xap xi gX*delta (gX = cos goc giua truc kep va khop goc -> dich); chon delta nho nhat lam het
    /// phan voi qua (binh phuong toi thieu, phat nhe do truot de khong can thi tu ve 0). true = da truot.</summary>
    private bool AdjustPinchShift(SquishyPinchable ball, Transform[] thumb, Transform[] index, Vector3 tgT, Vector3 tgI, Vector3 grip)
    {
        Vector3 vT = tgT - thumb[0].position, vI = tgI - index[0].position;
        float overT = vT.magnitude - _comfortReach * StraightReach(thumb);
        float overI = vI.magnitude - _comfortReach * StraightReach(index);
        float gT = Vector3.Dot(vT.normalized, grip), gI = Vector3.Dot(vI.normalized, grip);
        float shift = ball.PinchShift;
        const float lambda = 0.05f; // phat nhe do truot: ca 2 ngon con du tam thi bong tu ve giua 2 ngon that
        float num = lambda * shift, den = lambda;
        if (overT > 0f) { num += overT * gT; den += gT * gT; }
        if (overI > 0f) { num += overI * gI; den += gI * gI; }
        float want = Mathf.Clamp(shift - num / den, -_maxPinchShift, _maxPinchShift);
        float dt = Application.isPlaying ? Time.deltaTime : 1f / 72f;
        float next = Mathf.Lerp(shift, want, 1f - Mathf.Exp(-dt / Mathf.Max(_pinchShiftSeconds, 1e-3f)));
        if (Mathf.Abs(next - shift) < 1e-6f) return false;
        ball.SetPinchShift(next);
        return true;
    }

    [Header("Dang ngon khi kep bong")]
    [Tooltip("Ngon tro: khop giua phai CAO hon (xa mat bong hon) khop cuoi it nhat khoang nay (met) -- ngon cong vom tu nhien. " +
             "Truoc day buoc 'voi toi' duoi qua khop giua roi khop goc chuc ca ngon xuong -> khop giua vong/quap xuong thap hon khop cuoi.")]
    [SerializeField] private float _pipAboveDip = 0.003f;
    [Tooltip("Ngon tro: goc gap khop giua cho phep khi kep (do) -- khong duoi thang/gap nguoc, khong quap qua.")]
    [SerializeField] private Vector2 _pipFlexRange = new Vector2(15f, 95f);
    [Tooltip("Ngon tro: goc gap khop cuoi cho phep khi kep (do).")]
    [SerializeField] private Vector2 _dipFlexRange = new Vector2(0f, 60f);
    [Tooltip("Dang ngon khi kep truot toi dang moi tinh trong khoang nay (giay) -- ngon nam yen mot cho, khong rung.")]
    [SerializeField] private float _pinchSmoothSeconds = 0.08f;

    /// <summary>Dat 1 ngon dang kep bong. Do tim (2 vong: tho roi min) goc gap khop giua a + khop cuoi b quanh truc gap,
    /// moi cap gia lap luon buoc xoay khop goc dua bung ngon toi dich, roi cham diem:
    ///   voi toi (khoang khop goc -> bung ngon = khoang khop goc -> dich), dot cuoi song song huong chung t,
    ///   ngon tro: khop giua cao hon khop cuoi (_pipAboveDip) + goc gap trong gioi han, va it lech dang dang co.
    /// Chon cap tot nhat, xoay khop goc, roi loc theo thoi gian (_pinchSmoothSeconds).</summary>
    private void PinchPose(Transform[] chain, bool thumb, Vector3 target, Vector3 t, float w, Vector3 center, Vector3 outward, FingerState st)
    {
        Transform root = chain[0], mid = chain[Bones - 2], end = chain[Bones - 1];
        for (int pass = 0; pass < 2; pass++)
        {
            float range = pass == 0 ? 40f : 10f, step = pass == 0 ? 5f : 2f;
            Vector3 r = root.position, m = mid.position, e = end.position, tip = chain[Bones].position;
            Vector3 axM = FlexAxis(mid, thumb), axE = FlexAxis(end, thumb);
            Vector3 prox = (m - r).normalized;
            float wantDist = Vector3.Distance(target, r);
            float bestA = 0f, bestB = 0f, bestCost = float.MaxValue;
            for (float a = -range; a <= range + 1e-3f; a += step)
            {
                Quaternion qa = Quaternion.AngleAxis(a, axM);
                Vector3 e1 = m + qa * (e - m), tip1 = m + qa * (tip - m), axE1 = qa * axE;
                Vector3 interDir = (e1 - m).normalized;
                float pipFlex = Vector3.SignedAngle(prox, interDir, axM);
                for (float b = -range; b <= range + 1e-3f; b += step)
                {
                    Vector3 tip2 = e1 + Quaternion.AngleAxis(b, axE1) * (tip1 - e1);
                    Vector3 d = (tip2 - e1).normalized;
                    Vector3 pad = Vector3.Lerp(e1, tip2 + d * _tipExtension, _padAlongDistal);
                    float reach = (Vector3.Distance(pad, r) - wantDist) * 1000f;               // mm
                    Quaternion qr = Quaternion.FromToRotation(pad - r, target - r);            // buoc xoay khop goc
                    float par = Vector3.Angle(qr * d, t) / 8f;
                    float cost = reach * reach + w * par * par + (a * a + b * b) / (25f * 25f) * 0.5f;
                    if (!thumb)
                    {
                        float hM = Vector3.Dot(qr * (m - r) + r - center, outward);
                        float hE = Vector3.Dot(qr * (e1 - r) + r - center, outward);
                        float low = Mathf.Max(0f, hE + _pipAboveDip - hM) * 1000f;             // mm khop giua thap hon muc
                        float dipFlex = Vector3.SignedAngle(interDir, d, axE1);
                        float lim = Mathf.Max(0f, _pipFlexRange.x - pipFlex) + Mathf.Max(0f, pipFlex - _pipFlexRange.y)
                                  + Mathf.Max(0f, _dipFlexRange.x - dipFlex) + Mathf.Max(0f, dipFlex - _dipFlexRange.y);
                        cost += 4f * low * low + (lim / 3f) * (lim / 3f);
                    }
                    if (cost < bestCost) { bestCost = cost; bestA = a; bestB = b; }
                }
            }
            mid.rotation = Quaternion.AngleAxis(bestA, axM) * mid.rotation;
            end.rotation = Quaternion.AngleAxis(bestB, FlexAxis(end, thumb)) * end.rotation;
            RotateRoot(root, PadPoint(chain), target);
        }
        if (st.HasPrev) SmoothToward(chain, st.Prev, _pinchSmoothSeconds);
    }

    private static Vector3 DistalDir(Transform[] chain) => (chain[Bones].position - chain[Bones - 1].position).normalized;

    [Tooltip("Diem bung ngon dat vao vet lun khi kep bong: vi tri doc dot cuoi, tu khop cuoi (0) toi het thit dau ngon (1).")]
    [Range(0f, 1f)]
    [SerializeField] private float _padAlongDistal = 0.65f;

    /// <summary>Diem tren truc dot cuoi ung voi bung ngon (giua khop cuoi va het thit dau ngon).</summary>
    private Vector3 PadPoint(Transform[] chain)
    {
        Vector3 end = chain[Bones].position + DistalDir(chain) * _tipExtension;
        return Vector3.Lerp(chain[Bones - 1].position, end, _padAlongDistal);
    }

    /// <summary>Xoay dot cuoi cua ngon ve phia huong `t` (ti le w): nua goc o khop giua, phan con lai o khop cuoi,
    /// moi khop toi da _parallelMaxDeg.</summary>
    private void AlignDistal(Transform[] chain, Vector3 t, float w)
    {
        Vector3 want = Vector3.Slerp(DistalDir(chain), t, w);
        for (int step = 0; step < 2; step++)
        {
            Vector3 d = DistalDir(chain);
            Quaternion.FromToRotation(d, want).ToAngleAxis(out float angle, out Vector3 axis);
            if (angle > 180f) angle -= 360f;
            if (Mathf.Abs(angle) < 0.05f || !float.IsFinite(axis.x)) return;
            float part = step == 0 ? angle * 0.5f : angle;
            part = Mathf.Clamp(part, -_parallelMaxDeg, _parallelMaxDeg);
            Transform joint = chain[step == 0 ? Bones - 2 : Bones - 1];
            joint.rotation = Quaternion.AngleAxis(part, axis) * joint.rotation;
        }
    }

    /// <summary>Dang cam: diem `from` tren ngon ao can dat toi diem `to` tren be mat (ke ca khi dang ho ra).
    /// Bong: diem gan be mat nhat cua DOT CUOI (bung hoac dau ngon); hop: dau ngon.</summary>
    private bool HeldTarget(Transform[] chain, Transform dataTip,
        System.Collections.Generic.IReadOnlyList<SquishyPinchable> objects, out Vector3 from, out Vector3 to)
    {
        from = to = chain[Bones].position;
        if (dataTip == null) return false;
        Vector3 tipPos = chain[Bones].position;
        foreach (var obj in objects)
        {
            if (obj == null || !obj.isActiveAndEnabled) continue;
            if (obj.MeasuresInViewPlane(dataTip))
            {
                if (obj.TryGetHeldContact(dataTip, tipPos, out to)) return true;
            }
            else if (SquishyHeldTarget(chain, dataTip, obj, out from, out to)) return true;
        }
        from = tipPos;
        foreach (var body in PhysicsPinchGrabbable.Active)
            if (body != null && body.TryGetHeldContact(dataTip, tipPos, out to)) return true;
        return false;
    }

    /// <summary>Co phan nao cua ngon lot vao vat khong -- xet doc CA NGON (ke ca phan thit qua diem Tip).
    /// Tay do theo goc nhin (gan dau): bong chi xet dau ngon nhu cu.</summary>
    private bool Penetrates(Transform[] chain, System.Collections.Generic.IReadOnlyList<SquishyPinchable> objects)
    {
        foreach (var obj in objects)
        {
            if (obj == null || !obj.isActiveAndEnabled) continue;
            if (obj.MeasuresInViewPlane(chain[Bones]) ? obj.TryResolvePenetration(chain[Bones], out _)
                                                      : DeepestSquishy(chain, obj, out _, out _) > 0f) return true;
        }
        foreach (var body in PhysicsPinchGrabbable.Active)
            if (body != null && DeepestPoint(chain, body, -1, out _, out _) > 0f) return true;
        return false;
    }

    /// <summary>Cac diem tren TRUC ngon: moi dot _samplesPerBone diem, them 2 diem keo qua dau ngon
    /// (_tipExtension). Diem cua dot cuoi + phan keo dai nam o cuoi mang (tu DistalStart).</summary>
    private int FillSamples(Transform[] chain)
    {
        int n = 0;
        for (int j = 0; j < Bones; j++)
            for (int k = 1; k <= _samplesPerBone; k++)
                _samples[n++] = Vector3.Lerp(chain[j].position, chain[j + 1].position, (float)k / _samplesPerBone);
        if (_tipExtension > 0f)
        {
            Vector3 tip = chain[Bones].position, dir = tip - chain[Bones - 1].position;
            if (dir.sqrMagnitude > 1e-10f)
            {
                dir.Normalize();
                _samples[n++] = tip + dir * (_tipExtension * 0.5f);
                _samples[n++] = tip + dir * _tipExtension;
            }
        }
        return n;
    }

    private int DistalStart => (Bones - 1) * _samplesPerBone;
    private readonly Vector3[] _samples = new Vector3[Bones * 10 + 2];

    /// <summary>Diem lot sau nhat doc ngon vao bong (be mat da lun). Tra ve do sau (met), 0 = khong lot.</summary>
    private float DeepestSquishy(Transform[] chain, SquishyPinchable obj, out Vector3 point, out Vector3 resolved)
    {
        point = resolved = default;
        float deepest = 0f;
        int n = FillSamples(chain);
        for (int i = 0; i < n; i++)
        {
            float gap = obj.SkinGap(_samples[i], _fingerRadius, out Vector3 s);
            if (-gap > deepest) { deepest = -gap; point = _samples[i]; resolved = s; }
        }
        return deepest;
    }

    /// <summary>Dang cam bong: diem cua DOT CUOI (bung ngon / dau ngon) gan be mat nhat -> dat len be mat,
    /// neu khong ho qua SnapMaxGap. Truoc day chi dat diem Tip -> bung ngon va phan thit qua Tip lun vao bong.</summary>
    private bool SquishyHeldTarget(Transform[] chain, Transform dataTip, SquishyPinchable obj, out Vector3 from, out Vector3 to)
    {
        from = to = chain[Bones].position;
        if (!obj.SnapsHeldTip(dataTip)) return false;
        // Kep doi xung: DIEM BUNG NGON CO DINH dat vao day vet lun cua ngon nay (2 ngon doi nhau qua tam bong).
        // Diem co dinh -> khong nhay giua cac diem nhu khi chon "diem gan be mat nhat" moi khung.
        if (obj.TryGetPinchTarget(dataTip, _fingerRadius, out to, out _))
        {
            from = PadPoint(chain);
            return true;
        }
        int n = FillSamples(chain);
        float best = float.PositiveInfinity;
        for (int i = DistalStart; i < n; i++)
        {
            float gap = obj.SkinGap(_samples[i], _fingerRadius, out Vector3 s);
            if (gap < best) { best = gap; from = _samples[i]; to = s; }
        }
        return best <= obj.SnapMaxGap;
    }

    /// <summary>Chot mat vat cung cho ngon nay: dang cam -> mat vat da chot luc cam; vua cham ->
    /// mat gan diem lot sau nhat, giu nguyen cho toi khi roi vat.</summary>
    private void LockFace(Transform[] chain, Transform dataTip, FingerState st)
    {
        foreach (var body in PhysicsPinchGrabbable.Active)
        {
            if (body == null) continue;
            int held = body.HeldFace(dataTip);
            if (held >= 0) { st.FaceBody = body; st.Face = held; return; }
        }
        if (st.FaceBody != null && st.FaceBody.isActiveAndEnabled) return;
        st.FaceBody = null;
        st.Face = -1;
        foreach (var body in PhysicsPinchGrabbable.Active)
            if (body != null && DeepestPoint(chain, body, -1, out Vector3 q, out _) > 0f)
            {
                st.FaceBody = body;
                st.Face = body.FaceOf(q);
                return;
            }
    }

    /// <summary>Diem lot sau nhat doc ngon vao vat cung. Tra ve do sau (met), 0 = khong lot.</summary>
    private float DeepestPoint(Transform[] chain, PhysicsPinchGrabbable body, int face, out Vector3 point, out Vector3 resolved)
    {
        point = resolved = default;
        float deepest = 0f;
        int n = FillSamples(chain);
        for (int i = 0; i < n; i++)
        {
            Vector3 q = _samples[i];
            if (!body.ResolvePoint(q, _fingerRadius, face, out Vector3 r)) continue;
            float d = (r - q).sqrMagnitude;
            if (d > deepest) { deepest = d; point = q; resolved = r; }
        }
        return Mathf.Sqrt(deepest);
    }

    /// <summary>Xoay CA NGON quanh khop goc (dang ngon giu nguyen) cho toi khi dau ngon nam tren
    /// be mat (dang cam) va khong phan nao con lot vao vat.</summary>
    private void SolveRoot(Transform[] chain, Transform dataTip,
        System.Collections.Generic.IReadOnlyList<SquishyPinchable> objects, FingerState st)
    {
        Transform root = chain[0];
        for (int iter = 0; iter < _wholeFingerPasses; iter++)
        {
            bool moved = false;
            if (HeldTarget(chain, dataTip, objects, out Vector3 from, out Vector3 held))
                moved |= RotateRoot(root, from, held);
            foreach (var obj in objects)
            {
                if (obj == null || !obj.isActiveAndEnabled) continue;
                if (obj.MeasuresInViewPlane(chain[Bones]))
                {
                    if (obj.TryResolvePenetration(chain[Bones], out Vector3 t)) moved |= RotateRoot(root, chain[Bones].position, t);
                }
                else if (DeepestSquishy(chain, obj, out Vector3 q, out Vector3 r) > 0f)
                    moved |= RotateRoot(root, q, r);
            }
            foreach (var body in PhysicsPinchGrabbable.Active)
                if (body != null && DeepestPoint(chain, body, body == st.FaceBody ? st.Face : -1, out Vector3 q, out Vector3 r) > 0f)
                    moved |= RotateRoot(root, q, r);
            if (!moved) break;
        }
    }

    private static bool RotateRoot(Transform root, Vector3 from, Vector3 to)
    {
        Vector3 a = from - root.position, b = to - root.position;
        if (a.sqrMagnitude < 1e-10f || b.sqrMagnitude < 1e-10f || (to - from).sqrMagnitude < 1e-8f) return false;
        root.rotation = Quaternion.FromToRotation(a, b) * root.rotation;
        return true;
    }

    private void ClampRoot(Transform[] chain, Quaternion trackedRoot)
    {
        float angle = Quaternion.Angle(trackedRoot, chain[0].localRotation);
        if (angle > _maxRootCorrectionDeg)
            chain[0].localRotation = Quaternion.Slerp(trackedRoot, chain[0].localRotation, _maxRootCorrectionDeg / angle);
    }

    /// <summary>Chot dang ngon luc vua cham: dang tay that ngay truoc khi cham (neu co), roi
    /// duoi thang tung khop theo _straightenOnContact (tinh hinh hoc, khong can biet tu the nghi cua rig).</summary>
    private void FreezeShape(Transform[] chain, FingerState st)
    {
        for (int j = 0; j < Bones; j++) _scratch[j] = chain[j].localRotation;
        if (st.HasFree)
            for (int j = 1; j < Bones; j++) chain[j].localRotation = st.Free[j];

        for (int j = 1; j < Bones; j++)
        {
            Vector3 parentDir = chain[j].position - chain[j - 1].position;
            Vector3 dir = chain[j + 1].position - chain[j].position;
            if (parentDir.sqrMagnitude < 1e-10f || dir.sqrMagnitude < 1e-10f) continue;
            Quaternion straighten = Quaternion.Slerp(Quaternion.identity, Quaternion.FromToRotation(dir, parentDir), _straightenOnContact);
            chain[j].rotation = straighten * chain[j].rotation;
        }
        for (int j = 0; j < Bones; j++) st.Frozen[j] = chain[j].localRotation;
        for (int j = 0; j < Bones; j++) chain[j].localRotation = _scratch[j];
    }

    /// <summary>Dang ngon vua tinh -> truot tu dang lan truoc toi no (loc mu, hang so thoi gian `seconds`).</summary>
    private static void SmoothToward(Transform[] chain, Quaternion[] prev, float seconds)
    {
        if (seconds <= 0f) return;
        float dt = Application.isPlaying ? Time.deltaTime : 1f / 72f;
        float k = 1f - Mathf.Exp(-dt / seconds);
        for (int j = 0; j < Bones; j++) chain[j].localRotation = Quaternion.Slerp(prev[j], chain[j].localRotation, k);
    }

    private void BlendBack(Transform[] chain, FingerState st)
    {
        float k = 1f - Mathf.Exp(-Time.deltaTime / Mathf.Max(_releaseBlendSeconds, 1e-3f));
        bool done = true;
        for (int j = 0; j < Bones; j++)
        {
            Quaternion tracked = chain[j].localRotation;
            chain[j].localRotation = Quaternion.Slerp(st.Out[j], tracked, k);
            if (Quaternion.Angle(chain[j].localRotation, tracked) > 0.5f) done = false;
        }
        if (done) st.Blending = false;
    }

    private static void SaveOut(Transform[] chain, FingerState st)
    {
        for (int j = 0; j < Bones; j++) st.Out[j] = chain[j].localRotation;
        st.HasOut = true;
    }

    private static bool SameAsOut(Transform[] chain, FingerState st)
    {
        for (int j = 0; j < Bones; j++)
            if (Quaternion.Angle(chain[j].localRotation, st.Out[j]) > 0.01f) return false;
        return true;
    }

    /// <summary>Tim xuong theo ten. Tim lai neu bo xuong dang dung bi tat --
    /// HandVisual co the doi giua 2 bo xuong (OpenXR / Oculus) luc chay.</summary>
    private void EnsureChains()
    {
        if (_chains != null && _chains[0] != null && _chains[0][0].gameObject.activeInHierarchy) return;

        _chains = new Transform[ChainNames.Length][];
        for (int f = 0; f < ChainNames.Length; f++)
        {
            var chain = new Transform[ChainNames[f].Length];
            bool complete = true;
            for (int j = 0; j < chain.Length; j++)
            {
                chain[j] = FindActiveDeepChild(transform, ChainNames[f][j]);
                if (chain[j] == null) { complete = false; break; }
            }
            _chains[f] = complete ? chain : null;
        }
    }

    private static Transform FindActiveDeepChild(Transform parent, string name)
    {
        foreach (Transform child in parent)
        {
            if (!child.gameObject.activeInHierarchy) continue;
            if (child.name == name) return child;
            Transform result = FindActiveDeepChild(child, name);
            if (result != null) return result;
        }
        return null;
    }
}
