using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Vat CUNG dung physics that cua Unity (Rigidbody): cam bang cach PINCH
/// (dau ngon cai + dau ngon tro kep 2 ben vat), tha ra thi ROI theo trong
/// luc, nay, lan, va NEM duoc (vat giu van toc cua tay luc tha).
///
/// Khac SquishyPinchable (bong mem, tu tinh do lun + luc, khong dung physics):
/// vat nay khong bien dang. Cach cam/tha giong bong de 2 vat cam giac nhu nhau:
///   - cam: CA HAI dau ngon sat be mat (_grabMargin) va o 2 phia doi dien
///   - tha: 2 ngon mo ra them _releaseOpening so voi luc cam, lien tuc
///     _releaseDelay giay (chong tha nham khi dau ngon tay gang nhay)
///   - dang cam: vat gan theo CO TAY (Quest 72 Hz, muot) neu tay co wrist
///
/// Luc cam, Rigidbody chuyen sang kinematic (physics khong keo vat khoi tay)
/// nhung van day duoc vat khac; tha ra thi bat lai physics + gan van toc tay.
/// Roi qua xa (duoi _home 1.5 m, hoac cach _home 4 m) thi tu ve cho cu.
/// </summary>
[RequireComponent(typeof(Rigidbody))]
[DefaultExecutionOrder(100)] // sau FingerUDPReceiver / IsdkFingertipProxy, truoc FingertipSurfaceConstraint (200)
public class PhysicsPinchGrabbable : MonoBehaviour
{
    [System.Serializable]
    public struct Hand
    {
        public Transform thumbTip;
        public Transform indexTip;
        [Tooltip("Tuy chon: co tay -- dang cam thi vat gan theo co tay (muot hon dau ngon tay gang).")]
        public Transform wrist;
    }

    [SerializeField] private Hand[] _hands = new Hand[0];
    [Tooltip("Ban kinh dau ngon (met): diem dau ngon nam trong ngon, da ngon cach diem do chung nay.")]
    [SerializeField] private float _fingerRadius = 0.008f;
    [SerializeField] private float _grabMargin = 0.02f;
    [SerializeField] private float _releaseOpening = 0.015f;
    [SerializeField] private float _releaseDelay = 0.12f;
    [Tooltip("Tay dang cam bi mat dau ngan hon muc nay (giay) thi van giu vat.")]
    [SerializeField] private float _lostGraceSeconds = 0.35f;
    [Tooltip("Cam kieu 'doi dinh' (antipodal): moi dau ngon phai ep vao mat vat voi goc lech so voi phap tuyen mat do " +
             "khong qua muc nay (do) -- tuong ung non ma sat (mu ~ 1 -> 45 do). Cham 2 ngon vao CUNG 1 mat hay 2 dau 1 canh thi khong cam duoc.")]
    [SerializeField] private float _frictionConeDeg = 45f;
    [Tooltip("Dang cam: vat tu truot ve GIUA 2 dau ngon (theo huong kep) voi toc do nay (1/giay).")]
    [SerializeField] private float _recenterSpeed = 3f;
    [Tooltip("Dang cam: diem cham cua moi ngon tren mat hop chi TRUOT CHAM theo tay that voi toc do nay (1/giay), " +
             "va luon o tren CUNG 1 mat da chon luc cam -- khong nhay qua canh sang mat khac.")]
    [SerializeField] private float _contactSlideSpeed = 4f;
    [Tooltip("Tay khong co wrist: vat truot ve giua 2 dau ngon nhanh co nao.")]
    [SerializeField] private float _followSpeed = 20f;
    [Tooltip("Van toc nem = van toc trung binh cua tay trong khoang nay (giay) truoc luc tha.")]
    [SerializeField] private float _throwWindow = 0.1f;
    [SerializeField] private float _throwScale = 1f;

    [Header("Ve cho cu")]
    [Tooltip("Vi tri ban dau / khi dat lai. De trong = vi tri luc bat dau.")]
    [SerializeField] private Transform _home;
    [SerializeField] private float _resetBelow = 1.5f;
    [SerializeField] private float _resetDistance = 4f;

    public static readonly List<PhysicsPinchGrabbable> Active = new List<PhysicsPinchGrabbable>();

    public bool IsHeld => _heldBy >= 0;

    private Rigidbody _rb;
    private Collider _collider;
    private int _heldBy = -1;
    private bool[] _valid = new bool[0];
    private bool[] _regrabBlocked = new bool[0];
    private float _gapAtGrab, _releaseTimer, _lostTimer;
    private Vector3 _localPos;      // vi tri vat trong he co tay (hoac so voi diem giua 2 ngon)
    private Quaternion _localRot;
    private Vector3 _homePos;
    private Quaternion _homeRot;
    private Vector3 _throwLinear, _throwAngular; // van toc luc 2 ngon BAT DAU mo (luc tha that da tre _releaseDelay)
    private float _clock; // thoi gian mo phong rieng (Tick/PhysicsStep) -- chay thu duoc ngoai vong lap game
    // Dang cam (hop): mat ma moi ngon ep vao (0 = ngon cai, 1 = ngon tro), chot luc cam.
    // Ma mat = truc*2 + (1 neu mat phia duong). Diem cham luu trong toa do local cua hop.
    private readonly int[] _heldFace = { -1, -1 };
    private readonly Vector3[] _heldPoint = new Vector3[2];
    private readonly Queue<(float t, Vector3 p, Quaternion r)> _history = new Queue<(float, Vector3, Quaternion)>();

    private void Awake()
    {
        _rb = GetComponent<Rigidbody>();
        _collider = GetComponent<Collider>();
        _homePos = transform.position;
        _homeRot = transform.rotation;
        _rb.interpolation = RigidbodyInterpolation.Interpolate;
        _rb.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
    }

    private void OnEnable() => Active.Add(this);
    private void OnDisable() => Active.Remove(this);

    private void EnsureState()
    {
        if (_valid.Length == _hands.Length) return;
        _valid = new bool[_hands.Length];
        _regrabBlocked = new bool[_hands.Length];
        _heldBy = -1;
    }

    /// <summary>Dua vat ve cho cu, dung yen.</summary>
    public void ResetToHome()
    {
        if (_rb == null) Awake(); // goi truoc Awake (vd tu Editor)
        _heldBy = -1;
        _heldFace[0] = _heldFace[1] = -1;
        _rb.isKinematic = false;
        Vector3 p = _home != null ? _home.position : _homePos;
        Quaternion r = _home != null ? _home.rotation : _homeRot;
        _rb.position = p;
        _rb.rotation = r;
        transform.SetPositionAndRotation(p, r);
        _rb.linearVelocity = Vector3.zero;
        _rb.angularVelocity = Vector3.zero;
    }

    // Cam/tha tinh trong Update (moi khung hinh, cung nhip voi tay). Vat dang cam
    // di chuyen bang MovePosition trong FixedUpdate de van day duoc vat khac.
    private void Update() => Tick(Time.deltaTime);
    private void FixedUpdate() => PhysicsStep(Time.fixedDeltaTime);

    /// <summary>Mot buoc cam/tha. Public de chay thu ngoai Play mode.</summary>
    public void Tick(float dt)
    {
        EnsureState();
        for (int h = 0; h < _hands.Length; h++)
        {
            Transform a = _hands[h].thumbTip, b = _hands[h].indexTip;
            _valid[h] = a != null && b != null && a.gameObject.activeInHierarchy && b.gameObject.activeInHierarchy;
        }

        if (_heldBy >= 0) UpdateHeld(dt);
        if (_heldBy < 0) TryGrab();

        if (_heldBy < 0 && !_rb.isKinematic)
        {
            Vector3 home = _home != null ? _home.position : _homePos;
            if (transform.position.y < home.y - _resetBelow || (transform.position - home).sqrMagnitude > _resetDistance * _resetDistance)
                ResetToHome();
        }
    }

    private void TryGrab()
    {
        for (int h = 0; h < _hands.Length; h++)
        {
            if (!_valid[h]) { _regrabBlocked[h] = false; continue; }
            Vector3 thumb = Tracked(_hands[h].thumbTip), index = Tracked(_hands[h].indexTip);
            bool canGrab = SkinDistance(thumb) < _grabMargin && SkinDistance(index) < _grabMargin && IsAntipodal(thumb, index);
            if (_regrabBlocked[h])
            {
                if (!canGrab) _regrabBlocked[h] = false;
                continue;
            }
            if (!canGrab) continue;

            _heldBy = h;
            _gapAtGrab = Vector3.Distance(thumb, index);
            _releaseTimer = _lostTimer = 0f;
            _rb.isKinematic = true;
            _history.Clear();
            Pose anchor = Anchor(h);
            _localPos = Quaternion.Inverse(anchor.rotation) * (transform.position - anchor.position);
            _localRot = Quaternion.Inverse(anchor.rotation) * transform.rotation;
            // Chot mat cho tung ngon (IsAntipodal da xac nhan 2 mat doi dien)
            _heldFace[0] = FaceOf(thumb);
            _heldFace[1] = FaceOf(index);
            if (_heldFace[0] >= 0) _heldPoint[0] = ProjectOnFace(thumb, _heldFace[0]);
            if (_heldFace[1] >= 0) _heldPoint[1] = ProjectOnFace(index, _heldFace[1]);
            return;
        }
    }

    private void UpdateHeld(float dt)
    {
        int h = _heldBy;
        _lostTimer = _valid[h] ? 0f : _lostTimer + dt;
        bool release = _lostTimer > _lostGraceSeconds;
        if (_valid[h])
        {
            float gap = Vector3.Distance(Tracked(_hands[h].thumbTip), Tracked(_hands[h].indexTip));
            // Moc = khoang cach luc cam, nhung khong nho hon be day vat giua 2 ngon:
            // ngon that bop "xuyen" vao vat cung (khong co gi chan) thi noi tay ra
            // mot chut van con dang cam.
            _gapAtGrab = Mathf.Min(_gapAtGrab, Mathf.Max(gap, ObjectWidthBetweenFingers(h)));
            // Diem cham truot cham theo tay that, nhung khong roi khoi mat da chot
            float slide = 1f - Mathf.Exp(-_contactSlideSpeed * dt);
            SlideContact(0, Tracked(_hands[h].thumbTip), slide);
            SlideContact(1, Tracked(_hands[h].indexTip), slide);
            bool opening = gap > _gapAtGrab + _releaseOpening;
            // Nem: tay thuong da cham lai khi dieu kien tha du _releaseDelay -> lay van toc
            // ngay luc bat dau mo ngon.
            if (opening && _releaseTimer == 0f) MeasureVelocity(out _throwLinear, out _throwAngular);
            _releaseTimer = opening ? _releaseTimer + dt : 0f;
            release |= _releaseTimer >= _releaseDelay;
        }
        if (!release) return;

        _heldBy = -1;
        _heldFace[0] = _heldFace[1] = -1;
        _regrabBlocked[h] = _valid[h];
        _rb.isKinematic = false;
        if (_releaseTimer <= 0f) MeasureVelocity(out _throwLinear, out _throwAngular); // tha vi mat dau tay
        _rb.linearVelocity = _throwLinear * _throwScale;
        _rb.angularVelocity = _throwAngular;
        _releaseTimer = 0f;
    }

    /// <summary>Mot buoc physics: vat dang cam di theo tay. Public de chay thu.</summary>
    public void PhysicsStep(float dt)
    {
        _clock += dt;
        if (_heldBy < 0 || !_valid[_heldBy]) return;
        Pose anchor = Anchor(_heldBy);
        RecenterBetweenFingers(anchor, dt);
        Vector3 pos = anchor.position + anchor.rotation * _localPos;
        Quaternion rot = anchor.rotation * _localRot;
        if (_hands[_heldBy].wrist == null)
        {
            float k = 1f - Mathf.Exp(-_followSpeed * dt);
            pos = Vector3.Lerp(_rb.position, pos, k);
            rot = Quaternion.Slerp(_rb.rotation, rot, k);
        }
        _rb.MovePosition(pos);
        _rb.MoveRotation(rot);

        _history.Enqueue((_clock, pos, rot));
        while (_history.Count > 2 && _clock - _history.Peek().t > _throwWindow) _history.Dequeue();
    }

    /// <summary>Van toc (dai + xoay) trung binh cua vat trong _throwWindow cuoi.</summary>
    private void MeasureVelocity(out Vector3 linear, out Vector3 angular)
    {
        linear = angular = Vector3.zero;
        if (_history.Count < 2) return;
        var first = _history.Peek();
        (float t, Vector3 p, Quaternion r) last = default;
        foreach (var e in _history) last = e;
        float span = last.t - first.t;
        if (span < 1e-3f) return;
        linear = (last.p - first.p) / span;
        (last.r * Quaternion.Inverse(first.r)).ToAngleAxis(out float deg, out Vector3 axis);
        if (deg > 180f) deg -= 360f;
        if (!float.IsNaN(axis.x) && !float.IsInfinity(axis.x)) angular = axis * (deg * Mathf.Deg2Rad / span);
    }

    /// <summary>Kep dung kieu doi dinh: 2 diem cham tren 2 mat DOI DIEN, duong noi 2 dau ngon
    /// nam trong non ma sat cua ca 2 mat (Nguyen 1988, force-closure 2 ngon).</summary>
    private bool IsAntipodal(Vector3 thumb, Vector3 index)
    {
        Vector3 axis = index - thumb;
        if (axis.sqrMagnitude < 1e-8f) return false;
        axis.Normalize();
        float cosCone = Mathf.Cos(_frictionConeDeg * Mathf.Deg2Rad);
        // Phap tuyen HUONG RA tai diem cham: ngon cai phai ep nguoc chieu axis, ngon tro cung chieu
        return Vector3.Dot(OutwardNormal(thumb), -axis) > cosCone &&
               Vector3.Dot(OutwardNormal(index), axis) > cosCone;
    }

    /// <summary>Phap tuyen huong ra cua mat vat gan diem nay nhat.</summary>
    private Vector3 OutwardNormal(Vector3 point)
    {
        int face = FaceOf(point);
        if (face >= 0) return FaceNormal(face);
        Vector3 c = ClosestSurfacePoint(point, out bool inside);
        Vector3 dir = inside ? c - point : point - c;
        return dir.sqrMagnitude > 1e-10f ? dir.normalized : (point - _collider.bounds.center).normalized;
    }

    /// <summary>Mat hop gan diem nay nhat (ma = truc*2 + 1 neu phia duong). -1 = vat khong phai hop.
    /// Chon mat co khoang cach co dau (met) LON NHAT: ngoai hop = mat diem vuot ra xa nhat,
    /// trong hop = mat gan nhat.</summary>
    public int FaceOf(Vector3 world)
    {
        if (!(_collider is BoxCollider box)) return -1;
        Transform t = box.transform;
        Vector3 half = box.size * 0.5f, sc = t.lossyScale;
        Vector3 p = t.InverseTransformPoint(world) - box.center;
        int best = 0; float bestD = float.MinValue;
        for (int a = 0; a < 3; a++)
        {
            float d = (Mathf.Abs(p[a]) - half[a]) * Mathf.Abs(sc[a]);
            if (d > bestD) { bestD = d; best = a; }
        }
        return best * 2 + (p[best] >= 0f ? 1 : 0);
    }

    /// <summary>Phap tuyen huong ra (the gioi) cua mat hop.</summary>
    private Vector3 FaceNormal(int face)
    {
        Vector3 n = Vector3.zero;
        n[face / 2] = (face & 1) == 1 ? 1f : -1f;
        return _collider.transform.TransformDirection(n).normalized;
    }

    /// <summary>Chieu diem len mat hop (toa do local cua hop), lui vao trong mep mot chut de
    /// bung ngon nam han tren mat chu khong vat qua canh.</summary>
    private Vector3 ProjectOnFace(Vector3 world, int face)
    {
        var box = (BoxCollider)_collider;
        Transform t = box.transform;
        Vector3 half = box.size * 0.5f, sc = t.lossyScale;
        Vector3 p = t.InverseTransformPoint(world) - box.center;
        int axis = face / 2;
        for (int a = 0; a < 3; a++)
        {
            if (a == axis) continue;
            float inset = Mathf.Min(_fingerRadius * 0.75f / Mathf.Max(Mathf.Abs(sc[a]), 1e-6f), half[a]);
            p[a] = Mathf.Clamp(p[a], -half[a] + inset, half[a] - inset);
        }
        p[axis] = (face & 1) == 1 ? half[axis] : -half[axis];
        return p + box.center;
    }

    private void SlideContact(int finger, Vector3 tip, float k)
    {
        int face = _heldFace[finger];
        if (face < 0) return;
        _heldPoint[finger] = Vector3.Lerp(_heldPoint[finger], ProjectOnFace(tip, face), k);
    }

    /// <summary>Vi tri dau ngon THEO TAY THAT -- khong phai vi tri ngon ao da bi dat len mat vat
    /// (dang cam, ngon ao luon nam tren mat nen doc no se khong bao gio thay tay mo ra de tha).</summary>
    private static Vector3 Tracked(Transform tip) => FingertipSurfaceConstraint.TrackedPosition(tip);

    /// <summary>Dau ngon nay (ngon cai / tro) co thuoc tay dang cam vat khong.</summary>
    public bool IsHoldingTip(Transform tip) =>
        _heldBy >= 0 && tip != null && (_hands[_heldBy].thumbTip == tip || _hands[_heldBy].indexTip == tip);

    /// <summary>Mat hop ma dau ngon nay dang ep vao (dang cam). -1 = khong cam / khong phai hop.</summary>
    public int HeldFace(Transform dataTip)
    {
        if (_heldBy < 0 || dataTip == null) return -1;
        if (_hands[_heldBy].thumbTip == dataTip) return _heldFace[0];
        if (_hands[_heldBy].indexTip == dataTip) return _heldFace[1];
        return -1;
    }

    /// <summary>Truot vat (chi theo HUONG KEP) de tam vat ve giua 2 dau ngon: 2 ngon luon om
    /// 2 mat deu nhau. Lech theo huong khac (vd cam gan 1 dau hop) duoc giu nguyen.</summary>
    private void RecenterBetweenFingers(Pose anchor, float dt)
    {
        Vector3 a = Tracked(_hands[_heldBy].thumbTip), b = Tracked(_hands[_heldBy].indexTip);
        Vector3 axis = b - a;
        if (axis.sqrMagnitude < 1e-8f) return;
        axis.Normalize();
        Vector3 pos = anchor.position + anchor.rotation * _localPos;
        Vector3 center = pos + (_collider.bounds.center - _rb.position);
        float along = Vector3.Dot(center - (a + b) * 0.5f, axis);
        float k = 1f - Mathf.Exp(-_recenterSpeed * dt);
        pos -= axis * (along * k);
        _localPos = Quaternion.Inverse(anchor.rotation) * (pos - anchor.position);
    }

    /// <summary>Diem gan: co tay (neu co) hoac diem giua 2 dau ngon, xoay theo truc 2 ngon.</summary>
    private Pose Anchor(int h)
    {
        Transform w = _hands[h].wrist;
        if (w != null && w.gameObject.activeInHierarchy) return new Pose(w.position, w.rotation);
        Vector3 a = Tracked(_hands[h].thumbTip), b = Tracked(_hands[h].indexTip);
        Vector3 axis = b - a;
        Quaternion rot = axis.sqrMagnitude > 1e-8f ? Quaternion.LookRotation(axis.normalized, Vector3.up) : Quaternion.identity;
        return new Pose((a + b) * 0.5f, rot);
    }

    /// <summary>Khoang cach tu DA dau ngon toi be mat vat (am = da da lot vao trong).</summary>
    public float SkinDistance(Vector3 point)
    {
        Vector3 closest = ClosestSurfacePoint(point, out bool inside);
        float d = Vector3.Distance(point, closest);
        return (inside ? -d : d) - _fingerRadius;
    }

    /// <summary>Be day vat giua 2 ngon + 2 lan ban kinh ngon (met) = khe 2 dau ngon khi vua cham 2 mat.
    /// Hop dang kep 2 mat doi dien: lay be day hop theo truc kep (co dinh). Truoc day lay 2 diem be mat
    /// gan moi ngon nhat -- ngon that bop sau vao trong hop (khong co gi chan) thi 2 diem do roi sang mat
    /// ben canh, "be day" teo con ~1-2 cm, moc tha tut theo -> chi noi tay nhe (khe van &lt; be day hop)
    /// la hop roi (glove_diag 03/10 18:06: 4 lan tha khi khe 4.5-5.6 cm, hop 4 cm + ngon = 5.6 cm).</summary>
    private float ObjectWidthBetweenFingers(int h)
    {
        int f0 = _heldFace[0], f1 = _heldFace[1];
        if (f0 >= 0 && f1 >= 0 && f0 / 2 == f1 / 2 && f0 != f1 && _collider is BoxCollider box)
        {
            int axis = f0 / 2;
            return box.size[axis] * Mathf.Abs(box.transform.lossyScale[axis]) + 2f * _fingerRadius;
        }
        Vector3 a = Tracked(_hands[h].thumbTip), b = Tracked(_hands[h].indexTip);
        Vector3 pa = ClosestSurfacePoint(a, out _), pb = ClosestSurfacePoint(b, out _);
        return Vector3.Distance(pa, pb) + 2f * _fingerRadius;
    }

    /// <summary>Diem gan nhat TREN be mat (ca khi diem nam trong vat) -- Box, Sphere.
    /// Collider khac: dung Physics.ClosestPoint (chi dung khi diem nam ngoai).</summary>
    public Vector3 ClosestSurfacePoint(Vector3 world, out bool inside)
    {
        inside = false;
        if (_collider is BoxCollider box)
        {
            Transform t = box.transform;
            Vector3 half = box.size * 0.5f;
            Vector3 p = t.InverseTransformPoint(world) - box.center;
            Vector3 q = new Vector3(Mathf.Clamp(p.x, -half.x, half.x), Mathf.Clamp(p.y, -half.y, half.y), Mathf.Clamp(p.z, -half.z, half.z));
            if (q == p)
            {
                inside = true;
                // Day ra mat gan nhat (tinh theo met, vi box co the bi scale khong deu)
                Vector3 s = t.lossyScale;
                float dx = (half.x - Mathf.Abs(p.x)) * Mathf.Abs(s.x);
                float dy = (half.y - Mathf.Abs(p.y)) * Mathf.Abs(s.y);
                float dz = (half.z - Mathf.Abs(p.z)) * Mathf.Abs(s.z);
                if (dx <= dy && dx <= dz) q.x = Mathf.Sign(p.x) * half.x;
                else if (dy <= dz) q.y = Mathf.Sign(p.y) * half.y;
                else q.z = Mathf.Sign(p.z) * half.z;
            }
            return t.TransformPoint(q + box.center);
        }
        if (_collider is SphereCollider sphere)
        {
            Vector3 c = sphere.transform.TransformPoint(sphere.center);
            float r = sphere.radius * Mathf.Max(Mathf.Abs(sphere.transform.lossyScale.x),
                Mathf.Max(Mathf.Abs(sphere.transform.lossyScale.y), Mathf.Abs(sphere.transform.lossyScale.z)));
            Vector3 d = world - c;
            inside = d.sqrMagnitude < r * r;
            return c + (d.sqrMagnitude > 1e-12f ? d.normalized : Vector3.up) * r;
        }
        return Physics.ClosestPoint(world, _collider, transform.position, transform.rotation);
    }

    // --- Cho FingertipSurfaceConstraint: ngon ao khong xuyen vat, dang cam thi nam tren be mat ---

    /// <summary>Da ngon lot vao trong vat -> diem (tam dau ngon) de da ngon vua cham be mat.</summary>
    public bool TryResolvePenetration(Transform tip, out Vector3 resolved) =>
        ResolvePoint(tip.position, _fingerRadius, out resolved);

    /// <summary>Mot diem tren TRUC ngon tay (ngon day `radius` met) dang lot vao vat -> diem
    /// gan nhat de mat ngon vua cham be mat. Dung cho MOI dot ngon, khong chi dau ngon.</summary>
    public bool ResolvePoint(Vector3 point, float radius, out Vector3 resolved)
    {
        resolved = point;
        Vector3 s = ClosestSurfacePoint(point, out bool inside);
        Vector3 d = point - s;
        if (!inside && d.sqrMagnitude >= radius * radius) return false;
        Vector3 outward = inside ? -d : d;
        outward = outward.sqrMagnitude > 1e-12f ? outward.normalized : OutwardNormal(point);
        resolved = s + outward * radius;
        return true;
    }

    /// <summary>Nhu ResolvePoint, nhung diem LOT VAO TRONG hop luon duoc day ra mat `face` da chot
    /// (khong phai mat gan nhat) -- gan canh hop, mat gan nhat doi qua lai giua 2 mat moi khung
    /// lam ngon ao nhay. face &lt; 0: nhu ResolvePoint.</summary>
    public bool ResolvePoint(Vector3 point, float radius, int face, out Vector3 resolved)
    {
        if (face < 0 || !(_collider is BoxCollider box)) return ResolvePoint(point, radius, out resolved);
        Transform t = box.transform;
        Vector3 half = box.size * 0.5f;
        Vector3 p = t.InverseTransformPoint(point) - box.center;
        bool inside = Mathf.Abs(p.x) <= half.x && Mathf.Abs(p.y) <= half.y && Mathf.Abs(p.z) <= half.z;
        if (!inside) return ResolvePoint(point, radius, out resolved); // ben ngoai: diem gan nhat la duy nhat, khong nhay
        int axis = face / 2;
        float sign = (face & 1) == 1 ? 1f : -1f;
        float depth = (half[axis] - sign * p[axis]) * Mathf.Abs(t.lossyScale[axis]); // met tu diem toi mat da chot
        resolved = point + FaceNormal(face) * (depth + radius);
        return true;
    }

    /// <summary>Dau ngon cua tay DANG CAM (ho ra toi 3 cm hay lot vao) -> dat len be mat.
    /// Hop: diem cham co dinh tren mat da chot luc cam.</summary>
    public bool TryGetHeldContact(Transform dataTip, Vector3 visualTip, out Vector3 contact)
    {
        contact = visualTip;
        if (!IsHoldingTip(dataTip)) return false;
        int finger = _hands[_heldBy].thumbTip == dataTip ? 0 : 1;
        int face = _heldFace[finger];
        // Hop: dang cam la ngon NAM TREN MAT, du tay that (nhieu tracking) ho ra bao xa -- viec
        // tha do UpdateHeld quyet dinh (mo ngon ro rang du _releaseDelay). Truoc day ho > 3 cm la
        // ngon ao mo theo roi khung sau lai hut vao -> ngon cai dong mo lien tuc.
        if (face < 0 && SkinDistance(visualTip) > 0.03f) return false;
        contact = face >= 0
            ? _collider.transform.TransformPoint(_heldPoint[finger]) + FaceNormal(face) * _fingerRadius
            : SurfacePlusFinger(visualTip);
        return true;
    }

    private Vector3 SurfacePlusFinger(Vector3 p)
    {
        Vector3 s = ClosestSurfacePoint(p, out bool inside);
        Vector3 outward = inside ? s - p : p - s;
        if (outward.sqrMagnitude < 1e-12f) outward = s - _collider.bounds.center;
        return s + outward.normalized * _fingerRadius;
    }
}
