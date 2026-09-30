using Oculus.Interaction.Input;
using UnityEngine;

/// <summary>
/// On dinh CO TAY cua tay gang do Quest theo doi -- chen vao CHUOI DU LIEU tay
/// cua Meta (giong HandFilter / LastKnownGoodHand cua Interaction SDK), nen moi
/// thu doc tay nay (HandVisual, bong, goi y vi tri cho Python) deu nhan cung vi
/// tri da on dinh.
///
/// Vi sao can: gang moi (day deo den + tam luc giac tren mu ban tay) lam Quest
/// nhan hinh ban tay kem: co tay dung yen vai khung roi NHAY 5-12 cm, xoay dot
/// ngot 20-60 do (do bang glove_diag 30/09: nhay >3 cm 190 lan/phut, gang cu 10).
///
/// Cach lam:
///   1. Nhay XA hon muc tay that co the di trong khoang thoi gian do (toc do
///      toi da _maxSpeed) hoac Quest bao do tin cay thap -> nghi ngo: GIU vi tri
///      cu, khong nhay theo.
///   2. Nghi ngo nhung Quest bao cung 1 cho moi lien tuc _confirmFrames khung ->
///      tay that da o do (vd vung tay nhanh) -> nhan, va TRUOT toi do trong
///      _reacquireSeconds thay vi nhay.
///   3. Rung nho: loc One Euro (Casiez 2012; ban cua Meta).
///   4. Quest mat tay ngan (< _holdSeconds): giu dang tay cuoi cung.
///
/// Gan len 1 GameObject; _iModifyDataFromSourceMono = SyntheticHandData cua tay
/// phai, Update Mode = After Previous Step, Update After = SyntheticHandData.
/// HandVisual cua tay gang tro vao component nay thay vi SyntheticHandData.
/// </summary>
public class GloveWristStabilizer : Hand
{
    [Header("Chong nhay")]
    [Tooltip("Toc do toi da cua tay that (m/s). Co tay nhay xa hon toc do nay cho phep -> coi la loi tracking.")]
    [SerializeField] private float _maxSpeed = 2.5f;
    [Tooltip("Cong them vao nguong nhay (m) -- rung binh thuong khong bi coi la nhay.")]
    [SerializeField] private float _jumpMargin = 0.015f;
    [Tooltip("Toc do xoay toi da cua co tay (do/giay).")]
    [SerializeField] private float _maxAngularSpeed = 400f;
    [SerializeField] private float _rotationMargin = 6f;
    [Tooltip("Vi tri moi phai lap lai lien tuc bay nhieu khung (cach nhau < _confirmRadius) moi duoc nhan.")]
    [SerializeField] private int _confirmFrames = 3;
    [SerializeField] private float _confirmRadius = 0.02f;
    [SerializeField] private float _confirmAngle = 15f;
    [Tooltip("Khi nhan vi tri moi sau khi giu: truot toi do trong khoang nay (giay) thay vi nhay.")]
    [SerializeField] private float _reacquireSeconds = 0.15f;
    [Tooltip("Quest mat tay ngan hon muc nay (giay) thi giu dang tay cuoi cung.")]
    [SerializeField] private float _holdSeconds = 0.5f;

    [Header("Loc rung (One Euro)")]
    [SerializeField] private float _minCutoff = 1.5f;
    [SerializeField] private float _beta = 0.5f;

    /// <summary>So khung dang GIU (bo qua du lieu Quest) -- de chan doan.</summary>
    public int HeldFrames { get; private set; }
    public int Reacquires { get; private set; }

    private readonly HandDataAsset _last = new HandDataAsset();
    private IOneEuroFilter<Vector3> _posFilter;
    private IOneEuroFilter<Quaternion> _rotFilter;
    private bool _has, _hasCandidate;
    private Pose _accepted;          // vi tri Quest cuoi cung duoc tin
    private float _sinceAccepted;    // giay tu lan tin cuoi
    private Pose _candidate;
    private int _candidateCount;
    private Vector3 _glidePos;       // lech con lai sau khi nhan vi tri moi (giam dan ve 0)
    private Quaternion _glideRot = Quaternion.identity;
    private Pose _out;
    private float _lostTime;

    protected virtual void Awake()
    {
        _posFilter = OneEuroFilter.CreateVector3();
        _rotFilter = OneEuroFilter.CreateQuaternion();
    }

    protected override void Apply(HandDataAsset data)
    {
        base.Apply(data);
        float dt = Mathf.Max(Time.deltaTime, 1e-4f);

        // Quest mat tay: giu dang cuoi cung mot chut (tranh tay ao chop tat)
        if (!data.IsDataValid || !data.IsTracked)
        {
            _lostTime += dt;
            if (_has && data.IsConnected && _lostTime <= _holdSeconds)
            {
                data.CopyPosesFrom(_last);
                data.IsDataValid = data.IsTracked = data.IsHighConfidence = true;
                data.RootPoseOrigin = PoseOrigin.SyntheticPose;
                HeldFrames++;
            }
            else _has = false;
            return;
        }
        _lostTime = 0f;

        Pose raw = data.Root;
        if (!_has)
        {
            _has = true;
            _accepted = _out = raw;
            _sinceAccepted = 0f;
            _hasCandidate = false;
            _glidePos = Vector3.zero;
            _glideRot = Quaternion.identity;
            _posFilter.Reset();
            _rotFilter.Reset();
            _posFilter.Step(raw.position, dt);
            _rotFilter.Step(raw.rotation, dt);
            _last.CopyFrom(data);
            return;
        }

        // 1. Du lieu nay co tin duoc khong?
        _sinceAccepted += dt;
        float jump = Vector3.Distance(raw.position, _accepted.position);
        float turn = Quaternion.Angle(raw.rotation, _accepted.rotation);
        bool suspect = !data.IsHighConfidence ||
                       jump > _maxSpeed * _sinceAccepted + _jumpMargin ||
                       turn > _maxAngularSpeed * _sinceAccepted + _rotationMargin;

        bool accept = !suspect;
        if (suspect && data.IsHighConfidence)
        {
            // 2. Quest cu bao cung 1 cho moi -> tay that da o do
            if (_hasCandidate && Vector3.Distance(raw.position, _candidate.position) < _confirmRadius &&
                Quaternion.Angle(raw.rotation, _candidate.rotation) < _confirmAngle)
                _candidateCount++;
            else
            {
                _candidate = raw;
                _candidateCount = 1;
                _hasCandidate = true;
            }
            if (_candidateCount >= _confirmFrames)
            {
                accept = true;
                Reacquires++;
            }
        }

        Pose target;
        if (accept)
        {
            bool reacquire = suspect;
            _accepted = raw;
            _sinceAccepted = 0f;
            _hasCandidate = false;
            target = raw;
            if (reacquire)
            {
                // Bat dau truot: giu nguyen vi tri dang hien, lech giam dan ve 0
                Vector3 f = _posFilter.Step(raw.position, dt);
                Quaternion fr = _rotFilter.Step(raw.rotation, dt);
                _glidePos = _out.position - f;
                _glideRot = _out.rotation * Quaternion.Inverse(fr);
                Output(data, f, fr, dt, alreadyFiltered: true);
                return;
            }
        }
        else
        {
            target = _accepted; // giu cho cu
            HeldFrames++;
        }

        Output(data, target.position, target.rotation, dt, alreadyFiltered: false);
    }

    private void Output(HandDataAsset data, Vector3 pos, Quaternion rot, float dt, bool alreadyFiltered)
    {
        _posFilter.SetProperties(new OneEuroFilterPropertyBlock(_minCutoff, _beta));
        _rotFilter.SetProperties(new OneEuroFilterPropertyBlock(_minCutoff, _beta));
        Vector3 fp = alreadyFiltered ? pos : _posFilter.Step(pos, dt);
        Quaternion fr = alreadyFiltered ? rot : _rotFilter.Step(rot, dt);

        float decay = Mathf.Exp(-dt / Mathf.Max(_reacquireSeconds, 1e-3f));
        _glidePos *= decay;
        _glideRot = Quaternion.Slerp(Quaternion.identity, _glideRot, decay);

        _out = new Pose(fp + _glidePos, _glideRot * fr);
        data.Root = _out;
        data.RootPoseOrigin = PoseOrigin.FilteredTrackedPose;
        _last.CopyFrom(data);
    }
}
