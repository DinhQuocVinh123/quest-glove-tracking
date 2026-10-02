using UnityEngine;

/// <summary>
/// Dong hoc cua MOT ngon tay ao: 4 goc khop (xoe goc, gap goc, gap giua, gap dau) -> vi tri cac khop,
/// va ghi goc vao xuong. HandImageFit dung de khop ban tay ao vao diem anh; FingerUDPReceiver dung de
/// ep 3 ngon giua/ap ut/ut ve dang nam.
///
/// Ngon chi duoc gap theo cach khop that cho phep (gap vao long ban tay, trong gioi han goc) -- nho vay
/// anh 2D (khong co chieu sau) van suy ra duoc dang 3D: doan ngan tren anh = dot xuong chia vao/ra
/// camera, va vi khop chi gap ve MOT phia nen chi con mot cach gap khop voi anh (Taylor, CVIU 2000).
/// </summary>
public sealed class FingerChainFitter
{
    private const int Bones = 3;
    private const int Params = 4; // [0] xoe o goc, [1] gap o goc, [2] gap khop giua, [3] gap khop dau

    private readonly Transform[] _bones;
    private readonly bool _isThumb;
    private readonly Quaternion[] _rest = new Quaternion[Bones];
    private readonly Vector3[] _offset = new Vector3[Bones];   // khop ke tiep, trong he toa do cua xuong (chua nhan scale)
    private readonly Vector3[] _flexAxis = new Vector3[Bones]; // truc gap cua tung xuong (local)
    private readonly Vector3 _spreadAxis;                      // truc xoe cua xuong goc (local)
    private readonly float[] _min;
    private readonly float[] _max;
    private readonly float[] _angles = new float[Params];

    /// <summary>Goc hien tai (do): xoe, gap goc, gap giua, gap dau.</summary>
    public float[] Angles => _angles;

    /// <summary>Goc xoe ngang toi da cua 4 ngon (do). Ngon tro that xoe duoc ~20-30 do.</summary>
    public static float FingerSpreadLimit = 30f;

    /// <param name="bones">3 dot xuong tu goc ra ngoai, O TU THE NGHI (luc Awake).</param>
    /// <param name="tip">Diem dau ngon (con cua dot cuoi).</param>
    /// <param name="palmTarget">Mot diem nam phia LONG ban tay -- de tu tim truc gap NGON CAI: gap = dau ngon tien lai gan diem nay.</param>
    /// <param name="palmNormal">Huong ve phia long ban tay (the gioi) -- truc gap 4 ngon: gap = dau ngon di THANG xuong phia long
    /// ban tay. De trong = dung palmTarget cho ca 4 ngon (cach cu).</param>
    public FingerChainFitter(Transform[] bones, Transform tip, bool isThumb, Vector3 palmTarget, Vector3 palmNormal = default)
    {
        _bones = bones;
        _isThumb = isThumb;
        float scale = Mathf.Max(bones[0].lossyScale.x, 1e-6f);
        for (int j = 0; j < Bones; j++)
        {
            _rest[j] = bones[j].localRotation;
            Transform next = j + 1 < Bones ? bones[j + 1] : tip;
            _offset[j] = Quaternion.Inverse(bones[j].rotation) * (next.position - bones[j].position) / scale;
        }

        // Tu tim truc gap cua tung dot (khong can biet truoc rig dung truc X hay Y):
        //  - 4 ngon: truc nao xoay duong lam dau ngon di XUONG phia long ban tay nhieu nhat.
        //  - ngon cai: truc nao lam dau ngon tien lai gan giua long ban tay nhat (ngon cai gap cheo qua long).
        // Truoc day 4 ngon cung dung cach "tien gan giua long ban tay": ngon tro nam o mep ban tay nen
        // voi khop DAU, gap NGANG ve phia giua lai "gan" hon gap xuong -> gap khop dau 40 do lam dau ngon
        // dich 7 mm sang ngang, 0 mm xuong -> ngon tro ao cong ngoan nhu con ran.
        bool byNormal = !isThumb && palmNormal.sqrMagnitude > 1e-8f;
        for (int j = 0; j < Bones; j++)
            _flexAxis[j] = byNormal ? FindFlexAxisByNormal(j, palmNormal.normalized) : FindFlexAxis(j, palmTarget);
        Vector3 along = _offset[0].normalized;
        _spreadAxis = Vector3.Cross(along, _flexAxis[0]).normalized;

        // Gioi han goc khop (do, so voi tu the nghi cua rig). Ngon cai hep theo tam van dong that
        // (khop dot dau chi be nguoc duoc chut it).
        _min = isThumb ? new[] { -35f, -20f, -10f, -10f } : new[] { -FingerSpreadLimit, -15f, 0f, -5f };
        _max = isThumb ? new[] { 35f, 45f, 70f, 85f } : new[] { FingerSpreadLimit, 95f, 110f, 90f };
    }

    public float MinAngle(int p) => _min[p];
    public float MaxAngle(int p) => _max[p];
    public bool IsThumb => _isThumb;
    public Transform RootBone => _bones[0];

    public void SetAngles(float spread, float flex0, float flex1, float flex2)
    {
        _angles[0] = spread;
        _angles[1] = flex0;
        _angles[2] = flex1;
        _angles[3] = flex2;
        for (int p = 0; p < Params; p++) _angles[p] = Mathf.Clamp(_angles[p], _min[p], _max[p]);
    }

    /// <summary>Ghi goc vao xuong. blend: 1 = theo goc hien tai, 0 = tu the nghi.</summary>
    public void Apply(float blend)
    {
        for (int j = 0; j < Bones; j++)
        {
            Quaternion posed = _rest[j] * LocalDelta(j, _angles);
            _bones[j].localRotation = blend >= 1f ? posed : Quaternion.Slerp(_rest[j], posed, blend);
        }
    }

    /// <summary>Vi tri cac khop voi bo goc a, khi goc ngon dat o <paramref name="root"/> va xuong cha
    /// (vd xuong ban tay) co huong <paramref name="parentRot"/> -- tinh tay, khong dong vao Transform.
    /// joints[0] = goc ngon, joints[3] = dau ngon.</summary>
    public void JointsAt(float[] a, Vector3 root, Quaternion parentRot, Vector3[] joints)
    {
        float scale = _bones[0].lossyScale.x;
        Quaternion rot = parentRot;
        joints[0] = root;
        for (int j = 0; j < Bones; j++)
        {
            rot = rot * _rest[j] * LocalDelta(j, a);
            joints[j + 1] = joints[j] + rot * (_offset[j] * scale);
        }
    }

    private Quaternion LocalDelta(int bone, float[] a) => bone switch
    {
        0 => Quaternion.AngleAxis(a[0], _spreadAxis) * Quaternion.AngleAxis(a[1], _flexAxis[0]),
        1 => Quaternion.AngleAxis(a[2], _flexAxis[1]),
        _ => Quaternion.AngleAxis(a[3], _flexAxis[2]),
    };

    private Vector3 FindFlexAxis(int bone, Vector3 palmTarget)
    {
        Vector3 along = _offset[bone].normalized;
        Vector3 best = Vector3.right;
        float bestGain = float.NegativeInfinity;
        Vector3[] candidates = { Vector3.right, Vector3.left, Vector3.up, Vector3.down, Vector3.forward, Vector3.back };

        foreach (Vector3 axis in candidates)
        {
            if (Mathf.Abs(Vector3.Dot(axis, along)) > 0.5f) continue; // truc doc than xuong -> chi xoan, khong gap
            float before = TipDistance(bone, Quaternion.identity, palmTarget);
            float after = TipDistance(bone, Quaternion.AngleAxis(20f, axis), palmTarget);
            float gain = before - after;
            if (gain > bestGain)
            {
                bestGain = gain;
                best = axis;
            }
        }
        // Lam truc vuong goc han voi than xuong
        return Vector3.ProjectOnPlane(best, along).normalized;
    }

    private Vector3 FindFlexAxisByNormal(int bone, Vector3 palmNormal)
    {
        Vector3 along = _offset[bone].normalized;
        Vector3 best = Vector3.right;
        float bestGain = float.NegativeInfinity;
        Vector3[] candidates = { Vector3.right, Vector3.left, Vector3.up, Vector3.down, Vector3.forward, Vector3.back };
        Vector3 before = TipPosition(bone, Quaternion.identity);
        foreach (Vector3 axis in candidates)
        {
            if (Mathf.Abs(Vector3.Dot(axis, along)) > 0.5f) continue;
            float gain = Vector3.Dot(TipPosition(bone, Quaternion.AngleAxis(20f, axis)) - before, palmNormal);
            if (gain > bestGain)
            {
                bestGain = gain;
                best = axis;
            }
        }
        return Vector3.ProjectOnPlane(best, along).normalized;
    }

    private float TipDistance(int bone, Quaternion delta, Vector3 target) => Vector3.Distance(TipPosition(bone, delta), target);

    private Vector3 TipPosition(int bone, Quaternion delta)
    {
        Quaternion rot = _bones[0].parent != null ? _bones[0].parent.rotation : Quaternion.identity;
        float scale = _bones[0].lossyScale.x;
        Vector3 p = _bones[0].position;
        for (int j = 0; j < Bones; j++)
        {
            rot = rot * _rest[j] * (j == bone ? delta : Quaternion.identity);
            p += rot * (_offset[j] * scale);
        }
        return p;
    }

    /// <summary>Huong ve phia LONG ban tay (the gioi). Can biet tay trai hay phai.</summary>
    public static Vector3 PalmNormal(Transform wrist, Transform indexMcp, Transform middleMcp, Transform pinkyMcp, bool leftHand)
    {
        Vector3 fingerDir = middleMcp.position - wrist.position;
        Vector3 across = indexMcp.position - pinkyMcp.position;
        return Vector3.Cross(fingerDir, across).normalized * (leftHand ? -1f : 1f);
    }

    /// <summary>Mot diem nam phia LONG ban tay (giua long ban tay, nho ra 3 cm)
    /// -- de tu tim truc gap. Can biet tay trai hay phai de biet phia nao la long.</summary>
    public static Vector3 PalmTarget(Transform wrist, Transform indexMcp, Transform middleMcp, Transform pinkyMcp, bool leftHand)
    {
        Vector3 fingerDir = middleMcp.position - wrist.position;
        Vector3 across = indexMcp.position - pinkyMcp.position;
        Vector3 palmFacing = Vector3.Cross(fingerDir, across).normalized * (leftHand ? -1f : 1f);
        Vector3 center = (wrist.position + indexMcp.position + middleMcp.position + pinkyMcp.position) * 0.25f;
        return center + palmFacing * 0.03f;
    }
}
