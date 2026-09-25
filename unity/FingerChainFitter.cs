using UnityEngine;

/// <summary>
/// Dung lai dang 3D cua MOT ngon tay tu cac diem 2D (anh camera): tim GOC
/// KHOP sao cho ngon tay ao, nhin tu phia camera, trung voi cac diem do.
///
/// Vi sao can: anh 2D khong co chieu sau. Cach cu (xoay tung dot cho chia
/// dung huong 2D roi ep nam phang trong mat phang nhin) sai han khi ngon chia
/// VAO/RA khoi camera -- vd nam tay nhin tu tren xuong: dot goc ngon tro that
/// chia ra truoc (xa mat), tren anh chi la mot doan ngan huong len -> tay ao
/// dung thang ngon tro len.
///
/// O day ngon tay chi duoc gap theo dung cach khop that cho phep (gap vao
/// long ban tay, trong gioi han goc), nen chieu sau tu suy ra duoc: doan ngan
/// tren anh = dot xuong dang chia vao/ra camera, va vi khop chi gap ve MOT
/// phia nen chi con mot cach gap khop voi anh. Day la y tuong "khop mo hinh
/// ban tay vao diem 2D co rang buoc khop" (vd Taylor, CVIU 2000).
///
/// Huong co tay / long ban tay lay tu Quest (HandVisual), khong tu anh.
/// Chieu nhin xap xi vuong goc (bo qua phoi canh) -- du dung khi tay cach
/// camera vai chuc cm.
/// </summary>
public sealed class FingerChainFitter
{
    private const int Bones = 3;
    private const int Params = 4; // [0] xoe o goc, [1] gap o goc, [2] gap khop giua, [3] gap khop dau
    private const int Residuals = 6 + Params + 1 + 3;

    // Trong so (don vi: chieu dai long ban tay tren anh, cho moi DO lech).
    // Nho -> chi co tac dung khi du lieu anh khong du de quyet dinh.
    private const float TemporalWeight = 0.0005f; // giu gan goc cua khung truoc (chong rung). Qua lon se giu chat dap an SAI tu khung dau -- diem 2D da duoc lam muot san o FingerUDPReceiver.
    private const float CouplingWeight = 0.004f; // ngon thuong: khop dau gap ~2/3 khop giua
    private const float SpreadPriorWeight = 0.003f; // khong xoe ngang neu anh khong doi hoi
    private const float HyperextensionWeight = 0.006f; // ngon that hiem khi be nguoc ra mu ban tay

    private readonly Transform[] _bones;
    private readonly bool _isThumb;
    private readonly Quaternion[] _rest = new Quaternion[Bones];
    private readonly Vector3[] _offset = new Vector3[Bones];   // khop ke tiep, trong he toa do cua xuong (chua nhan scale)
    private readonly Vector3[] _flexAxis = new Vector3[Bones]; // truc gap cua tung xuong (local)
    private readonly Vector3 _spreadAxis;                      // truc xoe cua xuong goc (local)
    private readonly float[] _min;
    private readonly float[] _max;

    private readonly float[] _angles = new float[Params];
    private readonly float[] _previous = new float[Params];
    private readonly Vector3[] _joints = new Vector3[Bones + 1];

    // Bo dem cho buoc giai (tranh tao mang moi moi khung hinh)
    private readonly float[] _r = new float[Residuals];
    private readonly float[] _rTry = new float[Residuals];
    private readonly float[,] _jac = new float[Residuals, Params];
    private readonly float[] _trial = new float[Params];
    private readonly float[] _seed = new float[Params];
    private readonly float[,] _normal = new float[Params, Params + 1];
    private readonly float[,] _sys = new float[Params, Params + 1];

    private Vector2 _t1, _t2, _t3;
    private Vector3 _right, _up;
    private float _invPalm2D;

    /// <summary>Goc hien tai (do): xoe, gap goc, gap giua, gap dau.</summary>
    public float[] Angles => _angles;

    /// <param name="bones">3 dot xuong tu goc ra ngoai, O TU THE NGHI (luc Awake).</param>
    /// <param name="tip">Diem dau ngon (con cua dot cuoi).</param>
    /// <param name="palmTarget">Mot diem nam phia LONG ban tay -- de tu tim truc gap: gap = dau ngon tien lai gan diem nay.</param>
    public FingerChainFitter(Transform[] bones, Transform tip, bool isThumb, Vector3 palmTarget)
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

        // Tu tim truc gap cua tung dot: truc nao (vuong goc voi than xuong)
        // xoay duong lam dau ngon tien lai gan long ban tay nhieu nhat. Nho vay
        // khong can biet truoc rig dung truc X hay Y (moi model mot kieu).
        for (int j = 0; j < Bones; j++) _flexAxis[j] = FindFlexAxis(j, palmTarget);
        Vector3 along = _offset[0].normalized;
        _spreadAxis = Vector3.Cross(along, _flexAxis[0]).normalized;

        // Gioi han goc khop (do, so voi tu the nghi cua rig)
        _min = isThumb ? new[] { -45f, -45f, -20f, -20f } : new[] { -20f, -15f, 0f, -5f };
        _max = isThumb ? new[] { 45f, 45f, 75f, 90f } : new[] { 20f, 95f, 110f, 90f };
    }

    public void Reset()
    {
        for (int i = 0; i < Params; i++) _angles[i] = 0f;
    }

    public void SetAngles(float spread, float flex0, float flex1, float flex2)
    {
        _angles[0] = spread;
        _angles[1] = flex0;
        _angles[2] = flex1;
        _angles[3] = flex2;
        Clamp(_angles);
    }

    /// <summary>Tim goc khop khop voi anh.</summary>
    /// <param name="t1">Diem khop ke tiep tru diem goc ngon (tren anh, don vi = chieu dai long ban tay tren anh).</param>
    /// <param name="t2">Khop sau do, cung cach tinh.</param>
    /// <param name="t3">Dau ngon, cung cach tinh.</param>
    /// <param name="right">Huong "sang phai" cua anh trong the gioi.</param>
    /// <param name="up">Huong "len tren" cua anh trong the gioi.</param>
    /// <param name="palm2D">Chieu dai long ban tay AO khi chieu len mat phang anh (met).</param>
    public void Solve(Vector2 t1, Vector2 t2, Vector2 t3, Vector3 right, Vector3 up, float palm2D)
    {
        _t1 = t1; _t2 = t2; _t3 = t3;
        _right = right; _up = up;
        _invPalm2D = 1f / Mathf.Max(palm2D, 1e-5f);
        System.Array.Copy(_angles, _previous, Params);

        // Anh 2D thuong co 2 cach gap trong gan giong nhau (vd ngon THANG chia
        // ra xa camera va ngon CUON vao long ban tay). Giai tu goc cua khung
        // truoc VA tu mot dang mau (duoi / nua nam / nam chat -- moi khung thu
        // 1 dang, luan phien, cho nhe CPU cua kinh), lay ket qua khop anh nhat.
        // Sai khac voi khung truoc bi phat nhe (TemporalWeight), nen chi doi
        // sang dap an khac khi no khop anh ro rang hon.
        float cost = Refine(_angles);

        int s = _nextSeed;
        _nextSeed = (_nextSeed + 1) % Seeds.GetLength(0);
        for (int p = 0; p < Params; p++) _seed[p] = Seeds[s, p];
        Clamp(_seed);
        if (Refine(_seed) < cost) System.Array.Copy(_seed, _angles, Params);
    }

    private int _nextSeed;

    // Dang mau de bat dau giai: xoe, gap goc, gap giua, gap dau (do)
    private static readonly float[,] Seeds =
    {
        { 0f, 0f, 0f, 0f },
        { 0f, 30f, 40f, 25f },
        { 0f, 75f, 95f, 60f },
    };

    /// <summary>Ghi goc vao xuong. blend: 1 = theo ket qua giai, 0 = tu the nghi.</summary>
    public void Apply(float blend)
    {
        for (int j = 0; j < Bones; j++)
        {
            Quaternion posed = _rest[j] * LocalDelta(j, _angles);
            _bones[j].localRotation = blend >= 1f ? posed : Quaternion.Slerp(_rest[j], posed, blend);
        }
    }

    // --- Giai (Levenberg-Marquardt) ------------------------------------------

    private float Refine(float[] a)
    {
        float cost = Evaluate(a, _r);
        float lambda = 1e-2f;
        const float h = 0.5f; // buoc dao ham so (do)

        for (int iter = 0; iter < 12; iter++)
        {
            // Jacobian bang sai phan
            for (int p = 0; p < Params; p++)
            {
                System.Array.Copy(a, _trial, Params);
                _trial[p] += h;
                Evaluate(_trial, _rTry);
                for (int k = 0; k < Residuals; k++) _jac[k, p] = (_rTry[k] - _r[k]) / h;
            }

            // (J^T J + lambda * diag) d = -J^T r
            float[,] m = _normal;
            for (int p = 0; p < Params; p++)
            {
                for (int q = 0; q < Params; q++)
                {
                    float s = 0f;
                    for (int k = 0; k < Residuals; k++) s += _jac[k, p] * _jac[k, q];
                    m[p, q] = s;
                }
                float g = 0f;
                for (int k = 0; k < Residuals; k++) g += _jac[k, p] * _r[k];
                m[p, Params] = -g;
            }

            bool improved = false;
            for (int attempt = 0; attempt < 4 && !improved; attempt++)
            {
                float[,] sys = _sys;
                System.Array.Copy(m, sys, m.Length);
                for (int p = 0; p < Params; p++) sys[p, p] += lambda * (m[p, p] + 1e-6f);
                if (!SolveLinear(sys)) { lambda *= 10f; continue; }

                for (int p = 0; p < Params; p++) _trial[p] = a[p] + Mathf.Clamp(sys[p, Params], -25f, 25f);
                Clamp(_trial);
                float trialCost = Evaluate(_trial, _rTry);
                if (trialCost < cost)
                {
                    bool converged = cost - trialCost < 1e-7f;
                    System.Array.Copy(_trial, a, Params);
                    System.Array.Copy(_rTry, _r, Residuals);
                    cost = trialCost;
                    lambda = Mathf.Max(lambda * 0.3f, 1e-4f);
                    improved = true;
                    if (converged) return cost;
                }
                else
                {
                    lambda *= 10f;
                }
            }
            if (!improved) break;
        }
        return cost;
    }

    /// <summary>Tong binh phuong sai lech voi bo goc a.</summary>
    private float Evaluate(float[] a, float[] r)
    {
        ForwardKinematics(a);
        Vector3 p0 = _joints[0];
        Vector2 Project(Vector3 p) =>
            new Vector2(Vector3.Dot(p - p0, _right), Vector3.Dot(p - p0, _up)) * _invPalm2D;

        Vector2 e1 = Project(_joints[1]) - _t1;
        Vector2 e2 = Project(_joints[2]) - _t2;
        Vector2 e3 = Project(_joints[3]) - _t3;
        r[0] = e1.x; r[1] = e1.y;
        r[2] = e2.x; r[3] = e2.y;
        r[4] = e3.x; r[5] = e3.y;
        for (int p = 0; p < Params; p++) r[6 + p] = TemporalWeight * (a[p] - _previous[p]);
        r[6 + 0] += SpreadPriorWeight * a[0];
        r[6 + Params] = _isThumb ? 0f : CouplingWeight * (a[3] - 0.67f * a[2]);
        // Phat be nguoc: anh 2D khong phan biet duoc gap VE PHIA camera hay RA XA
        // -- khi mo ho thi chon cach gap vao long ban tay (tu nhien hon).
        for (int p = 1; p < Params; p++) r[6 + Params + p] = _isThumb ? 0f : HyperextensionWeight * Mathf.Min(0f, a[p]);

        float sum = 0f;
        for (int k = 0; k < Residuals; k++) sum += r[k] * r[k];
        return sum;
    }

    /// <summary>Vi tri cac khop (the gioi) voi bo goc a -- tinh tay, khong dong vao Transform.</summary>
    private void ForwardKinematics(float[] a)
    {
        Quaternion rot = _bones[0].parent != null ? _bones[0].parent.rotation : Quaternion.identity;
        float scale = _bones[0].lossyScale.x;
        _joints[0] = _bones[0].position;
        for (int j = 0; j < Bones; j++)
        {
            rot = rot * _rest[j] * LocalDelta(j, a);
            _joints[j + 1] = _joints[j] + rot * (_offset[j] * scale);
        }
    }

    private Quaternion LocalDelta(int bone, float[] a) => bone switch
    {
        0 => Quaternion.AngleAxis(a[0], _spreadAxis) * Quaternion.AngleAxis(a[1], _flexAxis[0]),
        1 => Quaternion.AngleAxis(a[2], _flexAxis[1]),
        _ => Quaternion.AngleAxis(a[3], _flexAxis[2]),
    };

    private void Clamp(float[] a)
    {
        for (int p = 0; p < Params; p++) a[p] = Mathf.Clamp(a[p], _min[p], _max[p]);
    }

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

    private float TipDistance(int bone, Quaternion delta, Vector3 target)
    {
        Quaternion rot = _bones[0].parent != null ? _bones[0].parent.rotation : Quaternion.identity;
        float scale = _bones[0].lossyScale.x;
        Vector3 p = _bones[0].position;
        for (int j = 0; j < Bones; j++)
        {
            rot = rot * _rest[j] * (j == bone ? delta : Quaternion.identity);
            p += rot * (_offset[j] * scale);
        }
        return Vector3.Distance(p, target);
    }

    /// <summary>Khu Gauss cho he Params x Params (cot cuoi la ve phai). Ket qua ghi vao cot cuoi.</summary>
    private static bool SolveLinear(float[,] m)
    {
        int n = Params;
        for (int c = 0; c < n; c++)
        {
            int pivot = c;
            for (int r = c + 1; r < n; r++) if (Mathf.Abs(m[r, c]) > Mathf.Abs(m[pivot, c])) pivot = r;
            if (Mathf.Abs(m[pivot, c]) < 1e-12f) return false;
            if (pivot != c)
            {
                for (int k = c; k <= n; k++) (m[c, k], m[pivot, k]) = (m[pivot, k], m[c, k]);
            }
            for (int r = 0; r < n; r++)
            {
                if (r == c) continue;
                float f = m[r, c] / m[c, c];
                for (int k = c; k <= n; k++) m[r, k] -= f * m[c, k];
            }
        }
        for (int r = 0; r < n; r++) m[r, n] /= m[r, r];
        return true;
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
