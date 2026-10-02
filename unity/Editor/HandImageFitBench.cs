using System.Collections.Generic;
using System.Text;
using UnityEngine;

/// <summary>Chay thu HandImageFit trong Editor (khong can kinh): ban tay gia lap co dap an dung,
/// chieu len camera + nhieu -> do sai so va so lan "nhay". Goi tu Unity RunCommand:
/// HandImageFitBench.Synthetic(3f, 8f).</summary>
public static class HandImageFitBench
{
    private static System.Random _rng;
    /// <summary>Vi tri / phap tuyen cum luc giac (he co tay) cho Replay; zero = uoc luong tu rig.</summary>
    public static Vector3 DorsalPoint, DorsalNormal;
    /// <summary>Replay: tai anh so ForceFid, dat dap an = (ForceP, ForceR, ForceA) truoc khi giai (0 = tat).</summary>
    public static int ForceFid;
    public static float HandScale; // >0: kich thuoc tay ao nhu tren kinh (cot hscale)
    public static Vector3 ForceP;
    public static Quaternion ForceR;
    public static float[] ForceA;

    private static float Gauss()
    {
        double u1 = 1.0 - _rng.NextDouble(), u2 = _rng.NextDouble();
        return (float)(System.Math.Sqrt(-2.0 * System.Math.Log(u1)) * System.Math.Cos(2 * System.Math.PI * u2));
    }

    public static bool BuildModel(out Transform wrist, out FingerChainFitter thumb, out FingerChainFitter index)
    {
        wrist = null; thumb = index = null;
        var recv = Object.FindAnyObjectByType<FingerUDPReceiver>();
        if (recv == null) return false;
        var rig = recv.GetComponent<HandFingerRig>();
        wrist = rig.indexProximal.parent;
        while (wrist != null && !wrist.name.Contains("Wrist")) wrist = wrist.parent;
        Vector3 palmT = FingerChainFitter.PalmTarget(wrist, rig.indexProximal, rig.middleProximal, rig.pinkyProximal, false);
        thumb = new FingerChainFitter(new[] { rig.thumbMetacarpal, rig.thumbProximal, rig.thumbDistal }, rig.thumbDistal.GetChild(0), true, palmT);
        index = new FingerChainFitter(new[] { rig.indexProximal, rig.indexIntermediate, rig.indexDistal }, rig.indexDistal.GetChild(0), false, palmT,
            FingerChainFitter.PalmNormal(wrist, rig.indexProximal, rig.middleProximal, rig.pinkyProximal, false));
        return true;
    }

    /// <summary>Nhu tren nhung du 5 ngon + diem phia long ban tay -- de dung mo hinh co cum luc giac.</summary>
    public static bool BuildFull(out Transform wrist, out FingerChainFitter[] fitters, out Vector3 palmT)
    {
        wrist = null; fitters = null; palmT = Vector3.zero;
        var recv = Object.FindAnyObjectByType<FingerUDPReceiver>();
        if (recv == null) return false;
        var rig = recv.GetComponent<HandFingerRig>();
        wrist = rig.indexProximal.parent;
        while (wrist != null && !wrist.name.Contains("Wrist")) wrist = wrist.parent;
        palmT = FingerChainFitter.PalmTarget(wrist, rig.indexProximal, rig.middleProximal, rig.pinkyProximal, false);
        Transform[][] chains =
        {
            new[] { rig.thumbMetacarpal, rig.thumbProximal, rig.thumbDistal },
            new[] { rig.indexProximal, rig.indexIntermediate, rig.indexDistal },
            new[] { rig.middleProximal, rig.middleIntermediate, rig.middleDistal },
            new[] { rig.ringProximal, rig.ringIntermediate, rig.ringDistal },
            new[] { rig.pinkyProximal, rig.pinkyIntermediate, rig.pinkyDistal },
        };
        fitters = new FingerChainFitter[5];
        Vector3 palmN = FingerChainFitter.PalmNormal(wrist, rig.indexProximal, rig.middleProximal, rig.pinkyProximal, false);
        for (int f = 0; f < 5; f++) fitters[f] = new FingerChainFitter(chains[f], chains[f][2].GetChild(0), f == 0, palmT, palmN);
        return true;
    }

    public static string Synthetic(params float[] noises)
    {
        var sb = new StringBuilder();
        if (!BuildModel(out Transform wrist, out var thumb, out var index)) return "khong tim thay FingerUDPReceiver";
        var rig = Object.FindAnyObjectByType<FingerUDPReceiver>().GetComponent<HandFingerRig>();
        Vector3 palmT = FingerChainFitter.PalmTarget(wrist, rig.indexProximal, rig.middleProximal, rig.pinkyProximal, false);
        Vector3 C = rig.indexProximal.position;
        Vector3 fingerDir = (rig.middleProximal.position - wrist.position).normalized;
        Vector3 normal = (palmT - (wrist.position + rig.indexProximal.position + rig.middleProximal.position + rig.pinkyProximal.position) * 0.25f).normalized;
        Vector3[] camDirs =
        {
            (-normal * 0.2f - fingerDir * 0.5f + Vector3.up * 0.6f).normalized,
            (normal * 0.7f - fingerDir * 0.3f).normalized,
            (-normal * 0.7f - fingerDir * 0.2f).normalized,
        };

        foreach (float noisePx in noises)
            for (int cd = 0; cd < camDirs.Length; cd++)
            {
                _rng = new System.Random(1 + cd);
                var fit = new HandImageFit(wrist, thumb, index);
                Vector3 camPos = C + camDirs[cd] * 0.4f;
                Quaternion camRot = Quaternion.LookRotation(C - camPos, Vector3.up);
                Quaternion camInv = Quaternion.Inverse(camRot);
                Vector3 camRight = camRot * Vector3.right, camUp = camRot * Vector3.up;
                Vector3 P0 = wrist.position;
                Quaternion R0 = wrist.rotation;
                var gt = new Vector3[9]; var sol = new Vector3[9]; var dirs = new Vector3[9]; var w = new float[9];
                var eW = new List<float>(); var eTip = new List<float>(); var eRot = new List<float>(); var eDepth = new List<float>();
                float imgSum = 0f, tSum = 0f;
                int jumpsP = 0, jumpsR = 0, jumpsA = 0, n = 0;
                Vector3 prevP = Vector3.zero, prevGP = Vector3.zero;
                Quaternion prevR = Quaternion.identity, prevGR = Quaternion.identity;
                float[] prevA = new float[8], prevGA = new float[8];
                float npx = noisePx / 900f;
                for (int f = 0; f < 150; f++)
                {
                    float t = f * 0.111f;
                    Vector3 P = P0 + new Vector3(0.05f * Mathf.Sin(0.4f * t), 0.03f * Mathf.Sin(0.3f * t), 0.04f * Mathf.Sin(0.25f * t));
                    Quaternion R = Quaternion.AngleAxis(20f * Mathf.Sin(0.5f * t), camRight) * Quaternion.AngleAxis(15f * Mathf.Sin(0.35f * t), camUp) * R0;
                    float s = 0.5f - 0.5f * Mathf.Cos(0.6f * t);
                    float[] ga = { 5f * s, 15f * s, 30f * s, 25f * s, 0f, 50f * s, 50f * s, 33f * s };
                    fit.ComputePoints(P, R, ga, gt);
                    for (int i = 0; i < 9; i++)
                    {
                        Vector3 c = camInv * (gt[i] - camPos);
                        dirs[i] = camRot * new Vector3(c.x / c.z + Gauss() * npx, c.y / c.z + Gauss() * npx, 1f);
                        w[i] = i == 1 ? 0.5f : 1f;
                    }
                    float gtDepth = (camInv * (gt[5] - camPos)).z;
                    Quaternion noisyQ = Quaternion.Euler(Gauss() * 15f, Gauss() * 15f, Gauss() * 15f) * R;
                    float t0 = Time.realtimeSinceStartup;
                    fit.Solve(camPos, camRot, dirs, w, gtDepth + Gauss() * 0.02f, noisyQ, f > 0);
                    tSum += Time.realtimeSinceStartup - t0;
                    fit.GetModelPoints(sol);
                    if (f >= 5)
                    {
                        n++;
                        eW.Add((sol[0] - gt[0]).magnitude);
                        eTip.Add(Mathf.Max((sol[4] - gt[4]).magnitude, (sol[8] - gt[8]).magnitude));
                        eRot.Add(Quaternion.Angle(fit.Rotation, R));
                        eDepth.Add(Mathf.Abs(fit.Depth - gtDepth));
                        imgSum += fit.ImageError;
                        float dP = (fit.Position - prevP).magnitude - (P - prevGP).magnitude;
                        float dR = Quaternion.Angle(fit.Rotation, prevR) - Quaternion.Angle(R, prevGR);
                        float dA = 0f;
                        for (int k = 0; k < 8; k++) dA = Mathf.Max(dA, Mathf.Abs(fit.Angles[k] - prevA[k]) - Mathf.Abs(ga[k] - prevGA[k]));
                        if (dP > 0.02f) jumpsP++;
                        if (dR > 10f) jumpsR++;
                        if (dA > 15f) jumpsA++;
                    }
                    prevP = fit.Position; prevR = fit.Rotation; System.Array.Copy(fit.Angles, prevA, 8);
                    prevGP = P; prevGR = R; System.Array.Copy(ga, prevGA, 8);
                }
                sb.AppendFormat("nhieu {0}px cam{1}: co tay med {2:F1} p95 {3:F1} cm | sau med {4:F1} p95 {5:F1} cm | dau ngon med {6:F1} p95 {7:F1} cm | huong med {8:F0} p95 {9:F0} do | anh {10:F3} | nhay vt {11} huong {12} ngon {13} /{14} | doi {15} | {16:F2} ms\n",
                    noisePx, cd, Med(eW) * 100, P95(eW) * 100, Med(eDepth) * 100, P95(eDepth) * 100, Med(eTip) * 100, P95(eTip) * 100,
                    Med(eRot), P95(eRot), imgSum / n, jumpsP, jumpsR, jumpsA, n, fit.Switches, tSum / 150 * 1000);
            }
        return sb.ToString();
    }

    /// <summary>Chay lai ban ghi that. Moi dong file: t, fid, 9 diem pixel (x,y), 9 do tin cay, vi tri + huong
    /// mat (thay cho camera), co tay Quest (vi tri + huong). Camera gan dung: tieu cu fx, tam anh giua.
    /// Gia lap hien thi 60 Hz (truot muot nhu ImageHandSolver) de dem so lan nhay giong glove_diag.
    /// outPath: ghi 9 diem ket qua chieu len anh (pixel) cho tung anh -- de ve len anh kiem tra.</summary>
    /// File co them 4 cot cuoi (tuy chon): cum luc giac 1/0, x, y (pixel), dien tich.
    public static string Replay(string path, string outPath, bool useDorsal = true, bool useQuestRot = false,
                                float fx = 950f, float width = 1280f, float height = 960f)
    {
        if (!BuildFull(out Transform wrist, out var fitters, out Vector3 palmT)) return "khong tim thay FingerUDPReceiver";
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        var lines = System.IO.File.ReadAllLines(path);
        var fit = new HandImageFit(wrist, fitters, palmT, new Vector3(75f, 95f, 60f));
        if (DorsalNormal != Vector3.zero) fit.SetDorsalModel(DorsalPoint, DorsalNormal);
        if (HandScale > 0f) fit.SetScale(HandScale / Mathf.Max(wrist.lossyScale.x, 1e-4f));
        var rigR = Object.FindAnyObjectByType<FingerUDPReceiver>().GetComponent<HandFingerRig>();
        Quaternion wInv = Quaternion.Inverse(wrist.rotation);
        Vector3 midL = wInv * (rigR.middleProximal.position - wrist.position);
        Vector3 pinkyL = wInv * (rigR.pinkyProximal.position - wrist.position);
        var dirs = new Vector3[9]; var w = new float[9]; var pts = new Vector3[9]; var fj = new Vector3[4];
        var o = new StringBuilder();
        float cx = width * 0.5f, cy = height * 0.5f;
        float lastSolve = -999f, depthPrior = 0f, lastT = 0f; bool hasDepth = false, hasRot = false;
        Quaternion rotPrior = Quaternion.identity;
        // hien thi
        Vector3 dP = Vector3.zero; Quaternion dR = Quaternion.identity; var dA = new float[8]; bool hasDisp = false;
        Vector3 prevDP = Vector3.zero; Quaternion prevDR = Quaternion.identity; var prevDA = new float[8];
        int frames = 0, jumpP = 0, jumpR = 0, jumpA = 0, solves = 0, fails = 0;
        var imgErr = new List<float>(); var questDiff = new List<float>(); var depthDiff = new List<float>();
        double t0 = 0; float tDisp = 0f; float msSum = 0f;
        for (int li = 0; li < lines.Length; li++)
        {
            var c = lines[li].Split(',');
            if (c.Length < 43) continue;
            float F(int i) => float.Parse(c[i], inv);
            double tAbs = double.Parse(c[0], inv);
            if (li == 0) t0 = tAbs;
            float t = (float)(tAbs - t0);
            // hien thi 60 Hz tu anh truoc toi anh nay
            while (tDisp < t)
            {
                if (fit.HasSolution && hasDisp && tDisp - lastSolve <= 1f)
                {
                    float a = 1f - Mathf.Exp(-(1f / 60f) / 0.06f);
                    dP = Vector3.Lerp(dP, fit.Position, a); dR = Quaternion.Slerp(dR, fit.Rotation, a);
                    for (int k = 0; k < 8; k++) dA[k] = Mathf.Lerp(dA[k], fit.Angles[k], a);
                    if (frames > 0)
                    {
                        if ((dP - prevDP).magnitude > 0.02f) jumpP++;
                        if (Quaternion.Angle(dR, prevDR) > 5f) jumpR++;
                        float m = 0f; for (int k = 0; k < 8; k++) m = Mathf.Max(m, Mathf.Abs(dA[k] - prevDA[k]));
                        if (m > 10f) jumpA++;
                    }
                    frames++;
                    prevDP = dP; prevDR = dR; System.Array.Copy(dA, prevDA, 8);
                }
                tDisp += 1f / 60f;
            }

            Vector3 camPos = new Vector3(F(29), F(30), F(31));
            Quaternion camRot = new Quaternion(F(32), F(33), F(34), F(35));
            Vector3 rw = new Vector3(F(36), F(37), F(38));
            Quaternion rq = new Quaternion(F(39), F(40), F(41), F(42));
            for (int i = 0; i < 9; i++)
            {
                float px = F(2 + 2 * i), py = F(3 + 2 * i), sc = F(20 + i);
                dirs[i] = camRot * new Vector3((px - cx) / fx, -(py - cy) / fx, 1f);
                w[i] = sc >= 0.2f ? (i == 1 ? 0.5f : 1f) : 0f;
            }
            float since = t - lastT;
            float z = Vector3.Dot(rw - camPos, camRot * Vector3.forward) + 0.03f; // tam tay xa hon co tay chut
            depthPrior = !hasDepth || since > 2f ? z : Mathf.Lerp(depthPrior, z, 1f - Mathf.Exp(-since / 0.6f));
            rotPrior = !hasRot || since > 1f ? rq : Quaternion.Slerp(rotPrior, rq, 1f - Mathf.Exp(-since / 0.3f));
            hasDepth = hasRot = true;
            lastT = t;

            int dorsal = -1; Vector3 dorsalDir = Vector3.zero; float dpx = 0f, dpy = 0f;
            if (useDorsal && c.Length >= 47)
            {
                dorsal = (int)F(43); dpx = F(44); dpy = F(45);
                if (dorsal == 1) dorsalDir = camRot * new Vector3((dpx - cx) / fx, -(dpy - cy) / fx, 1f);
            }
            if (ForceFid > 0 && (int)F(1) == ForceFid) fit.SetSolution(ForceP, ForceR, ForceA);
            float s0 = Time.realtimeSinceStartup;
            bool ok = fit.Solve(camPos, camRot, dirs, w, depthPrior, useQuestRot ? rotPrior : (Quaternion?)null,
                                t - lastSolve <= 0.5f, dorsal, dorsalDir);
            msSum += (Time.realtimeSinceStartup - s0) * 1000f;
            if (!ok) { fails++; continue; }
            solves++;
            if (!hasDisp || t - lastSolve > 1f)
            {
                dP = fit.Position; dR = fit.Rotation; System.Array.Copy(fit.Angles, dA, 8); hasDisp = true;
                prevDP = dP; prevDR = dR; System.Array.Copy(dA, prevDA, 8);
            }
            lastSolve = t;
            imgErr.Add(fit.ImageError);
            questDiff.Add(Quaternion.Angle(fit.Rotation, rq));
            depthDiff.Add(Mathf.Abs(fit.Depth - depthPrior));

            if (outPath != null)
            {
                fit.GetModelPoints(pts);
                Quaternion ci = Quaternion.Inverse(camRot);
                o.Append(c[1]);
                for (int i = 0; i < 9; i++)
                {
                    Vector3 q = ci * (pts[i] - camPos);
                    o.Append(',').Append((cx + q.x / q.z * fx).ToString("F1", inv)).Append(',').Append((cy - q.y / q.z * fx).ToString("F1", inv));
                }
                o.Append(',').Append(fit.ImageError.ToString("F4", inv)).Append(',').Append(fit.Pinch.ToString("F2", inv));
                for (int k = 0; k < 8; k++) o.Append(',').Append(fit.Angles[k].ToString("F1", inv));
                Quaternion fr = fit.Rotation;
                foreach (float v in new[] { fr.x, fr.y, fr.z, fr.w, rq.x, rq.y, rq.z, rq.w, fit.Depth, depthPrior, t, fit.Position.x, fit.Position.y, fit.Position.z })
                    o.Append(',').Append(v.ToString("F5", inv));
                // Goc ngon giua + goc ngon ut cua tay ao (khong co trong 9 diem) -> ve mat long ban tay
                fit.TryGetDorsalPoint(out Vector3 dWorld);
                foreach (Vector3 wpt in new[] { fit.Position + fit.Rotation * midL, fit.Position + fit.Rotation * pinkyL, dWorld })
                {
                    Vector3 q = ci * (wpt - camPos);
                    o.Append(',').Append((cx + q.x / q.z * fx).ToString("F1", inv)).Append(',').Append((cy - q.y / q.z * fx).ToString("F1", inv));
                }
                o.Append(',').Append(dorsal).Append(',').Append(dpx.ToString("F1", inv)).Append(',').Append(dpy.ToString("F1", inv));
                // 9 khop trong he camera (met) -- de ve goc nhin ngang
                for (int i = 0; i < 9; i++)
                {
                    Vector3 q = ci * (pts[i] - camPos);
                    o.Append(',').Append(q.x.ToString("F4", inv)).Append(',').Append(q.y.ToString("F4", inv)).Append(',').Append(q.z.ToString("F4", inv));
                }
                // 3 ngon giua/ap ut/ut (dang nam) chieu len anh -- de ve ca ban tay ao
                for (int f = 2; f < 5; f++)
                {
                    Transform rb = fitters[f].RootBone;
                    Vector3 rootL = wInv * (rb.position - wrist.position);
                    Quaternion parL = wInv * rb.parent.rotation;
                    fitters[f].JointsAt(new float[] { 0f, 75f, 95f, 60f }, fit.Position + fit.Rotation * rootL, fit.Rotation * parL, fj);
                    for (int k = 0; k < 4; k++)
                    {
                        Vector3 q = ci * (fj[k] - camPos);
                        o.Append(',').Append((cx + q.x / q.z * fx).ToString("F1", inv)).Append(',').Append((cy - q.y / q.z * fx).ToString("F1", inv));
                    }
                }
                o.Append('\n');
            }
        }
        if (outPath != null) System.IO.File.WriteAllText(outPath, o.ToString());
        return string.Format(inv,
            "{0} anh giai ({1} loi), {2:F2} ms/anh | sai so anh med {3:F3} p90 {4:F3} | lech huong Quest med {5:F0} p90 {6:F0} do | lech sau Quest med {7:F1} p90 {8:F1} cm\n" +
            "hien thi {9} khung 60Hz ({10:F0} s): nhay >2cm {11}, xoay >5 do/khung {12}, ngon >10 do/khung {13} | doi cach hieu {14}",
            solves, fails, msSum / Mathf.Max(1, solves + fails), Med(imgErr), P(imgErr, 0.9f), Med(questDiff), P(questDiff, 0.9f),
            Med(depthDiff) * 100, P(depthDiff, 0.9f) * 100, frames, frames / 60f, jumpP, jumpR, jumpA, fit.Switches);
    }

    /// <summary>Chuyen dang tay TRAN (HandPoseRecorder, Quest theo doi) sang goc khop cua tay gang ao: moi khung
    /// tim 4 goc ngon cai + 4 goc ngon tro sao cho HUONG tung dot trung voi tay Quest (khong phu thuoc co tay to
    /// nho). Tay trai -> lat truc X (ngon cai/tro o +X tay trai, -X tay phai). Ghi: t, 8 goc, sai lech huong
    /// trung binh tung ngon (do), khoang cach 2 dau ngon (m).</summary>
    public static string RetargetQuest(string path, string outPath, string hand = "L")
    {
        if (!BuildFull(out Transform wrist, out var fitters, out _)) return "khong tim thay FingerUDPReceiver";
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        var lines = System.IO.File.ReadAllLines(path);
        var head = lines[0].Split(',');
        int Col(string n) => System.Array.IndexOf(head, n);
        string[][] chains = { new[] { "HandThumb1", "HandThumb2", "HandThumb3", "HandThumbTip" },
                              new[] { "HandIndex1", "HandIndex2", "HandIndex3", "HandIndexTip" } };
        int[][] cols = new int[2][];
        for (int f = 0; f < 2; f++) { cols[f] = new int[4]; for (int k = 0; k < 4; k++) cols[f][k] = Col(chains[f][k] + "_x"); }
        float mirror = hand == "L" ? -1f : 1f;
        var o = new StringBuilder();
        var a = new float[2][] { new float[4], new float[4] };
        var j = new Vector3[4];
        var target = new Vector3[3];
        var errs = new List<float>[] { new List<float>(), new List<float>() };
        int n = 0;
        for (int li = 1; li < lines.Length; li++)
        {
            var c = lines[li].Split(',');
            if (c.Length < head.Length || c[1] != hand || c[2] != "1") continue;
            o.Append(c[0]);
            float[] err = new float[2];
            Vector3[] tips = new Vector3[2];
            for (int f = 0; f < 2; f++)
            {
                var pts = new Vector3[4];
                for (int k = 0; k < 4; k++)
                {
                    int ci = cols[f][k];
                    pts[k] = new Vector3(mirror * float.Parse(c[ci], inv), float.Parse(c[ci + 1], inv), float.Parse(c[ci + 2], inv));
                }
                tips[f] = pts[3];
                for (int k = 0; k < 3; k++) target[k] = wrist.rotation * (pts[k + 1] - pts[k]).normalized;
                FingerChainFitter fit = fitters[f];
                Transform root = fit.RootBone;
                Quaternion parent = root.parent.rotation;
                float Cost(float[] x)
                {
                    fit.JointsAt(x, root.position, parent, j);
                    float s = 0f;
                    for (int k = 0; k < 3; k++) s += (j[k + 1] - j[k]).normalized == Vector3.zero ? 0f : ((j[k + 1] - j[k]).normalized - target[k]).sqrMagnitude;
                    return s;
                }
                // Giam dan theo tung goc (toa do), buoc nho dan -- 4 an, du nhanh cho vai nghin khung
                float[] x = a[f];
                float cost = Cost(x), step = 8f;
                for (int it = 0; it < 60 && step > 0.05f; it++)
                {
                    bool improved = false;
                    for (int p = 0; p < 4; p++)
                        foreach (float s in new[] { step, -step })
                        {
                            x[p] += s;
                            float cc = Cost(x);
                            if (cc < cost) { cost = cc; improved = true; break; }
                            x[p] -= s;
                        }
                    if (!improved) step *= 0.5f;
                }
                fit.JointsAt(x, root.position, parent, j);
                float e = 0f;
                for (int k = 0; k < 3; k++) e += Vector3.Angle(j[k + 1] - j[k], target[k]);
                err[f] = e / 3f;
                errs[f].Add(err[f]);
                for (int p = 0; p < 4; p++) o.Append(',').Append(x[p].ToString("F1", inv));
            }
            o.Append(',').Append(err[0].ToString("F1", inv)).Append(',').Append(err[1].ToString("F1", inv))
             .Append(',').Append((tips[0] - tips[1]).magnitude.ToString("F4", inv)).Append('\n');
            n++;
        }
        System.IO.File.WriteAllText(outPath, o.ToString());
        return string.Format(inv, "{0} khung | lech huong dot: ngon cai med {1:F1} p90 {2:F1} do, ngon tro med {3:F1} p90 {4:F1} do",
            n, Med(errs[0]), P(errs[0], 0.9f), Med(errs[1]), P(errs[1], 0.9f));
    }

    private static float P(List<float> l, float q) { l.Sort(); return l.Count == 0 ? 0f : l[Mathf.Min(l.Count - 1, (int)(l.Count * q))]; }

    private static float Med(List<float> l) { l.Sort(); return l[l.Count / 2]; }
    private static float P95(List<float> l) { l.Sort(); return l[Mathf.Min(l.Count - 1, (int)(l.Count * 0.95f))]; }
}
