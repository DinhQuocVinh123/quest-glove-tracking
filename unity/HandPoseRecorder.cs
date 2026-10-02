using System;
using System.Collections.Generic;
using Oculus.Interaction;
using Oculus.Interaction.Input;
using UnityEngine;

/// <summary>
/// Ghi lai cach Quest theo doi ban tay TRAN (khop nao xoay bao nhieu, cac khop di cung nhau ra sao) de
/// hoc dang tay tu nhien cho tay gang ao (ImageHandSolver). Moi khung tay dang duoc theo doi: 1 dong CSV
/// voi vi tri + huong cua 26 khop (bo khop OpenXR) TINH TU CO TAY, khong phu thuoc mo hinh tay hien thi.
///
/// File: Application.persistentDataPath/hand_pose_*.csv. Lay ve: adb pull /sdcard/Android/data/[ten goi]/files/
/// Nho TAT (_record) sau khi ghi xong -- moi khung ghi ~2 KB.
/// </summary>
[DefaultExecutionOrder(1000)]
public sealed class HandPoseRecorder : MonoBehaviour
{
    [Tooltip("BAT de ghi. Chi ghi khung co tay dang duoc Quest theo doi.")]
    [SerializeField] private bool _record = true;
    [Tooltip("Cac HandVisual can ghi (moi tay 1 cot 'hand'). De trong = tu tim tay trai (OVRHandVisualLeft) va tay phai (tay gang, lay du lieu Quest goc).")]
    [SerializeField] private HandVisual[] _hands;

    private System.IO.StreamWriter _file;
    private int _rows;
    private readonly System.Text.StringBuilder _sb = new System.Text.StringBuilder(4096);

    private void Start()
    {
        if (_hands != null && _hands.Length > 0) return;
        var list = new List<HandVisual>();
        foreach (var hv in FindObjectsByType<HandVisual>(FindObjectsInactive.Exclude))
        {
            // Tay trai Quest hien thi (OVRHandVisualLeft) va tay gang (co FingerUDPReceiver -> Hand = du lieu Quest goc)
            if (hv.name == "OVRHandVisualLeft" || hv.GetComponent<FingerUDPReceiver>() != null) list.Add(hv);
        }
        _hands = list.ToArray();
        Debug.Log($"[HandPoseRecorder] Ghi {_hands.Length} tay", this);
    }

    private void LateUpdate()
    {
        if (!_record || _hands == null) return;
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        foreach (var hv in _hands)
        {
            IHand hand = hv != null ? hv.Hand : null;
            if (hand == null || !hand.IsTrackedDataValid) continue;
            if (!hand.GetJointPosesFromWrist(out ReadOnlyHandJointPoses poses)) continue;
            if (!hand.GetRootPose(out Pose root)) continue;
            if (_file == null) Open(poses.Count);

            _sb.Clear();
            _sb.Append(Time.time.ToString("F4", inv)).Append(',')
               .Append(hand.Handedness == Handedness.Left ? 'L' : 'R').Append(',')
               .Append(hand.IsHighConfidence ? 1 : 0).Append(',')
               .Append(hand.Scale.ToString("F4", inv));
            Append(root.position, root.rotation, inv);
            Transform head = Camera.main != null ? Camera.main.transform : null;
            Append(head != null ? head.position : Vector3.zero, head != null ? head.rotation : Quaternion.identity, inv);
            for (int i = 0; i < poses.Count; i++) Append(poses[i].position, poses[i].rotation, inv);
            _file.WriteLine(_sb.ToString());
            if (++_rows % 120 == 0) _file.Flush();
        }
    }

    private void Append(Vector3 p, Quaternion q, IFormatProvider inv)
    {
        foreach (float v in new[] { p.x, p.y, p.z }) _sb.Append(',').Append(v.ToString("F5", inv));
        foreach (float v in new[] { q.x, q.y, q.z, q.w }) _sb.Append(',').Append(v.ToString("F5", inv));
    }

    private void Open(int joints)
    {
        string path = System.IO.Path.Combine(Application.persistentDataPath,
            "hand_pose_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + ".csv");
        _file = new System.IO.StreamWriter(path, false, new System.Text.UTF8Encoding(false));
        var head = new System.Text.StringBuilder("t,hand,conf,scale,rx,ry,rz,rqx,rqy,rqz,rqw,hx,hy,hz,hqx,hqy,hqz,hqw");
        for (int i = 0; i < joints; i++)
        {
            string n = ((HandJointId)i).ToString();
            head.Append($",{n}_x,{n}_y,{n}_z,{n}_qx,{n}_qy,{n}_qz,{n}_qw");
        }
        _file.WriteLine(head.ToString());
        Debug.Log($"[HandPoseRecorder] Ghi vao {path}", this);
    }

    private void OnDisable()
    {
        _file?.Dispose();
        _file = null;
    }
}
