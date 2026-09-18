using UnityEngine;

/// <summary>
/// Giai đoạn 2 (Bước 5) — giữ tham chiếu đến các xương ngón tay trong
/// model bàn tay ảo, tự động tìm theo tên (không cần kéo thủ công).
/// Chưa xoay gì cả ở bước này — chỉ dựng khung để Bước 6 (test bằng
/// sine giả) và sau này (dữ liệu cảm biến thật) cắm dữ liệu vào.
///
/// Gắn script này vào GameObject StaticHandModel_Right.
/// </summary>
public class HandFingerRig : MonoBehaviour
{
    [Header("Ngón cái (3 đốt)")]
    public Transform thumbMetacarpal;
    public Transform thumbProximal;
    public Transform thumbDistal;

    [Header("Ngón trỏ (3 khớp cử động)")]
    public Transform indexProximal;
    public Transform indexIntermediate;
    public Transform indexDistal;

    [Header("Ngón giữa")]
    public Transform middleProximal;
    public Transform middleIntermediate;
    public Transform middleDistal;

    [Header("Ngón áp út")]
    public Transform ringProximal;
    public Transform ringIntermediate;
    public Transform ringDistal;

    [Header("Ngón út")]
    public Transform pinkyProximal;
    public Transform pinkyIntermediate;
    public Transform pinkyDistal;

    private void Awake()
    {
        AutoFindMissingBones();
    }

    /// <summary>Tự tìm các xương còn thiếu (chưa gán tay) theo đúng tên trong model.</summary>
    [ContextMenu("Auto Find Bones")]
    public void AutoFindMissingBones()
    {
        thumbMetacarpal   = thumbMetacarpal   ? thumbMetacarpal   : FindDeepChild(transform, "XRHand_ThumbMetacarpal");
        thumbProximal     = thumbProximal     ? thumbProximal     : FindDeepChild(transform, "XRHand_ThumbProximal");
        thumbDistal       = thumbDistal       ? thumbDistal       : FindDeepChild(transform, "XRHand_ThumbDistal");

        indexProximal     = indexProximal     ? indexProximal     : FindDeepChild(transform, "XRHand_IndexProximal");
        indexIntermediate = indexIntermediate ? indexIntermediate : FindDeepChild(transform, "XRHand_IndexIntermediate");
        indexDistal       = indexDistal       ? indexDistal       : FindDeepChild(transform, "XRHand_IndexDistal");

        middleProximal     = middleProximal     ? middleProximal     : FindDeepChild(transform, "XRHand_MiddleProximal");
        middleIntermediate = middleIntermediate ? middleIntermediate : FindDeepChild(transform, "XRHand_MiddleIntermediate");
        middleDistal       = middleDistal       ? middleDistal       : FindDeepChild(transform, "XRHand_MiddleDistal");

        ringProximal     = ringProximal     ? ringProximal     : FindDeepChild(transform, "XRHand_RingProximal");
        ringIntermediate = ringIntermediate ? ringIntermediate : FindDeepChild(transform, "XRHand_RingIntermediate");
        ringDistal       = ringDistal       ? ringDistal       : FindDeepChild(transform, "XRHand_RingDistal");

        // Model gốc đặt tên ngón út là "Little" (theo chuẩn XRHand), không phải "Pinky".
        pinkyProximal     = pinkyProximal     ? pinkyProximal     : FindDeepChild(transform, "XRHand_LittleProximal");
        pinkyIntermediate = pinkyIntermediate ? pinkyIntermediate : FindDeepChild(transform, "XRHand_LittleIntermediate");
        pinkyDistal       = pinkyDistal       ? pinkyDistal       : FindDeepChild(transform, "XRHand_LittleDistal");
    }

    private static Transform FindDeepChild(Transform parent, string name)
    {
        foreach (Transform child in parent)
        {
            if (child.name == name) return child;
            Transform result = FindDeepChild(child, name);
            if (result != null) return result;
        }
        return null;
    }
}
