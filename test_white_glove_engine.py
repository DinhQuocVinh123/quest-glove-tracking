import cv2
import numpy as np
import torch
import time
from mmpose.apis import inference_topdown, init_model

CONFIG = '/Users/taminhtri/VR/mmpose/configs/hand_2d_keypoint/rtmpose/hand5/rtmpose-m_8xb256-210e_hand5-256x256.py'
CHECKPOINT = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.pth'

print("Initializing RTMPose-m Top-Down Hand Model on MPS...")
t0 = time.time()
model = init_model(CONFIG, CHECKPOINT, device='mps')
print(f"Model loaded in {time.time() - t0:.2f}s")

# Test on live snapshot
img = cv2.imread('/Users/taminhtri/VR/live_snapshot.jpg')
h, w = img.shape[:2]
print(f"Loaded image: {w}x{h}")

# Test multiple candidate boxes
candidates = [
    np.array([[w * 0.4, h * 0.4, w * 0.8, h * 0.9]]), # Center-right
    np.array([[w * 0.3, h * 0.2, w * 0.7, h * 0.7]]), # Center
]

for i, bbox in enumerate(candidates):
    t_start = time.time()
    results = inference_topdown(model, img, bboxes=bbox)
    torch.mps.synchronize()
    inf_time = (time.time() - t_start) * 1000
    
    kpts = results[0].pred_instances.keypoints[0]
    scores = results[0].pred_instances.keypoint_scores[0]
    
    # Palm knuckles: 0 (wrist), 1 (thumb), 5 (index), 9 (mid), 13 (ring), 17 (pinky)
    core_scores = scores[[0, 1, 5, 9, 13, 17]]
    print(f"Candidate {i}: Latency={inf_time:.1f}ms, Mean score={np.mean(scores):.3f}, Knuckle core={np.mean(core_scores):.3f}")

print("Verification complete!")
