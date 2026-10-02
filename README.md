# Quest 3 Haptic Glove Hand Tracking

A **drop-in module** that adds gloved-hand joint tracking to any Unity project for
Quest 3 — using only the headset's own passthrough camera. No external webcam needed.

Quest's built-in hand tracking fails once you wear a haptic glove. This restores it:
the headset streams passthrough frames to a PC, the PC runs a hand-pose network
(RTMPose), and sends the thumb + index keypoints (pixel coordinates) back; Unity rebuilds
the virtual hand from those image points.

## How it works

```
         QUEST 3                                PC (needs an NVIDIA GPU)
   ┌─────────────────────┐                  ┌──────────────────────────┐
   │ GloveLiveStreamer   │   JPEG over TCP  │ run_glove_quest_stream   │
   │ (passthrough camera)├─────────────────>│  • locate the hand       │
   │                     │     port 5007    │  • run RTMPose           │
   │                     │                  │  • reject bad detections │
   │ FingerUDPReceiver   │ 9 px pts over UDP│                          │
   │ (drives hand bones) │<─────────────────┤                          │
   └─────────────────────┘     port 5005    └──────────────────────────┘
                                   ▲
                    the headset DISCOVERS the PC via a beacon
                    broadcast on UDP 5008 — no IP to configure
```

Auto-discovery is deliberate: switch networks, switch hotspots, get a new IP — nothing
to edit and no rebuild. It runs only at connection time and then sits outside the data
path, so it **adds no per-frame latency**.

---

# Adding this to a Unity project

### Prerequisites

- **Quest 3 or 3S** running Horizon OS **v74 or newer** (the passthrough camera API
  does not exist before that)
- A Unity project with the **Meta XR SDK** and hand tracking enabled
- A **hand model** whose bones follow the `XRHand_*` naming convention
  (e.g. `XRHand_IndexProximal`, `XRHand_ThumbMetacarpal`) — the hand model shipped
  with the Meta XR SDK already matches
- A PC with an **NVIDIA GPU**, on the same network as the headset

### Step 1 — Copy the C# files

Copy these from `unity/` into your project's `Assets/Scripts/`:

| File | Role |
|---|---|
| `GloveLiveStreamer.cs` | Streams passthrough camera frames to the PC |
| `FingerUDPReceiver.cs` | Receives the 9 image points (wrist, thumb, index) and drives the hand |
| `ImageHandSolver.cs` | Fits wrist + thumb + index to the image points once per camera frame, smooths between frames (added automatically by `FingerUDPReceiver`) |
| `HandImageFit.cs` | The solver itself: each point becomes a ray from the camera at capture time; solves wrist pose + 8 finger angles |
| `FingerChainFitter.cs` | Finger kinematics (joint axes, limits) used by the solver |
| `HandFingerRig.cs` | Finds the finger bones by their `XRHand_*` names |

Optional:
- `GloveDatasetCollector.cs` — capture training data.
- `HandPoseRecorder.cs` — logs Quest's tracked bare hand; used to learn natural finger poses.
- `Editor/HandImageFitBench.cs` — offline replay/tests of the solver (goes in `Assets/Editor/`).

### Step 2 — Request camera permission

Add to `Assets/Plugins/Android/AndroidManifest.xml`:

```xml
<manifest ... xmlns:horizonos="http://schemas.horizonos/sdk">
  <horizonos:uses-horizonos-sdk horizonos:minSdkVersion="74" ... />
  <uses-permission android:name="horizonos.permission.HEADSET_CAMERA" />
```

Without this the camera will never start.

### Step 3 — Wire up the scene

**a) The streamer:** create an empty GameObject, add `GloveLiveStreamer`, and drag your
running `PassthroughCameraAccess` component into the `Passthrough Camera` field.

Leave `Use Auto Discovery` ticked — the IP field below it is then unused.

**b) The virtual hand:** add `FingerUDPReceiver` to the GameObject holding your hand
model (`HandFingerRig` is added automatically). If that object has a `Hand Visual`
component, drag it into the matching field so this script doesn't fight Quest's own
hand tracking.

> **Important:** the hand object should be a **child of `CenterEyeAnchor`**. The camera
> is head-mounted and moves with your head, so coordinates must be head-relative.

`HandFingerRig` locates bones by name, so there is usually nothing else to assign.

### Step 4 — Set up the Python environment

See `requirements.txt` — **follow the install order given there**. A plain
`pip install -r requirements.txt` will not work; the OpenMMLab packages are strict
about versions and install order.

Then clone mmpose **next to** the `.py` files:

```bash
git clone https://github.com/open-mmlab/mmpose.git
```

### Step 5 — Run

```bash
# Start the PC first, then launch the app on the headset:
python run_glove_quest_stream.py --record
```

It uses `checkpoints/rtmpose_glove_pinch.pth` (see *Fine-tuning for pinch* below) if present,
otherwise the stock model. `--original` forces the stock model, `--checkpoint <file>` any other.

A debug window shows the detected hand skeleton plus diagnostics (FPS, confidence,
why a frame was rejected, and so on). The base model **downloads itself** on first run.

**To find out why tracking drops out,** add `--record`: every processed frame is saved
as a JPEG to `recordings/<timestamp>/`, with `log.csv` noting per frame whether a hand
was found and, if not, which check rejected it. You can then replay the frames offline
to test a fix without putting the headset back on. Recordings are large (~200 MB per
minute) and are git-ignored. Frames are written on a background thread, so recording no
longer slows tracking down.

**Speed:** by default the script skips mmpose's flip test (the model would otherwise run
twice per frame) and the back-of-hand tile detector, which Unity no longer uses. Measured
with recorded frames streamed at the headset's 18 fps: 9.9 → 16.9 processed frames/s,
send-to-result latency 119 → 65 ms. `--flip` and `--dorsal` turn them back on.

**If it won't connect:** the headset and PC must be on the same network, and that
network must allow devices to talk to each other directly. Corporate networks often
block this — a phone hotspot is the most reliable option.

---

## Protocol (if you want to replace either half)

**Headset → PC, TCP port 5007:** repeating `[4-byte big-endian length][payload]`. The payload
is a JPEG, optionally preceded by a 24-byte header `GLV1 | uint32 frame id | float hand x | float
hand y | float hand size | uint8 valid | 3 pad` (big-endian): where the Quest-tracked glove wrist
projects into this image. Python searches there first and rejects detections far from it (which
stops it locking onto the bare left hand), and echoes the frame id back as `fid` so Unity can use
the camera pose at capture time.

**PC → headset, UDP port 5005:** comma-separated `key:value` text

| Key | Meaning |
|---|---|
| `valid` | `1` = data is trustworthy, `0` = return the virtual hand to its rest pose |
| `fid` | Frame id from the image header this result belongs to |
| `pts` | 21 points as `x\|y;x\|y;...`, origin at the wrist, scaled by palm length, `+y` is up |
| `pinch` | 0..1, distance between thumb and index tips |
| `thumb0/1/2`, `index0/1/2` | Per-joint bend angles (base / middle / tip) |

**Point order:** `0` wrist, `1-4` thumb, `5-8` index, `9-12` middle, `13-16` ring,
`17-20` little. Each finger runs from its base joint out to the fingertip.

---

## The Python files

**Main pipeline:**

| File | Role |
|---|---|
| `run_glove_quest_stream.py` | **The main script.** Headset camera → PC → Unity |
| `export_onnx.py` | Exports a checkpoint to ONNX (normalisation + SimCC decoding built in) for running the model on the headset; `--check recordings/<ts>` compares against mmpose |
| `run_glove_to_unity.py` | Webcam variant, same pipeline |
| `run_white_haptics_glove.py` | Local preview on the PC, no headset required |
| `dataset_receiver.py` | Receives training samples from `GloveDatasetCollector.cs` |

**Labelling and fine-tuning (current):**

| File | Role |
|---|---|
| `auto_label_glove.py` | Auto-labels a `--record` session: keeps good model frames, tracks points through short dropouts with two-way optical flow, queues the rest for review → `labels/auto_labels.jsonl` |
| `label_glove_frames.py` | Labelling tool (drag pre-filled points; per-point seen / estimated / unknown). `--review-auto` = frames the auto-labeller could not do → `labels/glove_labels.jsonl` |
| `finetune_pinch.py` | Fine-tunes the stock RTMPose on those labels, with the same crop/padding as at runtime; reports before/after on a time-split hold-out |

**Older dataset and training tools:**
`manual_label_quest.py`, `review_quest_import.py`, `review_dataset.py`,
`import_quest_dataset.py`, `run_split_screen_finetune.py`, `finetune_glove.py`

The remaining `run_*.py` files are older experiments, kept for reference. They are
**not part of the current pipeline**.

> All `.py` files must stay **in the same directory** — they locate `mmpose/` and
> `dataset/` relative to their own location.

---

## Fine-tuning for pinch

The first fine-tune (`finetune_glove.py`, 170 samples, `rtmpose_glove_finetuned.pth`) tracked
**worse** than the stock model, which is why `--original` used to be the recommended flag.

The second (`finetune_pinch.py` → `rtmpose_glove_pinch.pth`) works. Differences: training crops
match runtime exactly (padding, hint box); ~270 frames the stock model already got right are kept
so it does not forget; unknown points are ignored and estimated ones down-weighted; the
backbone learns 4x slower than the head. Data: one 4-minute free-pinch session, 30
hand-corrected hard frames. Result on later sessions with a different seat/background:
dropouts while the hand is in view 12.5% → 0–3%, except a hand resting low at the keyboard
(not in the training data yet).

```bash
python auto_label_glove.py recordings/<session>
python label_glove_frames.py --review-auto --count 30
python finetune_pinch.py            # close run_glove_quest_stream.py first (2 GB GPU)
```

Always check on a session that was **not** used for training.

### Automatic labels from colour tape (no hand labelling)

Wrap three rings of matte tape around each finger, centred on the joints: **orange** on the
thumb (keypoints 2, 3, 4) and **blue** on the index finger (6, 7, 8). The wrist and the two
finger bases (0, 1, 5) are left untaped. `color_label_glove.py` finds the rings with an HSV
threshold, groups neighbouring pieces starting from the fingertip the model predicted (or from
the previous frame's rings when the model lost the hand), and orders them from base to tip. The
untaped points are copied from the model at weight 0.6, only when the model agrees with the tape.
The tip ring sits ~1 cm short of the real fingertip, so the tip label is extended along the finger
by a ratio measured from the data.

```bash
python run_glove_quest_stream.py --record                 # pinch for a few minutes
python color_label_glove.py recordings/<session>          # -> labels/color_labels.jsonl + labels/color_sheets/
python finetune_pinch.py --compare checkpoints/rtmpose_glove_pinch.pth --out checkpoints/rtmpose_glove_pinch_color.pth
```

First session (new glove, 821 frames, ~1.5 min of pinching): 803 frames labelled, including 111
frames where the model had lost the hand. On held-out 5-second chunks of the same session, fingertips
correct: 87% → 99% (colour-labelled frames), 69% → 100% (hard frames); median error 0.068 → 0.019
hand sizes. A model trained this way relies on the tape, so keep it on the glove.

## Lessons learned (so you don't repeat them)

- **The headset sends faster than the PC can process.** TCP never drops data — it
  queues. Stale frames pile up and latency **grows over time**, reaching 1–2 seconds.
  `LatestFrameReader` always discards the backlog and processes only the newest frame.

- **The model has to be told where to look.** RTMPose is top-down: it does not scan the
  image, it is handed a box. Searching only a few fixed regions means the hand vanishes
  the moment you move it to the edge of the frame. The current code sweeps a grid
  covering the whole frame, a few cells per frame so the frame rate doesn't collapse.

- **Filter on shape, not just on score.** The model has never seen this glove, so its
  confidence always sits near the threshold and a cluttered background pushes it under.
  Checking whether the result actually *looks* like a hand (bone lengths, total finger
  length, sudden jumps between frames) is far more reliable.

- **Never threshold on absolute pixel distances.** With a head-mounted camera the hand
  can be right in front of your face and seen edge-on, which collapses the joints on top
  of each other in 2D. Pixel thresholds written for a desktop webcam will reject exactly
  the good frames.

- **Search boxes need several sizes, not one.** If every box is smaller than the hand,
  each one contains only part of it — and a top-down model will still cram all 21 joints
  into that fragment, collapsing them into a cluster while reporting *high* confidence.
  Telltale symptom: "palm length = 0 px" while the hand fills half the frame.

- **Let search boxes extend past the image edge.** The head-mounted camera often sees
  the hand at the bottom of the frame with the wrist cut off. Clamping boxes to the image
  squashes them so they no longer contain the whole hand, and the model collapses the
  joints into a tiny cluster (rejected as "palm too small") even though the hand is
  clearly visible. The image is now padded with gray (`EDGE_PAD_RATIO`) so boxes can
  overhang it; replaying a real 1030-frame recording raised the share of frames with a
  tracked hand from 85% to 89%, with no extra false detections when no hand was in view.

- **Also check the hand's size relative to the frame.** Every other check compares the
  skeleton against itself, so a perfectly proportioned *tiny* hand passes all of them.
  That lets the tracker lock onto some small detail (a module on the glove, a mark on a
  monitor) and stay locked, since the tracked box then keeps following it. The camera is
  head-mounted and the hand is attached to the wearer's arm, so there is a hard floor on
  how small it can plausibly appear.

- **Hold the last good pose for a moment before giving up.** On cluttered backgrounds the
  model's confidence fluctuates and dips below threshold for a frame or two at a time.
  Declaring "invalid" instantly makes the virtual hand flicker between the real pose and
  the rest pose, which looks far worse than briefly holding a slightly stale pose. Keep
  the hold short so a genuine loss still falls back.

- **Aim each bone along the backbone; don't rotate about a fixed axis.** Real fingers
  bend in arbitrary planes. Rotating each joint about one preselected axis can never
  reproduce the shape, no matter how accurate the angle.

- **Pinch only closes if both fingers share a plane.** Keeping each bone's rest-pose
  depth leaves the thumb permanently angled toward the viewer, so it can never meet the
  index finger (`Preserve Rest Depth` must be OFF). *Superseded by the next point when
  `Use Anatomical Fit` is on.*

- **Flattening 2D directions breaks as soon as a finger points toward or away from the
  camera.** A fist seen from above: the index finger's first bone points *away* from the
  eye, shows up as a short segment pointing up in the image, and the flattened virtual
  finger sticks straight up. `FingerChainFitter` instead solves for the **joint angles**
  whose projection matches the 2D points, with each joint only allowed to bend the way a
  real one does (toward the palm, within limits). That constraint is what recovers the
  missing depth. Two traps found while building it: the 2D image often fits two poses
  almost equally well (finger straight and pointing away vs. curled into the palm), so
  the solver also restarts from a few seed poses; and a strong "stay close to last frame"
  term locks in a wrong first answer — keep it weak and smooth the 2D points instead.
  *(Superseded:)* wrist and palm orientation used to come from Quest hand tracking — but
  with the glove on, Quest's wrist shakes 4–10 cm / 30–50°. `ImageHandSolver` now solves
  the wrist **from the image too**: every keypoint is a ray from the camera at capture
  time, and one solve per camera frame fits wrist pose + thumb/index angles to those rays.
  Quest only supplies a weak depth hint. Two lessons: (1) natural finger shapes come from a
  pose prior learned on the user's **bare** hand (which Quest tracks well), not hand-tuned
  rules; (2) a colour cue on the back of the hand (tan hexagon tiles) was tried and
  disabled — it latched onto same-coloured objects (desk, boxes, mouse) and bent the
  virtual index finger when reaching far.

- **The middle, ring and little fingers are occluded by the hand itself during a pinch**,
  and the model guesses wildly there. The Python side forces them into a closed pose
  while pinching rather than trusting the model.
