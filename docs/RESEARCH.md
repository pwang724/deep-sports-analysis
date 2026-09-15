# Ecosystem research (September 2026)

What is actually state of the art for each piece of the tennis pipeline, and
what to use. Compiled from five parallel literature and repo sweeps. Numbers
are from papers or READMEs; items released after June 2026 are flagged.

## TL;DR

```
                     accuracy SOTA              real-time / practical pick
players (2D)         ViTPose-G  81.1 COCO AP    RF-DETR Keypoint 71.8 AP @ 9.7ms T4, Apache
players (feet)       RTMW-l 70.2 WholeBody AP   same, via rtmlib on CPU
players (3D)         PromptHMR-Vid (non-comm.)  TRAM + WHAM (MIT); depth + foot contact unreliable
court (14 pts)       heatmap CNN, 3.8px error   TennisCourtDetector data + RF-DETR Keypoint or heatmap CNN
ball                 TrackNetV5 (no weights)    WASB (MIT, tennis weights, tolerates camera motion)
object detection     RF-DETR-2XL 60.1 COCO AP   RF-DETR-M 54.7 @ 4.4ms, Apache
```

Is RF-DETR SOTA? Yes for real-time object detection: no shipped, T4-benchmarked
model beats RF-DETR-2XL (60.1 COCO AP) as of 2026-09-15, and RF-DETR beats every
YOLO at equal latency. For keypoints it is the best single-stage real-time
model. It is not the right tool for a 5-10 px tennis ball, where multi-frame
heatmap trackers win every controlled comparison.

## 1. Object detection

| Model | COCO AP | T4 ms | License |
|---|---|---|---|
| RF-DETR-2XL | 60.1 | 17.2 | PML 1.0 (paid) |
| RF-DETR-L | 56.5 | 6.8 | Apache 2.0 |
| RF-DETR-M | 54.7 | 4.4 | Apache 2.0 |
| YOLO26x / l | 57.5 / 55.0 | 11.8 / 6.2 | AGPL |
| DEIMv2-X / EdgeCrafter-X | 57.8 / 57.9 | 13.8 / 12.7 | custom, commercial needs contact |
| RT-DETRv4-X | 57.0 | 12.9 | Apache 2.0 |
| SAM 3.1 | 56.4 zero-shot | ~30 ms H200 | SAM license, not real-time |

- YOLO27 announced, not released. Preview numbers on a faster GPU only.
- RF-DETR has the best small-object AP among real-time models (AP_S 36.1 at M,
  43.2 at 2XL) and the best domain-transfer score (RF100-VL 61.7 at M).
- For fine-tuning on ~1000 images: RF-DETR-S or M. DINOv2 backbone plus
  Objects365 pretraining transfers best.
- Small objects: resolution and tiling beat architecture. SAHI slicing gives
  +5 to +14 AP. Train at 1024+ or on crops.

Sources: rf-detr repo and paper 2511.09554, Ultralytics YOLO26/27 docs, DEIMv2,
EdgeCrafter, RT-DETRv4 repos, SAM 3 repo, SAHI 2202.06934.

## 2. Human pose

### 2D, 17 keypoints
| Model | COCO AP | Notes |
|---|---|---|
| ViTPose-G | 81.1 test-dev | Still the top public number. No 2026 model beat it. |
| ViTPose++-H / L | 79.4 / 78.6 val | What the VLM Run gateway serves (`vitpose-plus-large`) |
| RTMPose-l | 77.3 | 11-25 ms on CPU. Best top-down without a GPU. |
| RF-DETR Keypoint Preview | 71.8 @ 9.7 ms T4 | Best single-stage. Apache. Still "preview", one size. |
| YOLO26x-pose | 71.6 @ 12.2 ms | AGPL |
| Sapiens2 (Meta, Apr 2026) | 82.3 on Meta's own 308-kp set | No COCO number. Commercial-OK custom license. |

### Whole-body, 133 keypoints (has 6 foot points)
RTMW-l 70.2 COCO-WholeBody AP, DWPose-l best foot AP 70.4. Both Apache, both
in `rtmlib` (pip, ONNX, CPU). Use when ankle-only is not enough.

### 3D mesh, moving camera
- Per-frame body+hands+feet: SAM 3D Body (CVPR 2026), Fast SAM 3D Body (MIT
  code, ~7 FPS). Neither solves camera motion; you add SLAM yourself.
- World-frame trajectory: PromptHMR-Vid best (non-commercial). TRAM and WHAM
  are MIT. WHAM outputs foot-contact probabilities.
- CalTennis (Jun 2026, 11M frames, 40 players, multi-phone) benchmarked all of
  them on tennis. Conclusion: joint angles are now accurate, but every model
  fails at depth and foot contact. Get court position from the homography, not
  from 3D pose.

### Hosted
VLM Run gateway serves ViTPose-Plus (used by vision-demos). fal.ai serves
DWPose video and SAM 3D Body stills. Roboflow serves RF-DETR Keypoint via the
platform, not yet in the open `inference` package.

## 3. Ball tracking

Every controlled comparison favors multi-frame heatmap trackers over
single-frame box detectors for a small blurred ball:

| Comparison | Detector | Heatmap tracker |
|---|---|---|
| TrackNetV3 paper, badminton | YOLOv7 F1 68 | TrackNetV3 F1 98.6 |
| WASB paper, tennis | single-frame DeepBall F1 47 | WASB F1 95.6 |
| RacketVision (AAAI 2026), 3 sports | YOLO11, RTMDet | TrackNetV3 wins all |

| Tracker | Tennis F1 | Camera motion | Weights | License |
|---|---|---|---|---|
| WASB (NTT, BMVC 2023) | 95.6 | trained on sets with pans/zooms | yes, 5 sports | MIT |
| TrackNetV3 | best on badminton | needs fixed camera (median background) | yes | MIT-labelled, unclear |
| TrackNetV4 (ICASSP 2025) | 97.5 game-level | frame differencing, assumes static | placeholder links | MIT |
| TrackNetV5 (Dec 2025) | 98.6 | not evaluated | **none released**, proprietary | no |
| TrackNetV6 (ICMR, Jun 2026) | unknown | unknown | unknown | unknown |
| BlurBall (CVPRW 2026) | table tennis F1 96.5 | fixed | yes | MIT |
| RF-DETR-S (Roboflow blog) | mAP50 59.9 on 85 images | none assumed | fine-tune | Apache |

Recommendation for handheld: WASB. It is the only open, weight-released,
multi-sport tracker with published tolerance to camera motion. Stabilize frames
first (feature-tracked homography). Label handheld frames at the blur-streak
center per BlurBall. Fall back to RF-DETR at 1024 px with tiling only if the
ball is large in frame or WASB fails on your footage.

## 4. Court detection

- De-facto dataset: yastrebksv/TennisCourtDetector, 8,841 broadcast frames,
  14 keypoints, no license stated. HF mirror labelled MIT.
- Field standard: heatmap CNN (ResNet50 / EfficientNetV2) on those 14 points,
  3.8 px mean error, then RANSAC homography from the best 4+ points.
- RF-DETR Keypoint can do the same with a custom 14-point skeleton and puts
  court, ball, and players in one Apache network. YOLO26-pose is the AGPL
  version of the same trick (YOCO-Sport, 2026, did it for badminton).
- All datasets are broadcast. Handheld needs heavy perspective augmentation
  plus a few hundred self-labelled frames.
- For a moving camera the tracker matters more than the detector: detect every
  N frames, propagate the homography with Lucas-Kanade tracked corners, refit
  when reprojection error grows or a cut is detected. Tennis-AI-Tracker (Sep
  2026) and the Stanford EE367 smartphone pipeline both do this.
- yastrebksv metrics at 7 px tolerance: heatmap alone 0.936 precision, plus
  line refinement and homography fill-in 0.963, 1.83 px median error. Inference
  is per-frame with no tracking, broadcast views only.
- Soccer field-registration code that transfers: TVCalib (MIT, segment
  reprojection optimizer, handles lens distortion), KpSFR (MIT), No Bells Just
  Whistles / PnLCalib (GPL-2.0, avoid in a closed product). BroadTrack (WACV
  2025) and Theiner CVPRW 2026 model PTZ cameras with a fixed position, which a
  handheld phone violates, so track the full 8-DoF homography instead.
- Practical smoothing recipe from the SoccerNet 2024 winner: per-frame
  estimate, then Savitzky-Golay filter on the camera parameters.

## 5. Datasets worth using

| Dataset | Content | Size | License |
|---|---|---|---|
| RacketVision (AAAI 2026, HF) | ball + racket keypoints, tennis/badminton/TT broadcast | 150k tennis frames | MIT |
| TrackNet tennis | ball xy + visibility + hit/bounce | 20k frames | none |
| TennisCourtDetector | 14 court keypoints | 8,841 frames | none |
| TennisExpert / TennisVL (Mar 2026) | 202 Slam matches, shot sequences, ball, court, commentary | 472 h, 40k rallies | CC BY 4.0 |
| F3Set (ICLR 2025) | fine-grained stroke events | 7k+ rallies, YouTube IDs | none |
| CalTennis (Jun 2026) | multi-phone 60 Hz, SMPL-X estimates, no GT | 11M frames | CC BY-NC (conflict) |
| THETIS | 12 stroke classes, Kinect | 1,980 clips | research |
| PadelTracker100 | ball, ViTPose-L poses, shot events | 100k frames | CC BY 4.0 |

Caveat: almost all are broadcast-derived. Annotations are open, footage rights
are the broadcaster's. No public handheld tennis dataset with ball labels
exists. Expect to self-label a few thousand frames.

## 6. Existing open-source tennis systems

None are production grade. All assume a fixed broadcast camera except
Tennis-AI-Tracker (Sep 2026, YOLO26, 0 stars, unproven). Most reuse
TennisCourtDetector for court and TrackNet or YOLO for ball. Tutorial repo:
abdullahtarek/tennis_analysis. Reference for pipeline style:
jeremyipark/vision-demos.

Commercial single-camera analogue: SwingVision (iPhone, on-device, fixed
elevated baseline position, patents describe single-frame CNN candidates plus
physics-consistent trajectory fitting plus court homography).

## 8. Shot classification and event detection

- **Stroke type from pose + ball: BST** (CVPRW 2026, MIT, pretrained weights
  for TenniSet). Input is 2D pose plus ball trajectory plus court position.
  99.2% on TenniSet 6 classes, 83% on ShuttleSet 25 classes. The ball stream is
  the main gain. Most reusable open classifier for tennis.
- **Hit and bounce spotting: E2E-Spot / F3ED** on the Tennis (3,345 clips,
  6 events, frame-accurate) and F3Set data. BSD-3 code, video via YouTube IDs.
- **Cheap bounce detector:** CatBoost on windowed ball x/y differences
  (TennisProject, no license; RallyVision port is Apache). Exact-frame
  precision 92%, recall 62%; within 1 frame 79% / 86%.
- **Rally boundaries:** audio racket-impact clustering (TennisExpert recipe) or
  a rule-based state machine on ball track (tennisvision repo).
- **Point outcome labels** exist only in F3Set / TennisTV.
- Pose-only stroke classifiers (MediaPipe + transformer) top out around 84% on
  forehand vs backhand. Add the ball.

## 9. Commercial products, what they actually do

| Product | Cameras | Approach |
|---|---|---|
| SwingVision | 1 iPhone, fixed on fence, 60 fps | On-device, Neural Engine. Patent US 11,893,808: learned direct 2D-to-3D regression using court lines as reference, trained against radar/LiDAR. Claims speed within 10%, line calls 97%. |
| UNFORCE (ex AceSense) | 1 phone, upload | CatBoost + MediaPipe pose + TrackNet. Published F1: forehand 0.92, backhand 0.91, serve 0.88. Serve speed median error 6 km/h. Court keypoints 99% hard, 97% clay. Notes handheld homography drift as a failure mode. |
| Baseline Vision | 2 cams in one net-post unit | $1,999, trajectory extrapolation to bounce |
| PlayReplay | 4 cams, net posts | ITF Silver, USTA Pro Circuit 2026 |
| Zenniz | 4 cams + 30 mics | audio-video fusion, 7 mm |
| Bolt6 | 12 cams | ITF Gold, Australian Open |
| Hawk-Eye | 10-18 cams | ITF Gold, 2.6 mm; SkeleTRACK 29 body + 7 racket points |

Only SwingVision and UNFORCE do single-phone ball tracking. Everything with
certified line calling uses 2 or more fixed cameras. UNFORCE's stack is
essentially this repo's plan, which is a useful sanity check on feasibility.

## 7. Post-June-2026 items

RF-DETR Keypoint (Jun 17), CalTennis (Jun 18), TrackNetV6 (Jun 15, no details),
YOLO27 (announced only), Fast SAM 3D Body at ECCV (Sep 11), TennisVAR / TRACE
benchmark (Aug 13, dataset unreleased), TT4D revision (Sep 9), Tennis-AI-Tracker
(Sep 10), FSDC-DETR small-object DETR (Jul, ECCV 2026), YOLO-Net tennis
YOLO11 variant (Jul, PLOS ONE).
