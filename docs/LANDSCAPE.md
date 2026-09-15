# Competitive landscape

Who does what for each of the three tracking problems, open source versus
closed. Details and sources in [RESEARCH.md](RESEARCH.md).

## Joints (player pose)

| Open source | Numbers | License |
|---|---|---|
| ViTPose-G | 81.1 COCO AP, still the top public number | Apache 2.0 |
| RTMPose / RTMW | 77 AP; RTMW gives 133 points incl. feet; runs on CPU | Apache 2.0 |
| RF-DETR Keypoint | 71.8 AP at 9.7 ms on T4, single-stage | Apache 2.0 |
| YOLO26-pose | 71.6 AP at 12.2 ms | AGPL |
| Sapiens2 (Meta) | 308 points, no COCO number | custom, commercial OK |
| SAM 3D Body / Fast SAM 3D Body | 3D mesh with hands and feet, per frame | SAM license / MIT code |
| TRAM, WHAM | 3D world-frame trajectory from a moving camera | MIT |

| Closed / hosted | What it is |
|---|---|
| VLM Run gateway | serves ViTPose-Plus (what vision-demos uses) |
| fal.ai | DWPose video, SAM 3D Body stills |
| Roboflow platform | RF-DETR Keypoint hosted training and inference |
| Hawk-Eye SkeleTRACK | 29 body + 7 racket points, pro tour only, 18 cameras |
| Sportsbox AI | monocular 3D kinematics, golf only |
| Tennis AI, OnCourtAI, UNFORCE | phone apps, MediaPipe-class pose |

**Verdict:** fully commoditized. Open beats closed. The only closed capability
you cannot replicate is Hawk-Eye's racket tracking, which needs a camera array.

## Court

| Open source | Numbers | License |
|---|---|---|
| TennisCourtDetector | 8,841 frames, 14 points, 1.8 px median error | none stated |
| Tennis-Vision repo | per-frame court re-detection, tolerates broadcast pan | MIT |
| TVCalib (soccer) | full camera calibration incl. lens distortion | MIT |
| PnLCalib (soccer) | best accuracy on SoccerNet | GPL-2.0 |
| RF-DETR Keypoint / YOLO-pose | train your own 14-point skeleton | Apache / AGPL |

| Closed | What it is |
|---|---|
| SwingVision | court lines as 3D reference, on-device, fixed fence mount required |
| UNFORCE | 99% hard court, 97% clay; lists handheld drift as a known failure |
| Baseline Vision, PlayReplay, Zenniz, Bolt6, Hawk-Eye | fixed mount, calibrated once |

**Verdict:** open is fine for broadcast footage. Nobody, open or closed,
publishes a solution for a handheld camera. Both single-phone products require
a fixed mount. This is an open problem.

## Ball

| Open source | Numbers | Camera | License |
|---|---|---|---|
| WASB (NTT, 2023) | tennis F1 95.6 | trained on sets with pans and zooms | MIT |
| TrackNetV3 / V4 | 98+ F1 on badminton | static only | MIT |
| BlurBall (2026) | blur-aware, table tennis F1 96.5 | static | MIT |
| RF-DETR / YOLO fine-tune | worse on blur; mAP50 ~60 on small sets | any | Apache / AGPL |
| RacketVision dataset | 150k tennis frames with ball labels | broadcast | MIT |

| Closed | What it is |
|---|---|
| TrackNetV5 | best published numbers, weights withheld, proprietary |
| SwingVision | patented learned 2D-to-3D regression, 60 fps on iPhone, fixed mount |
| UNFORCE | uses open TrackNet, fixed mount, publishes error bars |
| Baseline Vision | 2 cameras in one net-post unit, extrapolates to bounce |
| PlayReplay, Zenniz, Bolt6, Hawk-Eye | 4 to 18 cameras, millimeter accuracy; Zenniz adds 30 microphones |

**Verdict:** single-camera ball tracking is the moat. Every certified
line-calling product uses 2 or more fixed cameras. The two single-phone
products either patented a model or use open TrackNet with a fixed mount. The
best open weights are three years old and nothing released has beaten them.

## End-to-end systems

| Open | Notes | License |
|---|---|---|
| HarshTomar1234/Tennis-Vision | honest metrics, event recall 72%, per-frame court | MIT |
| yo-WASSUP/Good-Tennis | static camera, 4-click court, Apache | Apache 2.0 |
| abdullahtarek/tennis_analysis | 887 stars, tutorial, court from frame 0 | none |
| rondo-labs/Padex (padel) | most productised, 15-class shot classifier | GPL-3.0 |

| Closed | Notes |
|---|---|
| SwingVision | $96 to $300 per year, iPhone only, real-time |
| UNFORCE | €19 per month, iOS and Android, post-hoc |
| CourtCheck | student web product, no license |
| PlaySight, Wingfield | club installs, ~$5k plus subscription |
| NVIDIA Nemotron Tennis (Sep 2026) | 31B video-QA model, not a tracker |

## Where that leaves this project

- Joints are free.
- Court on a fixed camera is free.
- Court tracking from a handheld phone, and ball tracking that survives it,
  are unsolved in the open. Those are exactly the constraints in
  [PLAN.md](PLAN.md).
- UNFORCE proves the rest of the stack reaches product quality with open
  components.
