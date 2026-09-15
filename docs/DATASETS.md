# Datasets

Public datasets usable for tennis and racket-sport video analysis, as of
September 2026. Grouped by what they label. Broadcast-derived sets have open
annotations but broadcaster-owned footage: fine for training, not for
redistribution. Sets with clean self-recorded or CC-licensed video are marked.

## Tier 1: use these first

| Dataset | Labels | Size | Camera | License | Link |
|---|---|---|---|---|---|
| **RacketVision** (AAAI 2026) | ball xy + visibility, racket bbox + 5 keypoints, tennis / badminton / table tennis | 435k frames, 150k tennis, 64k ball-labelled | broadcast | MIT | [HF](https://huggingface.co/datasets/linfeng302/RacketVision), [GitHub](https://github.com/OrcustD/RacketVision) |
| **TennisCourtDetector** | 14 court keypoints | 8,841 frames, 1280x720, hard/clay/grass | broadcast | none stated (HF mirror says MIT) | [GitHub](https://github.com/yastrebksv/TennisCourtDetector), [HF mirror](https://huggingface.co/datasets/Gholamreza/tennis_court_keypoints_dataset) |
| **TrackNet tennis** | ball xy, visibility, hit / bounce event | 20k frames (+16k extra across 9 courts) | broadcast | none | [GitLab](https://gitlab.nol.cs.nycu.edu.tw/open-source/TrackNet), [PyTorch re-upload](https://github.com/yastrebksv/TrackNet) |
| **CalTennis** (Caltech / Perona, Jun 2026) | multi-view synced video, camera intrinsics/extrinsics, SMPL-X pose estimates. No manual GT, no ball, no court. | 11M frames, 51 h, 40 players, 2-6 iPhones at 60 Hz | fixed tripods, non-broadcast | CC BY-NC 4.0 on HF, CC BY 4.0 in paper (conflict) | [HF](https://huggingface.co/datasets/demalenk/caltennis), [paper](https://arxiv.org/abs/2606.20542) |
| **TennisExpert / TennisVL** (Mar 2026) | shot sequences (type, technique, direction, outcome, bounce), player boxes, ball position, court keypoints, scoreboard, commentary | 202 Slam matches, 472 h, 40k rallies, 162k shots | broadcast | CC BY 4.0 (paper) | [GitHub](https://github.com/LZYAndy/TennisExpert), [paper](https://arxiv.org/abs/2603.13397) |

## Tennis: ball

| Dataset | Labels | Size | License | Link |
|---|---|---|---|---|
| WASB-SBDT bundle | repackaged TrackNet tennis + 4 other sports | same | per source; code MIT | [GitHub](https://github.com/nttcom/WASB-SBDT/blob/main/GET_STARTED.md) |
| Multi-View Tennis Ball (IEEE DataPort, Oct 2025) | per-frame ball xy, drone + court camera, day/night | 2.9 GB | IEEE subscription | [DataPort](https://ieee-dataport.org/documents/multi-view-tennis-ball-dataset-trajectory-estimation-drone-and-court-cameras-annotated) |
| Roboflow: RowerUp tennis ball | bbox, also racket, person | 3,243 images | CC BY 4.0 | [Universe](https://universe.roboflow.com/rowerup/tennis-ball-8bsra) |
| Roboflow: cs356 tennis balls | bbox | 1,241 | CC BY 4.0 | [Universe](https://universe.roboflow.com/cs356/tennis-balls-pww4x) |
| Roboflow: Viren Dhanwani | bbox, used in the popular tutorials | ~578 | CC BY 4.0 | [Universe](https://universe.roboflow.com/viren-dhanwani/tennis-ball-detection) |
| Kaggle tenis_backview | 10 back-view matches, player + ball + court lines | unknown | unknown | [Kaggle](https://www.kaggle.com/datasets/gastonarielfrancois/tenis-backview) |
| YOLO-Net tennis (Jul 2026) | ball, racquet, player bbox | 6,648 images at 1280 px | see Kaggle | [Kaggle](https://www.kaggle.com/datasets/taowang1123/yolo-net) |

## Tennis: court

| Dataset | Labels | Size | License | Link |
|---|---|---|---|---|
| TennisCourtDetector | 14 points | 8,841 | none | above |
| Roboflow: desertsloth1 | keypoints | ~2,500 | CC BY 4.0 | [Universe](https://universe.roboflow.com/desertsloth1/tennis-court-detection-onesd) |
| Roboflow: Pei Ling | keypoints | 535 | CC BY 4.0 | [Universe](https://universe.roboflow.com/pei-ling/tennis-court-keypoint-detection) |
| Roboflow: EpuResearchSpace | keypoints | 403 | CC BY 4.0 | [Universe](https://universe.roboflow.com/epuresearchspace/tennis_court-bkajz) |
| Roboflow: several smaller | keypoints | 100-200 each | CC BY 4.0 | [search](https://universe.roboflow.com/search?q=class:%22tennis+court%22) |

All broadcast. No handheld court dataset exists. Self-label a few hundred
frames from your own footage.

## Tennis: events, shot type, rallies (broadcast)

| Dataset | Labels | Size | License | Link |
|---|---|---|---|---|
| **F3Set** (ICLR 2025) | fine-grained per-hit events: hitter side, FH/BH, serve/return, direction, outcome. 1,000+ compound types | 7,445 train / 2,271 test rallies | none; videos via YouTube IDs | [GitHub](https://github.com/F3Set/F3Set) |
| **E2E-Spot Tennis** (ECCV 2022) | 6 frame-accurate events: near/far serve contact, swing contact, bounce | 3,345 clips, 28 matches, 33,791 events | none; videos via YouTube IDs | [GitHub](https://github.com/jhong93/spot) |
| TenniSet (2017) | 11 temporal event classes + commentary captions | 5 Olympic matches, 746 points | MIT | [GitHub](https://github.com/HaydenFaulkner/Tennis) |
| TennisTV (Sep 2025) | 2,527 QA across 8 tasks for VLM eval | 1,298 videos | not stated | [ModelScope](https://modelscope.cn/datasets/FDUBay/TennisTV) |
| TennisVAR / TRACE (Aug 2026) | 41k stroke events, 25k tactical units, QA | 11,189 rallies | **not released** | [project](https://whynotgit2025.github.io/TennisVAR/) |

## Tennis: pose and stroke type (non-broadcast)

| Dataset | Labels | Size | Camera | License | Link |
|---|---|---|---|---|---|
| CalTennis | see Tier 1 | | | | |
| THETIS (2013) | 12 stroke classes, RGB + depth + 2D/3D skeleton | 1,980 clips, 55 subjects, indoor, no ball | Kinect | research | [GitHub](https://github.com/THETIS-dataset/dataset) |
| Tennis Player Actions (2024) | 4 actions, 18-joint COCO keypoints | 2,000 images | self-recorded | CC BY 4.0 | [Mendeley](https://data.mendeley.com/datasets/nv3rpsxhhk/1) |
| Tennis Shot Side + Top View (2024) | straight vs cross-court, actual ball landing positions | 472 clips | fixed side + overhead | CC BY 4.0 | [Mendeley](https://data.mendeley.com/datasets/75m8vz7jr2/4) |
| SportsPose (2023) | 3D poses, 5 sports incl. tennis | 176k poses, 1.5M frames | 7-camera lab | academic, request | [GitHub](https://github.com/ChristianIngwersen/SportsPose) |
| AthletePose3D (CVPR 2025) | 3D pose, 12 sports, fast motion | 1.3M frames | multi-cam | academic | [GitHub](https://github.com/calvinyeungck/AthletePose3D) |

## Badminton

| Dataset | Labels | Size | License | Link |
|---|---|---|---|---|
| TrackNetV2 Shuttlecock | shuttle xy + visibility | 78k frames, 26 matches | none | [HackMD](https://hackmd.io/@TUIK/rJkRW54cU) |
| ShuttleSet / ShuttleSet22 | 18 shot types, landing xy, player positions per stroke | 36k strokes, 44 matches | MIT / CC BY 4.0 | [GitHub](https://github.com/wywyWang/CoachAI-Projects) |
| BadmintonDB | 10 shot types, rally outcomes | 9,671 strokes | none | [GitHub](https://github.com/kwban/badminton-db) |
| VideoBadminton | 18 stroke classes, racket trajectories | 7,822 clips at 60 fps | CC BY-NC-SA 4.0 | [paper](https://arxiv.org/abs/2403.12385) |
| FineBadminton (MM 2025) | 3-level labels incl. quality scores | 33k strokes, 120 matches | unclear | [project](https://ilearn-lab.github.io/MM25-FineBadminton/) |
| BFMD (Mar 2026) | dense full-match: hits, trajectories, pose, shot types | 19 matches, 16,751 hits | check paper | [paper](https://arxiv.org/abs/2603.25533) |
| CourtKeyNet data | 4 court corners | merged Roboflow sets | MIT | [GitHub](https://github.com/adithyanraj03/Paper_09_Data-Set_CourtKeyNet) |
| Roboflow: shuttlecock | bbox | 8.1k | CC BY 4.0 | [Universe](https://universe.roboflow.com/mathieu-cartron/shuttlecock-cqzy3) |

## Table tennis

| Dataset | Labels | Size | License | Link |
|---|---|---|---|---|
| OpenTTGames / TTNet | ball xy, bounce / net events, segmentation, 120 fps | 12 videos, 4,271 events | CC BY-NC-SA 4.0 | [OSAI](https://lab.osai.ai/) |
| Extended OpenTTGames (Dec 2025) | + 1,457 stroke labels, posture, 282 rally outcomes | same | CC BY-NC-SA 4.0 | [GitHub](https://github.com/moamal01/table_tennis_data) |
| BlurBall (2025) | ball xy + blur orientation + blur length, camera calibration | 64k frames, 26 games | CC BY-SA 4.0 | [site](https://cogsys-tuebingen.github.io/blurball/) |
| TT4D (May 2026) | 3D ball, spin, 3D meshes, calibration | 140+ h, 211k points | CC BY 4.0, **not yet hosted** | [paper](https://arxiv.org/abs/2605.01234) |
| P2ANet | dense action labels | 2,721 clips | MIT code, Baidu only | [GitHub](https://github.com/Fred1991/P2ANET) |
| TTStroke-21 / MediaEval | 20 stroke classes | 1,155 clips | restrictive agreement | [paper](https://arxiv.org/abs/2301.13576) |

## Padel and pickleball

| Dataset | Labels | Size | License | Link |
|---|---|---|---|---|
| PadelTracker100 (2026) | ball trajectory, player positions, ViTPose-L poses, 6 shot events | 100k frames, 2 WPT matches | CC BY 4.0 | [Zenodo](https://zenodo.org/records/14653706) |
| Roboflow: padel ball | bbox | 9.2k | CC BY 4.0 | [Universe](https://universe.roboflow.com/padel-ll7pp/padel-dataset) |
| Roboflow: padel court keypoints | keypoints | 656 | CC BY 4.0 | [Universe](https://universe.roboflow.com/joshs-workspace-p1aa0/padel-court-detection) |
| Roboflow: pickleball, several | bbox / keypoints | 100-340 each | CC BY 4.0 | [search](https://universe.roboflow.com/search?q=class%3Apickleball+ball) |

No academic pickleball or padel video dataset exists beyond PadelTracker100.

## What does not exist

- A handheld or phone-recorded tennis dataset with ball labels. Every ball set
  is broadcast. CalTennis is phone-recorded but has no ball or court labels
  and the phones are on tripods.
- A tennis dataset with ground-truth 3D ball positions from a single camera.
  TT4D does this for table tennis.
- Anything labelled "TennisMPII". MPII Human Pose has a tennis activity
  category, that is all.

## Suggested combination for this project

1. Ball detector: RacketVision tennis split (150k) + TrackNet tennis (20k),
   both broadcast, then a few thousand self-labelled handheld frames labelled
   at the blur-streak center per BlurBall.
2. Court: TennisCourtDetector (8,841) + the ~4k Roboflow sets, with heavy
   perspective augmentation, then a few hundred handheld frames.
3. Stroke classification: F3Set or E2E-Spot for events, BST pretrained
   weights on TenniSet as a starting point.
4. Pose evaluation: CalTennis as a benchmark for 3D pose on real tennis
   motion, once 3D is on the table.
