# Pose backend results

Two backends, run on the same frames and scored two ways: against hand-labelled
tennis joints, and against each other on the US Open final broadcast.
Code: `src/dsa/pose/eval_pose.py`, `src/dsa/cloud/bench_detectors.py`, `src/dsa/pose/tracking.py`.

## 1. Ground truth: Tennis Player Actions dataset

1,994 self-recorded images, one player each, ~200 px tall, 17 labelled COCO
joints. RF-DETR runs on the full image; ViTPose runs top-down on a box.

| backend | detected | mean OKS | PCK@0.05 | PCK@0.10 | s/img, L4 fp16 |
|---|---|---|---|---|---|
| RF-DETR Keypoint, full image | 99.4% | 0.773 | 0.847 | 0.967 | 0.038 |
| RF-DETR Keypoint, padded crop of the box | 100% | 0.688 | 0.737 | 0.879 | 0.035 |
| ViTPose-Plus-Huge on labelled box | 100% | 0.824 | 0.906 | 0.990 | 0.058 |
| ViTPose-Plus-Huge on RF-DETR box | 99.4% | 0.823 | 0.903 | 0.989 | |

The crop row is the like-for-like comparison: RF-DETR given the same pixels
ViTPose sees. It scores worse than RF-DETR on the full image, so the
bottom-up model is not being handicapped by resolution; a single body filling
the input is out of its training distribution. Reproduced on an L4 in fp16
to three decimals of the original M2 fp32 run.

OKS is the COCO keypoint similarity, 1.0 is perfect. PCK@t is the fraction of
joints within t of box height from the label.

Per-joint median error, % of box height:

| joint | RF-DETR | ViTPose |
|---|---|---|
| shoulders | 2.0 | 1.9 |
| elbows | 2.0 | 1.5 |
| wrists | 2.3 | 1.4 |
| hips | 3.2 | 3.0 |
| knees | 1.6 | 1.4 |
| ankles | 1.5 | 1.4 |

By action, mean OKS:

| action | RF-DETR | ViTPose |
|---|---|---|
| backhand | 0.768 | 0.798 |
| forehand | 0.746 | 0.808 |
| ready | 0.783 | 0.802 |
| serve | 0.796 | 0.890 |

Findings:
- ViTPose is better by 5 OKS points overall, and the gap is in the arms:
  wrists 1.4% vs 2.3% of height, elbows 1.5% vs 2.0%. Legs and torso are a
  wash. The arm is what stroke analysis reads, so this matters.
- Serve is where ViTPose pulls furthest ahead, 0.89 vs 0.80. Arms overhead is
  the pose RF-DETR handles worst.
- ViTPose on RF-DETR's box scores the same as on the labelled box. Detector
  box quality is not the bottleneck; RF-DETR's boxes are good enough.

## 2. Broadcast: US Open final, 34.5 s rally block at 30 fps

1,034 frames from 320.5 s of the extended highlights. End-on broadcast camera.
Near player ~234 px tall, far player ~109 px.

Coverage after the far-court fix (see below):

| | near player found | far player found | both |
|---|---|---|---|
| frames of 1,034 | 881 | 1,021 | 878 |

The missing near-player frames are mostly frames where the near player is
between points or out of the court region, not detector misses.

Agreement between the two backends, median distance in % of box height, and
mean confidence each model assigns:

| joint | far: dist | far: RF conf | far: VP conf | near: dist | near: RF conf | near: VP conf |
|---|---|---|---|---|---|---|
| shoulders | 1.3 | 0.99 | 0.90 | 1.3 | 0.99 | 0.92 |
| elbows | 2.0 | 0.90 | 0.83 | 1.4 | 0.90 | 0.91 |
| wrists | 3.1 | 0.86 | 0.77 | 3.6 | 0.62 | 0.77 |
| hips | 1.3 | 0.99 | 0.79 | 1.9 | 0.99 | 0.79 |
| knees | 1.6 | 0.94 | 0.85 | 1.2 | 0.98 | 0.92 |
| ankles | 1.2 | 0.91 | 0.77 | 1.0 | 0.98 | 0.87 |
| nose / eyes | 1.4 | 0.80 | 0.88 | 1.8 | 0.05 | 0.62 |

Findings:
- The two models agree within 1 to 2% of body height on everything except
  wrists, which is the joint that moves fastest and blurs most. There is no
  ground truth here, so agreement is a consistency check, not accuracy.
- RF-DETR gives ~0 confidence on the near player's face. The near player
  faces away from the camera, so this is the model being honest about an
  occluded joint. ViTPose still commits at 0.6. Neither is wrong; they differ
  in what confidence means.
- Wrist confidence is the lowest joint for both, on both players. Expect
  racket-arm tracking to need temporal smoothing.

## 3. Player detection, scored on labels

197 broadcast frames from TennisSegmentation (Australian Open, end-on, both
players masked). A player counts as found at IoU >= 0.5 with the label.

| detector | thr | far player | near player | s/img, L4 |
|---|---|---|---|---|
| RF-DETR Keypoint + far-court crop (the fix below), fp32 | 0.40 | 0.401 | 0.903 | 0.096 |
| YOLO11m @1280 fp16 | 0.40 | 0.985 | 1.000 | 0.030 |
| RT-DETRv2-r50 @1280 fp16 | 0.40 | 0.650 | 0.959 | 0.070 |
| **RF-DETR Medium @1152 fp16** | 0.40 | **1.000** | **1.000** | **0.025** |
| RF-DETR Large @1152 fp16 | 0.40 | 0.995 | 1.000 | 0.024 |

Decision: RF-DETR Medium at 1152 px is the detector. The far-court crop
pass below was tuned on one clip and does not transfer: on this set its far
player recall is 40%. Benchmark: `src/dsa/cloud/bench_detectors.py`.

### Open-vocabulary detectors, prompted "tennis player"

Same 197 frames. Question: can one text-prompted model return only the two
players in one shot? Recall is at the lowest threshold that keeps the far
player; boxes/image counts everything returned at that threshold (target: 2).

| detector | thr | far player | near player | boxes/image | s/img, L4 |
|---|---|---|---|---|---|
| RF-DETR Medium, "person" (reference) | 0.4 | 1.000 | 1.000 | 11.6 | 0.020 |
| SAM 3 | 0.6 | 1.000 | 1.000 | 5.4 | 0.217 |
| OWLv2 large | 0.3 | 1.000 | 1.000 | 3.3 | 0.206 |
| Grounding DINO base | 0.1 | 1.000 | 1.000 | 7.7 | 0.143 |
| YOLOE 11l | 0.1 | 0.447 | 0.851 | 1.9 | 0.028 |

No. Every model scores ball kids in the same band as players (SAM 3: kids
0.84-0.90, line judges 0.6-0.75), because on a single frame a crouching kid
in kit on the court is a tennis player by appearance.

Prompting harder does not help either (SAM 3, same frames):

| prompt strategy | thr | far | near | boxes/image |
|---|---|---|---|---|
| "tennis player" | 0.6 | 1.000 | 1.000 | 5.4 |
| "tennis player hitting the ball with a racket" | 0.3 | 0.939 | 1.000 | 3.1 |
| "tennis player" with negatives ball boy / ball kid / line judge / umpire / spectator | 0.6 | 1.000 | 1.000 | 5.4 |

Negatives change nothing: SAM 3 scores the kids lower as "ball boy" than as
"tennis player". A descriptive prompt only trades far-player recall for
fewer extras. The distinction is behavioural and lives across frames, so
player selection has to be decided per track, not per frame. Plan: RF-DETR
boxes, ByteTrack, then a per-track judge (VLM or contrastive CLIP prompts)
with the sport as a string.

### SAM 3 video tracking vs RF-DETR + ByteTrack

Can SAM 3's video mode replace detector plus tracker? Same 34.5 s clip,
1,035 frames at stride 2, three 15 s segments, one L4 per segment. No track
labels exist, so each player is located per frame by a position band that
follows the person across frames (`dsa/pose/track_metrics.py`); "IDs" is how
many distinct track ids that player received, 1 per segment being perfect.

| | RF-DETR Medium + ByteTrack | SAM 3 video, "tennis player" |
|---|---|---|
| near player found | 92.4% | 80.7% |
| near player IDs / changes | 6 / 7 | 6 / 24 |
| far player found | 97.5% | 33.6% |
| far player IDs / changes | 5 / 4 | 6 / 20 |
| boxes per frame | 7.1 | 4.9 |
| seconds per frame | 0.037 | 0.517 |
| GPU cost per minute of video | $0.015 | $0.207 |

SAM 3 loses the far player two frames in three, flips identity 3-5x more
often, and costs 14x. Even with its mask NMS active (`kernels` 0.16.x), a
third of its boxes are mask artifacts spanning to the frame edge. Run in
fp16; an fp32 run was started and abandoned because speed and cost alone
rule it out. Decision: RF-DETR Medium + ByteTrack. The SAM 3 code was
removed; `src/dsa/cloud/bench_tracking.py` remains as the tracker-quality
harness for the chosen pipeline.

## 4. The far-player problem and the crop fix it replaced

RF-DETR at default settings found the far player in 88 of 1,034 frames. The
model shrinks a 1080p frame to its inference resolution, and a 109 px person
becomes too small. Tested fixes on one rally frame:

| approach | far player confidence | extra cost |
|---|---|---|
| threshold 0.4, default | not detected | |
| threshold 0.1, default | 0.14 | many false positives |
| threshold 0.2, inference at 1152 px | 0.24 | 3x slower |
| 2x upscaled crop of the far half, threshold 0.2 | **0.69** | +1 pass on a half-frame |

The far-court crop pass is now default in `joints.py`. Coverage went from
88 to 1,021 frames. Line judges and ball kids picked up at that scale sit
outside the court region and are dropped by the on-court filter.

## 5. Speed on an M2 MacBook, 16 GB, MPS

Measured uncontended:

| stage | per frame |
|---|---|
| RF-DETR full frame + far crop | ~0.5 s |
| ViTPose-Huge on 2 player boxes | ~0.25 s |
| ViTPose-Huge on a crowd of 8 | ~0.6 s |

The 12-minute highlight clip at 30 fps is ~22k frames, so ~4.5 hours on this
laptop for both backends. The full 3.5 h match would be ~35 hours here and
about 1 hour on a rented GPU.

## Decision

- **Boxes:** RF-DETR Medium at 1152 px, fp16. 100% recall on both players on
  labelled broadcast frames, 20 ms per frame on an L4, Apache licence.
- **Identity:** ByteTrack on those boxes. 6 ids / 7 switches for the near
  player over 34 s across 3 segments; SAM 3 video was worse and 14x the cost.
- **Joints:** ViTPose-Plus-Huge on each tracked box. 0.824 OKS vs 0.773 for
  RF-DETR Keypoint's best mode; the gap is in wrists, elbows and serves.
- **Open:** which tracks are the players. No single-frame model separates a
  ball kid from a player (five models, three prompt strategies); it needs a
  per-track judge with the sport as a string.

## Caveats

- The labelled dataset is close-range and self-recorded. It says who is more
  accurate on tennis poses at ~200 px, not on a 109 px broadcast far player.
  For that we need to label a few hundred broadcast frames ourselves.
- One rally block, one camera. Other end-on framings will need the on-court
  filter's baseline constants retuned or, better, replaced by the court
  detector's homography.
