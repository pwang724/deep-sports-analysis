# Pose backend results

Two backends, run on the same frames and scored two ways: against hand-labelled
tennis joints, and against each other on the US Open final broadcast.
Scripts: `src/eval_gt.py`, `src/joints.py`, `src/compare.py`.

## 1. Ground truth: Tennis Player Actions dataset

1,994 self-recorded images, one player each, ~200 px tall, 17 labelled COCO
joints. RF-DETR runs on the full image; ViTPose runs top-down on a box.

| backend | detected | mean OKS | PCK@0.05 | PCK@0.10 |
|---|---|---|---|---|
| RF-DETR Keypoint | 99.4% | 0.773 | 0.847 | 0.967 |
| ViTPose-Plus-Huge on labelled box | 100% | 0.824 | 0.906 | 0.990 |
| ViTPose-Plus-Huge on RF-DETR box | 99.4% | 0.823 | 0.903 | 0.989 |

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

## 3. The far-player problem and fix

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

## 4. Speed on an M2 MacBook, 16 GB, MPS

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

Keep both, with the roles they already have. RF-DETR for detection, tracking,
and the far-court pass. ViTPose on the two tracked player boxes for the joints
that get analysed. RF-DETR's own joints are stored too, as a second opinion
and for the uncertainty they carry.

Ground truth said ViTPose is 5 OKS points better and the gap is in the arms.
That is the number that decides it; the two-stage cost is one extra 0.25 s
per frame.

## Caveats

- The labelled dataset is close-range and self-recorded. It says who is more
  accurate on tennis poses at ~200 px, not on a 109 px broadcast far player.
  For that we need to label a few hundred broadcast frames ourselves.
- One rally block, one camera. Other end-on framings will need the on-court
  filter's baseline constants retuned or, better, replaced by the court
  detector's homography.
