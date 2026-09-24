# Training results

Measurements behind the phase-0 labeler choices in
[training/PROGRESS.md](training/PROGRESS.md). Code: `src/dsa/astra/`.

## Body joints: Astra vs ViTPose (2026-09-23)

100 images of Tennis Player Actions, 25 per action (seed 0), 17 COCO joints.
Both see the same view: the labelled box padded 20%. Astra gets the crop
upscaled to 768 px and returns pixel coordinates; ViTPose-Plus-Huge numbers
are its stored per-image results from `output/eval_gt` on the same images.
Metrics as in [RESULTS-pose.md](RESULTS-pose.md).

| labeler | OKS | PCK@0.05 | PCK@0.10 | s / image |
|---|---|---|---|---|
| ViTPose-Plus-Huge | **0.818** | **0.902** | **0.989** | 0.06 (L4, fp16) |
| Astra, high effort | 0.743 | 0.797 | 0.964 | 34 (one call) |
| Astra, medium effort | 0.739 | 0.792 | 0.964 | 17 (one call) |

Mean OKS by action (ViTPose / Astra medium): ready position 0.79 / 0.80,
serve 0.87 / 0.78, forehand 0.81 / 0.69, backhand 0.81 / 0.71. Astra is level
on a still player and falls behind mid-swing; its largest per-joint gaps are
wrists, elbows and hips. Its worst images keep the body shape but offset
joints or appear to swap left and right on players facing away.

Decision: ViTPose labels body joints. Reasoning effort does not matter for
this task, so Astra calls default to medium.

## Feet: Astra on COCO-WholeBody (2026-09-23)

100 people from COCO-WholeBody val (seed 0) with all six foot points labelled
visible and a box at least 150 px tall. Everyday photos, not tennis. Astra
sees the padded, upscaled box (as for joints) and returns both ankles and the
six foot points; OKS uses the COCO-WholeBody foot sigmas. Baseline: every foot
point on the labelled ankle of its side.

| labeler | OKS | PCK@0.05 | PCK@0.10 | s / call |
|---|---|---|---|---|
| Astra, medium effort | **0.787** | **0.815** | **0.907** | 15 |
| Astra, left / right ignored | 0.870 | 0.908 | | |
| ankle copy (baseline) | 0.485 | 0.337 | 0.782 | |

Median error is about 2% of box height on every foot point (4 px on a 200 px
player), against 4-9% for the baseline. Nearly all of the loss is 10 people
whose left and right feet were swapped: points on the right feet, in each
other's slots. Swaps are removable at labelling time by attaching each
Astra foot to the nearest ViTPose ankle.

## Stroke type, coarse: Astra on Tennis Player Actions (2026-09-23)

Same 100 images as the joint test. Astra sees the whole frame and picks
forehand, backhand, serve or ready position; 8 s per call.

| | forehand | backhand | serve | ready |
|---|---|---|---|---|
| forehand (25) | **24** | 0 | 0 | 1 |
| backhand (25) | 0 | **25** | 0 | 0 |
| serve (25) | 0 | 0 | **25** | 0 |
| ready position (25) | 0 | 0 | 0 | **25** |

99% accuracy. The one miss is a distant player early in a forehand
backswing, called ready. This is an easy test: single frames of clear poses
from a public dataset Astra may have seen. Fine-grained types (slice,
volley, overhead) from clips are scored later on F3Set.

## Person boxes: Astra vs RF-DETR on TennisSegmentation (2026-09-23)

100 of the 197 broadcast frames (seed 0), 1280 x 720, far player about 80 px
tall. Astra sees the whole frame and boxes every person on or beside the
court; RF-DETR Medium at 1152 px is scored from its stored benchmark boxes on
the same frames. A player is found at IoU >= 0.5 with its label.

| labeler | far recall | far IoU | near recall | near IoU | boxes / frame | s / frame |
|---|---|---|---|---|---|---|
| RF-DETR Medium, conf 0.4 | 100% | 0.890 | 100% | 0.901 | 11.7 | 0.02 (L4) |
| Astra, medium effort | 100% | 0.882 | 100% | 0.904 | 15.5 | 28 (one call) |

A tie on the labelled players. Other people are unlabelled, so boxes per
frame is only a count. RF-DETR stays the box labeler: equal accuracy at a
thousandth of the time, and it already runs in the pipeline.

## View and in play: Astra vs our cut lists (2026-09-23)

Provisional truth, not hand labels: the USO manifest for view, the Richard /
Dylan cut lists for in play (`dsa.astra.view_play` docstring has the sampling).
50 samples per class per task, 11 s per call.

| task | agree | truth yes, Astra no | truth no, Astra yes |
|---|---|---|---|
| view (USO keep vs discard) | 87% | 0 / 50 | 13 / 50 |
| in play, first prompt | 76% | 16 / 50 | 8 / 50 |
| in play, main court only | 82% | 16 / 50 | 2 / 50 |
| view, agreed definition (any height) | 85%; 99% counting low end-on as usable | 0 / 50 | 15 / 50 |
| in play, agreed definition (strict) | 83% | 16 / 50 | 1 / 50 |

View: all 13 disagreements are the low end-on camera, which the USO recipe
discarded for that clip. Astra called it usable. Our phone footage is a low
end-on view, so it should count as usable, and then Astra agrees on all 100.

In play: the first prompt counted play on neighbouring courts; naming the
main court fixed 6 of 8 false yeses. Most remaining "truth yes, Astra no"
cases viewed show walking, ball bouncing before a serve or an empty court:
the cut windows are padded activity edits, not ball-in-play spans. The cut
lists cannot score a strict in-play label; the gold set must.

With the agreed definitions (view: behind a baseline at any height; in play:
serve toss to end of rally), 14 of the 15 view disagreements are the low
end-on camera, now correct; one is a court-level corner view Astra wrongly
accepted.

## Racket: Astra on RacketVision (2026-09-23)

RacketVision tennis, the 3 matches downloaded: every racket with the tip,
handle and at least 4 of 5 points labelled (39). Each gets a 360 px crop that
contains the racket at a random offset, upscaled to 768 (`dsa.astra.racket`).
Error is relative to racket length (tip to handle; median 64 px in 1080p).
22 s per call.

| labeler | mean error | PCK@0.1 | PCK@0.2 | PCK@0.1, left/right either way |
|---|---|---|---|---|
| Astra, medium effort | 0.126 | 58% | 83% | 61% |
| box centre (baseline) | 0.342 | 13% | 21% | 13% |
| RacketVision RTMPose-M on the labelled box | 0.048 | 93% | 97% | 96% |
| RacketVision RTMDet-M + RTMPose-M, full frame | 0.055 | 81% | 89% | 84% |

Astra's median error per point: tip 0.067, head bottom 0.068, handle 0.098, left
0.102, right 0.109. Every point is 6-7 px from its label at this size, so
PCK@0.1 is tight. Even the worst crops have Astra on the right racket with the
right orientation; the misses are offsets along the shaft or a flipped head
width. 39 rackets is a small sample.

RacketVision's released model (`dsa.cloud.racket_pose`, one L4 on Modal) on
the same 39 rackets, all in its test split. Full frame: its detector found 36
of 39 (IoU >= 0.1 at score >= 0.3); the 3 misses count as 0 PCK, and mean error
is over the found ones. Median error per point on the labelled box: tip 0.022,
head bottom 0.017, handle 0.026, left 0.026, right 0.038, a third of Astra's.

Where RTMPose gets its box, same 39 rackets (`dsa.cloud.racket_pose`; nearest
predicted racket scored, none within 0.5 racket lengths = missed):

| box from | found | PCK@0.1 | PCK@0.2 |
|---|---|---|---|
| labelled racket box | 100% | 93% | 97% |
| RacketVision's RTMDet | 92% | 81% | 89% |
| RTMDet, else forearm square | 92% | 81% | 89% |
| forearm square (0.5 x person height, pushed out from the wrist) | 74% | 57% | 63% |
| wrist square (0.8 x person height) | 74% | 45% | 56% |
| RF-DETR person box | 18% | 4% | 7% |

The person box is useless: the racket is too small in it. Wrist boxes find
fewer rackets than the detector, and the forearm fallback recovers none of the
detector's 3 misses. In every miss a ViTPose wrist is on the handle (within
0.05 person heights), so the loss is in picking the hand or in the box, not
in finding the player. The detector stays; wrists assign each racket to a
player.

RacketVision's model is the racket labeler: about 2 px error against Astra's
6-7 px, milliseconds against 22 s, and it needs no training. Its detector
misses 8% of rackets; Astra stays the fallback for those and for our footage
until the gold set checks both.

## Ball: WASB vs Astra on TrackNet (2026-09-23)

TrackNet tennis test games 8-10 (WASB's test split): 80 frames with a visible
ball, 20 without (`dsa.astra.ball`). Both see frames t-1, t, t+1 and place the
ball in t. WASB is its released tennis HRNet run locally without its tracker;
Astra gets the three full 1280 x 720 frames. 16 s per Astra call.

| labeler | F1 @ 4 px | F1 @ 10 px | recall @ 10 px | median error | ball claimed on 20 empty frames |
|---|---|---|---|---|---|
| WASB | 0.886 | 0.962 | 95% | 2.1 px | 2 |
| Astra, medium effort | 0.691 | 0.933 | 96% | 2.2 px | 5 |

Astra found the ball in all 80 frames that had one, typically within 2 px, and
only one was far off (117 px, a different object). It loses to WASB on
precision at 4 px and on empty frames. WASB stays the ball labeler: better,
and milliseconds per frame against 16 s. Astra can check WASB where the two
disagree.

## Court: Astra vs TennisCourtDetector (2026-09-23)

TennisCourtDetector val split (held out from its training), 100 frames of
1280 x 720 broadcast, 14 line intersections each; 1,394 labelled points in
frame (`dsa.astra.court`). TCD is its released model with both of its
post-processing steps (line refinement, homography fit). Astra gets the frame
and a numbered court diagram; it ran no code (checked in the Codex event log).

| labeler | within 7 px | within 15 px | median error | frames with every point within 15 px | time |
|---|---|---|---|---|---|
| TennisCourtDetector | 95.6% | 97.9% | 1.9 px | 97 | ms (MPS) |
| Astra, medium effort | 96.7% | 99.9% | 2.2 px | 98 | 22 s |

A tie; TCD matches its published 1.83 px. TCD's two worst frames are an
angled Indian Wells camera where it loses points entirely; Astra's worst
point in 100 frames is 17 px off. Worst frames: `output/astra_eval/court/worst.jpg`.

## Events and fine stroke: Astra on E2E-Spot and F3Set (2026-09-23)

Human labels on three US Open matches (E2E-Spot test: 2019 final; F3Set test:
Djokovic-Zverev 2021, Tomljanovic-Jabeur 2022), videos downloaded at 720p and
checked for alignment (`dsa.astra.events`). Astra only: no event model is run.
22 s per call.

Timing: 16 consecutive frames on one 4 x 4 sheet (480 px tiles). 80 windows
with a labelled hit, 20 without.

| | near player | far player | all |
|---|---|---|---|
| contact within 2 frames | 96% | 37% | 70% (60% within 1) |
| hit missed | 0 | 10 | 10 / 80 |
| hitter right when found | | | 100% |
| hit claimed in an empty window | | | 1 / 20 |
| bounce within 2 frames (E2E-Spot) | | | 56% of 18 |

Fine stroke: 100 F3Set shots (29 serves), 12 frames 4 apart from 0.3 s
before contact, hitter and contact frame given.

| | near player | far player | all |
|---|---|---|---|
| forehand / backhand (rallies) | 98% | 60% | 82% |
| technique (rallies) | | | 82%; 1 of 11 slices found |
| direction (rallies) | 63% | 43% | 55% |
| serve direction (T / body / wide) | | | 76% |

Rerun with a second sheet: the top 55% of each frame (the far half of the
court) enlarged, given for every timing window and for far-player strokes
(`--far`, `output/astra_eval/events_far`). Same samples.

| | first run | with far crop |
|---|---|---|
| contact within 2 frames: near / far | 96% / 37% | 93% / 40% |
| hits missed: near / far | 0 / 10 | 1 / 5 |
| hitter right when the hit is found | 100% | 100% |
| bounce within 2 frames | 56% | 61% |
| forehand / backhand: near / far | 98% / 60% | 98% / **87%** |
| forehand / backhand, all rallies | 82% | **93%** |
| technique, all rallies; slices found | 82%; 1 / 11 | 80%; 1 / 11 |
| direction, rallies: near / far | 63% / 43% | 66% / 53% |
| serve direction | 76% | 83% |

The crop fixes forehand / backhand for the far player and nothing else: far
contact timing stays at 40%, slices stay missed (10 of 11 called ground
strokes), and direction by eye stays near 60%.

## Early-fusion ball head vs WASB (2026-09-23)

RF-DETR Medium encoder with COCO weights, patch embed widened to 9 channels
(frames t-1, t, t+1; neighbours start at zero), stride-4 heatmap head,
1280 x 720 input. Trained on TrackNet games 1-7 (14,160 frames), 12 epochs,
batch 16, AdamW 1e-4 (10x on the new layers), on one L40S. Scored on all of
games 8-10 (5,675 frames, 5,472 with a ball); visible when the peak > 0.5,
as WASB. `dsa.train.ball_fusion`, `dsa.cloud.train_ball`; outputs in
output/train/ball_fusion/.

| model | P@4 | R@4 | F1@4 | F1@10 | median px | missed visible | ball on empty |
|---|---|---|---|---|---|---|---|
| 9 channels (early fusion) | 0.927 | 0.808 | **0.863** | 0.893 | 1.7 | 766 | 62 |
| 3 channels (1 frame, ablation) | 0.710 | 0.477 | 0.570 | 0.644 | 2.3 | 1,828 | 32 |
| WASB (3 frames, no tracker) | 0.933 | 0.873 | **0.902** | 0.957 | 1.9 | 370 | 17 |

F1@4 by game (9 ch / 1 ch / WASB): game 8 0.88 / 0.70 / 0.91, game 9 0.85 /
0.66 / 0.85, game 10 0.86 / 0.38 / 0.94. The single frame collapses on game
10 (blue court, hard sun and shade): 26% of its visible-ball frames put a
confident peak on the baseline centre mark or a lit line corner; with three
frames that falls to 4%. The fused model's remaining misses are faint balls
on clay and balls against bodies, rackets, lines and the net; its false
alarms are balls held between points that TrackNet leaves unlabelled. WASB
misses nearly still balls (serve toss, bounce before a serve). 91% of balls
are found by at least one of the two, 77% by both. Loss was still falling at
epoch 12.

