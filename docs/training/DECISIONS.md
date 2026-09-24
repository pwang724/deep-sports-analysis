# Training decisions

Decisions for the trained model and its data, each with the numbers behind
it. Pipeline decisions and the choice to train one model (decision 6) are in
[../DECISIONS.md](../DECISIONS.md); full score tables and worst-miss notes in
[../RESULTS-training.md](../RESULTS-training.md); the labeler table in
[DATA.md](DATA.md#labelers).

## 1. How a labeler is chosen (2026-09-23)

For every label type, each candidate labeler is scored against public human
labels on about 100 samples, with the metric the trained model will be judged
by, and Peter signs off on the table plus the worst misses (phase 0). The
candidates are the best released specialist model, if one exists, and Astra
(`gpt-6-astra` through the local Codex CLI). Ties go to the faster labeler
unless the slower one fails more gracefully.

Astra runs at medium reasoning effort: high effort scored the same on joints
(0.743 vs 0.739 OKS) at twice the time (34 s vs 17 s per image). A spot check
of the Codex event log confirms it answers from the images and runs no code.

All samples are broadcast or everyday footage. Every choice is re-checked on
our own footage in the gold set (phase 1).

## 2. Labeler per task (2026-09-23)

| Task | Labeler | Score | Alternative | Why |
|---|---|---|---|---|
| Person boxes | RF-DETR Medium @ 1152 | 100% recall, IoU 0.89-0.90 | Astra: 100%, IoU 0.88-0.90 | tie; 0.02 s vs 28 s per frame |
| Body joints (17) | ViTPose-Plus-Huge | 0.818 OKS | Astra 0.743 | better, most of all mid-swing (forehand 0.81 vs 0.69) |
| Neck | shoulder midpoint of ViTPose | | | no model needed |
| Head top | Astra, in the same call as the feet | not scored (no head-top labels downloaded) | | a simple point; checked in the gold set |
| Feet (6) | **Deferred** (left unlabelled) | 0.870 OKS on COCO photos (0.787 before the side fix) | ankle copy 0.485 | on broadcast players (shoe ~15 px) Astra is several px off, e.g. heel placed at the ankle; not needed yet. Astra still returns them, cached, for later |
| Racket (5) | RacketVision RTMDet-M + RTMPose-M | 81% of points within 0.1 racket length | Astra 58% (given a crop) | 2 px vs 6-7 px error; no training needed |
| Ball | WASB (tennis weights) | F1 0.886 @ 4 px, 0.962 @ 10 px | Astra 0.691, 0.933 | more precise, fewer false balls (2 vs 5 of 20) |
| Court (14) | Astra, one keyframe per camera shot | 96.7% within 7 px, median 2.2 px | TennisCourtDetector 95.6%, 1.9 px | tie; Astra's worst point is 17 px, TCD lost points on an angled camera |
| View | Astra, one frame | 99 / 100 | | provisional: truth was a cut list |
| In play | Astra, 4-frame sheet | 83% vs padded cut windows | | provisional: truth was a cut list |
| Stroke, coarse (4 classes) | Astra, one frame | 99 / 100 | | |
| Hitter | Astra, strip around the hit | 100% when the hit is found | | |
| Forehand / backhand | Astra, strip around the hit plus an enlarged far-court crop | 93% (near 98%, far 87%) | same without the crop: 82% (far 60%) | |
| Shot direction | computed from the WASB track on the court | to build | Astra 61% by eye | geometry, not judgment |
| Hit timing | deferred | | Astra: contact within 2 frames 93% near, 40% far | far player unreliable |
| Technique (slice, ...) | open | | Astra found 1 of 11 slices | |

## 3. 30 keypoints per person (2026-09-23)

17 COCO body joints, neck, head top, 6 foot points (big toe, small toe, heel
per foot) and 5 racket points (tip, head bottom, handle, left and right of the
head). Feet give stance and split step, the racket gives swing path and
contact, the head gives gaze and balance. Neck needs no labeler; head top is
labelled by Astra in the feet call and checked in the gold set. Feet are
deferred (2026-09-23): the slots stay in the format but are unlabelled (NaN,
masked) until a labeler holds up at broadcast scale.

## 4. Specialists over Astra for geometry, Astra for judgment (2026-09-23)

Where a released model exists it won or tied on geometry: joints (0.818 vs
0.743 OKS), ball (F1 0.886 vs 0.691 at 4 px), rackets (81% vs 58%), boxes
(tie). Astra is close on geometry (ball within 10 px 96% of the time, joints
0.74 OKS) but 16-28 s a call, so it fills what no model covers (feet, view,
in play, stroke type, hitter) and checks frames where a specialist and it
disagree. The exception is court points, where it tied and is kept because
it degrades better and one call covers a whole camera shot.

## 5. The trained model needs no player-vs-not stage (2026-09-23)

The model reads full frames and detects every person; players are the tracks
the event head names as hitters. Boxes are an output, not an input.

## 6. Rackets are found by RacketVision's detector; wrists only assign them (2026-09-23)

Same 39 test-split rackets, RTMPose given boxes from different sources:

| box from | rackets found | PCK@0.1 |
|---|---|---|
| labelled racket box (not available in practice) | 100% | 93% |
| RacketVision's RTMDet | 92% | 81% |
| square around the forearm (ViTPose) | 74% | 57% |
| square around the wrist (ViTPose) | 74% | 45% |
| RF-DETR person box | 18% | 4% |

A forearm fallback for the detector's 3 misses recovered none. A ViTPose
wrist is on the handle in every miss, so wrists are used to say which player
holds which racket. Astra is the fallback where the detector misses.

## 7. Events and fine stroke: Astra names hitter and forehand / backhand; the rest is computed or deferred (2026-09-23)

Scored on human labels from three US Open matches (E2E-Spot, F3Set test
sets): 80 hit windows of 16 frames, 100 shots (29 serves).

- **Hitter: Astra.** Right on every hit it found (100%).
- **Forehand / backhand: Astra, with the far half of the court enlarged as a
  second image.** 93% on rally shots (near 98%, far 87%); without the crop
  the far player was 60%.
- **Shot direction: computed from the WASB ball track mapped onto the court**
  (cross-court, down the line, down the middle, inside-out, inside-in). Astra
  by eye gets 61% (near 66%, far 53%); it is a geometry question.
- **Hit timing: deferred.** Astra finds contact within 2 frames 93% of the
  time for the near player but 40% for the far one, crop or not. Later
  labeler: ball direction change (WASB) + audio onset + ViTPose wrist speed.
- **Technique (topspin / slice / volley / ...): open.** Astra called 10 of 11
  slices ground strokes. Candidates: the ball's path after the hit, or a small
  model trained on F3Set's labels.

## 8. Data lives on PW_SSD (2026-09-23)

The internal disk filled (780 MB free) during downloads. `data/`, `output/`
and `models/` are symlinks into `/Volumes/PW_SSD/deep-sports-analysis/`
(exFAT, 336 GB free at the move), so the SSD must be connected to run
anything, and virtualenvs cannot live there.

## 9. Singles from behind the baseline only, for now (2026-09-23)

Labelling set, test set and the first model cover singles filmed from behind
a baseline (any height). Doubles, drone, fence-side and side-on footage are
out of scope: they need 4 players, other hitter logic and other court
geometry, and the core case is enough to prove the model. Of the first 150
selected YouTube videos, 29 were doubles and 5 drone or fence; they are
replaced by singles. Frame by frame, a keyframe where Astra names more than
2 players is treated like a non-view frame (scene labels only). Doubles can
come back once singles works.

## 10. The ball stays a head of the one model; WASB labels it for now (2026-09-23)

Early fusion works: three stacked frames lift the ball head from F1 0.570 to
0.863 at 4 px on TrackNet games 8-10, against WASB's 0.902, after one untuned
12-epoch run on 14k frames from 7 matches. Where it finds the ball it is as
precise as WASB (median 1.7 vs 1.9 px, precision 0.93 both); the gap is
recall (766 missed visible balls vs 370), worst on the one unfamiliar court
(game 10: 0.86 vs 0.94). The misses of the two rarely overlap, so training
on WASB's labels over our varied footage is the direct fix, plus longer
training and between-point negatives. WASB remains the ball labeler and
the fallback until the head matches it on the test set.

## 11. One slim Astra call per keyframe; head top by rule (2026-09-24)

A Codex top-up buys only ~550-600 Astra calls, and each keyframe cost two:
the scene call and a head-top call. The slim scene call (no duplicate frame,
320 px time sheet, thin boxes) keeps accuracy on court, view, in play and
players and uses ~12% instead of ~16% of the meter per 100 keyframes. The
head-top call is replaced by a rule on ViTPose and the box (1.9 px median
from Astra) and now labels every frame, not only keyframes. Together about
3x less Codex use per keyframe. Batching 4 keyframes a call saves a little
more but within the meter's resolution; held. Sol and Luna were worse than
Astra on court and in play (RESULTS-training), so Astra stays.

