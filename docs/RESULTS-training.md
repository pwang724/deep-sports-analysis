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
