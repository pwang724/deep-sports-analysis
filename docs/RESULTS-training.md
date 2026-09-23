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
