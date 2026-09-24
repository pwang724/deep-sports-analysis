# Training decisions

Decisions for the trained model and its data. Pipeline decisions and the
choice to train one model (6) are in [../DECISIONS.md](../DECISIONS.md).

## 1. Labelers: the best existing model per task, Astra where none exists (2026-09-23)

Each labeler was scored against public human labels before use (phase 0).
Specialist models won wherever one exists: RF-DETR for boxes, ViTPose for
joints, WASB for the ball, RacketVision's released model for rackets. Astra
won where nothing we run covers the task: feet, coarse stroke type, view and
in play. No labeler needed training so far.

Astra is close on most geometric tasks (ball within 10 px 96% of the time,
joints 0.74 OKS) but less precise and 16-28 s per call, so it checks
disagreements and fills misses rather than labels geometry. Rackets are found
by RacketVision's detector, not by boxes from our person detector or wrists:
those found 18-74% of rackets against its 92%. All choices are on broadcast or
generic footage; the gold set re-checks them on ours. Table:
[DATA.md](DATA.md#labelers).
