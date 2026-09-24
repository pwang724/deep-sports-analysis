# Decisions

One entry per settled design choice: what, why, and what was rejected. Numbers
live in [RESULTS-pose.md](RESULTS-pose.md); this file records the calls.

## 1. Person detection: RF-DETR Medium at 1152 px, fp16 (2026-09-16)

Finds every person in the frame; nothing sport-specific. 100% recall on both
players on 197 labelled broadcast frames at 20 ms/frame on an L4, Apache.

Rejected: RF-DETR Keypoint Preview with a hand-tuned far-court crop (40% far
player recall on the labelled set, tuned to one clip); YOLO11 (close second,
AGPL, kept as an alternative); RT-DETRv2; open-vocabulary detectors as the
primary detector (slower, looser boxes, weaker on the small far player).

## 2. Identity within a shot: ByteTrack (2026-09-16)

Motion-and-overlap matching on the detector's boxes. No appearance model.
The default runner resets identity at fixed segment boundaries. With a view
manifest (`--shots`), it instead starts a fresh tracker for each kept camera
shot/view interval; detected cuts are never merged. See [PREPROCESSING.md](PREPROCESSING.md).
Neither mode guarantees identity purity within a shot; see decision 4.

Rejected: SAM 3 video mode as detector plus tracker. It lost the far player in
two frames of three, switched identity 3-5x more often, and cost 14x.

## 3. Joints: ViTPose-Plus-Huge, top-down on each tracked box (2026-09-16)

0.824 OKS on 1,994 labelled tennis images vs 0.773 for RF-DETR Keypoint on the
full image and 0.688 on a matched crop. The gap is in wrists and elbows and
widest on serves, which is what stroke analysis reads.

## 4. Who is who: one VLM call per batch of new tracks (2026-09-16)

The detector returns everyone: players, ball kids, line judges, umpire,
camera operators. Deciding which tracks are the players, and which player each
one is, is done by a vision-language model, not by rules:

- When a segment starts, and again whenever ByteTrack spawns new track ids,
  take one frame in which the new tracks are visible, draw every current box
  with its track id, and ask the model which numbered people are which player
  in the match and who is neither. Sport and player names are strings in the
  prompt. Cache the answer per track id; it holds for the id's lifetime.
- Batch: new ids born within a few frames of each other share one call. A
  rally has 15-20 tracks, so an hour of tennis is a few hundred to a couple of
  thousand short calls.
- Role and identity are one question. The model sees the kits and the
  scoreboard, so it can name the player, not just say "player".
- ViTPose then runs only on player tracks, which is about 30% of the boxes.

Why a model over rules, measured: no single-frame detector separates a ball
kid from a player. SAM 3, OWLv2, Grounding DINO and YOLOE, prompted "tennis
player" with and without contrastive negatives, all score a crouching kid in
kit on the court as a tennis player. The distinction is behavioural and
contextual, which a VLM looking at a whole labelled frame can see.

Rejected: court-position rules (broadcast-specific, break on any other camera);
track-length heuristics (a filter at best); appearance clustering with a
consistency check (not needed; the VLM answer is taken as final).

Status: not built yet. Nothing downstream needs it until per-player questions
are asked of the joints table, which today holds everyone. The trained model
(decision 6) does not need it: it detects every person and players are the
tracks its event head names as hitters.

## 5. Preprocessing belongs to each video (2026-09-17)

Its job is to remove unwanted footage before analysis. Each video has a silo
containing its cleanup code, settings, and instructions. A new video may need
an entirely different method; it does not inherit the previous video's rules.
Only video I/O and the retained-interval manifest are shared with downstream
tracking. See [PREPROCESSING.md](PREPROCESSING.md).

The USO highlights silo currently uses DINOv2 keep/discard references, camera
cuts and observed changes of view. That method and its calibration belong to
this clip alone. Kept intervals retain source timestamps and camera boundaries.
The learned replacement is decision 6.

## 6. Direction: one multi-task model; the pipeline becomes the labeler (2026-09-23)

The target is a single network that reads raw video and audio once and
returns view and in-play state, players with joints and identity, court
points, hit/bounce events and stroke type. Per-video preprocessing becomes
two of its outputs instead of hand-built recipes.

No dataset labels all of this, so training merges public datasets (each
covering a few heads) with labels produced by the best model per task over
public self-recorded footage, plus Astra (via Codex) for categorical
questions. Losses are masked to the labels each sample has.

Tennis-only first: RF-DETR Medium (DINOv2 backbone + DETR decoder) with
three-frame early fusion, a per-frame ball head, and a second transformer
across frames; DINOv3 is the first ablation. Video encoders and VLM-as-model
rejected on resolution and cost; the sport-general design is deferred. WASB
labels the ball and is the fallback. Decisions 1-5 remain the
production path until the model matches them on a held-out gold set.
Plan: [training/](training/README.md).


## 7. Labelers: the best existing model per task, Astra where none exists (2026-09-23)

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
[training/DATA.md](training/DATA.md#labelers).
