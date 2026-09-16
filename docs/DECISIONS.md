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
Identity resets at shot cuts and at segment boundaries; that is accepted, see
decision 4.

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
are asked of the joints table, which today holds everyone.
