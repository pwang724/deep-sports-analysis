# Progress

In order. Plan in [README.md](README.md), decisions in [DECISIONS.md](DECISIONS.md); each "confirm" is Peter's sign-off
on a score table plus the worst misses.

## Phase 0: choose a labeler per task (public ground truth)

- [x] Training plan, model and data docs
- [x] Move data/, output/, models/ to PW_SSD (symlinks in the repo)
- [x] Astra harness: `codex exec` wrapper, output schemas, cache (`dsa.astra`)
- [x] Joints: Astra vs ViTPose on Tennis Player Actions. Confirmed: ViTPose (0.818 vs 0.743 OKS, 100 images)
- [x] Neck: shoulder midpoint from ViTPose, no model. Head top: Halpe training data only, checked in the gold set
- [x] Feet: Astra on COCO-WholeBody val, 100 people: 0.787 OKS, 0.870 with left / right swaps fixed by nearest ViTPose ankle. Confirmed: Astra + ankle side fix
- [x] Person boxes: Astra vs RF-DETR on TennisSegmentation, 100 frames: tie (100% recall, IoU 0.88-0.90 both). Confirmed: RF-DETR
- [x] Stroke type, coarse: Astra on Tennis Player Actions: 99% (99 / 100). Confirmed: Astra
- [x] View / in play definitions agreed: view = camera behind a baseline, any height, whole court; in play = serve toss to end of rally. Astra labels both provisionally; strict scoring in the gold set
- [x] Download TrackNet tennis, TennisCourtDetector (labels + 100 val images unpacked), RacketVision (3 test matches), E2E-Spot / F3Set test videos, COCO-WholeBody val
- [ ] Label format and converters (TrackNet, TennisCourtDetector)
- [x] Ball: WASB vs Astra on TrackNet, 100 frames: F1@4px 0.886 vs 0.691, @10px 0.962 vs 0.933. Confirmed: WASB; Astra checks disagreements
- [x] Racket: RacketVision's released RTMDet + RTMPose vs Astra on 39 test-split rackets: PCK@0.1 of racket length 93% (labelled box) / 81% (own detector) vs Astra 58%. Use RacketVision's model; no training needed. Boxes from wrists (74% found) or person boxes (18%) lose to its detector (92%); wrists assign rackets to players
- [x] Court: Astra vs TennisCourtDetector's released model, 100 val frames: within 7 px 96.7% vs 95.6%, median 2.2 vs 1.9 px. Confirmed: Astra, one keyframe per camera shot; no court model trained
- [x] Events: Astra on E2E-Spot / F3Set, 80 hits: contact within 2 frames 96% near, 37% far; hitter 100% when found. Confirmed: Astra for hitter; hit timing deferred (later: ball direction change + audio + wrist speed)
- [x] Stroke type, fine: Astra on 100 F3Set shots: forehand / backhand 98% near, 60% far; slice 1 / 11; direction 55%. Confirmed: Astra for forehand / backhand; direction computed from the WASB track on the court
- [ ] Shot direction from WASB track + court points; score on F3Set
- [ ] Technique (slice etc.): pick a method (ball path or a model on F3Set)
- [ ] Hit timing labeler (ball direction change + audio + wrist speed); score on E2E-Spot / F3Set

## In parallel: early-fusion ball experiment

- [ ] 9-channel RF-DETR Medium + ball head, trained on TrackNet
- [ ] Compare with WASB on the same held-out split; decide ball head vs WASB

## Phase 1: gold set on our domain

- [ ] Review tool (correct pre-filled labels)
- [ ] Label: own sessions + 2-3 YouTube recordings (far-player joints, court, events, strokes, view, in play)
- [ ] Re-check phase-0 choices on it

## Phases 2-3: footage and labels

- [ ] Decide YouTube sourcing (platform terms); collect 100 h; normalize to 30 fps
- [ ] Label factory on Modal; spot-check samples

## Phases 4-6: model

- [ ] v0 per frame (people, court, ball, view, in play)
- [ ] v1 temporal (events, stroke, identity, audio)
- [ ] Self-training rounds
