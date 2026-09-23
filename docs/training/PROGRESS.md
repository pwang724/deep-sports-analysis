# Progress

In order. Plan in [README.md](README.md); each "confirm" is Peter's sign-off
on a score table plus the worst misses.

## Phase 0: choose a labeler per task (public ground truth)

- [x] Training plan, model and data docs
- [x] Astra harness: `codex exec` wrapper, output schemas, cache (`dsa.astra`)
- [x] Joints: Astra vs ViTPose on Tennis Player Actions. Confirmed: ViTPose (0.818 vs 0.743 OKS, 100 images)
- [x] Neck: shoulder midpoint from ViTPose, no model. Head top: Halpe training data only, checked in the gold set
- [x] Feet: Astra on COCO-WholeBody val, 100 people: 0.787 OKS, 0.870 with left / right swaps fixed by nearest ViTPose ankle. Confirmed: Astra + ankle side fix
- [x] Person boxes: Astra vs RF-DETR on TennisSegmentation, 100 frames: tie (100% recall, IoU 0.88-0.90 both). Confirmed: RF-DETR
- [x] Stroke type, coarse: Astra on Tennis Player Actions: 99% (99 / 100). Confirmed: Astra
- [ ] View / in play, provisional: view 100% if low end-on counts as usable; in play 82% vs padded cut windows, mostly label looseness. Awaiting definitions
- [ ] Download TennisCourtDetector, TrackNet tennis, RacketVision (COCO-WholeBody val: done)
- [ ] Label format and converters (TrackNet, TennisCourtDetector)
- [ ] Ball: WASB vs Astra on TrackNet, then confirm
- [ ] Racket labeler: train on RacketVision; score it and Astra, then confirm
- [ ] Court labeler: fine-tune on TennisCourtDetector
- [ ] Court: Astra vs court labeler on TennisCourtDetector, then confirm
- [ ] Events: Astra vs event labeler on TrackNet (+ E2E-Spot), then confirm
- [ ] Stroke type, fine: F3Set, then confirm

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
