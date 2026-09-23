# Progress

In order. Plan in [README.md](README.md); each "confirm" is Peter's sign-off
on a score table plus the worst misses.

## Phase 0: choose a labeler per task (public ground truth)

- [x] Training plan, model and data docs
- [ ] Astra harness: `codex exec` wrapper, output schemas, image sheets, cache
- [ ] Joints: Astra vs ViTPose on Tennis Player Actions, then confirm
- [ ] Person boxes: Astra vs RF-DETR on TennisSegmentation + Tennis Player Actions, then confirm
- [ ] Stroke type, coarse: Astra on Tennis Player Actions (4 classes), then confirm
- [ ] View / in play, provisional: Astra vs USO manifest and Richard / Dylan cuts, then confirm
- [ ] Download TennisCourtDetector, TrackNet tennis, RacketVision
- [ ] Label format and converters (TrackNet, TennisCourtDetector)
- [ ] Ball: WASB vs Astra on TrackNet, then confirm
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
