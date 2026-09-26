# Progress

In order. Plan in [README.md](README.md), decisions in [DECISIONS.md](DECISIONS.md); each "confirm" is Peter's sign-off
on a score table plus the worst misses.

## Phase 0: choose a labeler per task (public ground truth)

- [x] Training plan, model and data docs
- [x] Move data/, output/, models/ to PW_SSD (symlinks in the repo)
- [x] Astra harness: `codex exec` wrapper, output schemas, cache (`dsa.astra`)
- [x] Joints: Astra vs ViTPose on Tennis Player Actions. Confirmed: ViTPose (0.818 vs 0.743 OKS, 100 images)
- [x] Neck: shoulder midpoint from ViTPose, no model. Head top: Astra, in the same call as the feet; checked in the gold set
- [x] Feet: Astra on COCO-WholeBody val, 100 people: 0.787 OKS, 0.870 with left / right swaps fixed by nearest ViTPose ankle. Confirmed: Astra + ankle side fix. **Deferred 2026-09-23**: several px off on broadcast-size players in the smoke review; feet left unlabelled for now
- [x] Person boxes: Astra vs RF-DETR on TennisSegmentation, 100 frames: tie (100% recall, IoU 0.88-0.90 both). Confirmed: RF-DETR
- [x] Stroke type, coarse: Astra on Tennis Player Actions: 99% (99 / 100). Confirmed: Astra
- [x] View / in play definitions agreed: view = camera behind a baseline, any height, whole court; in play = serve toss to end of rally. Astra labels both provisionally; strict scoring in the gold set
- [x] Download TrackNet tennis, TennisCourtDetector (labels + 100 val images unpacked), RacketVision (3 test matches), E2E-Spot / F3Set test videos, COCO-WholeBody val
- [ ] Label format and converters (TrackNet, TennisCourtDetector)
- [x] Ball: WASB vs Astra on TrackNet, 100 frames: F1@4px 0.886 vs 0.691, @10px 0.962 vs 0.933. Confirmed: WASB; Astra checks disagreements
- [x] Racket: RacketVision's released RTMDet + RTMPose vs Astra on 39 test-split rackets: PCK@0.1 of racket length 93% (labelled box) / 81% (own detector) vs Astra 58%. Use RacketVision's model; no training needed. Boxes from wrists (74% found) or person boxes (18%) lose to its detector (92%); wrists assign rackets to players
- [x] Court: Astra vs TennisCourtDetector's released model, 100 val frames: within 7 px 96.7% vs 95.6%, median 2.2 vs 1.9 px. Confirmed: Astra, one keyframe per camera shot; no court model trained
- [x] Events: Astra on E2E-Spot / F3Set, 80 hits: contact within 2 frames 93% near, 40% far (with far crop); hitter 100% when found. Confirmed: Astra for hitter; hit timing deferred (later: ball direction change + audio + wrist speed)
- [x] Stroke type, fine: Astra on 100 F3Set shots: forehand / backhand 93% (98% near, 87% far with an enlarged far-court crop); slice 1 / 11; direction 61%. Confirmed: Astra for forehand / backhand; direction computed from the WASB track on the court
- [ ] Shot direction from WASB track + court points; score on F3Set
- [ ] Technique (slice etc.): pick a method (ball path or a model on F3Set)
- [ ] Hit timing labeler (ball direction change + audio + wrist speed); score on E2E-Spot / F3Set

## In parallel: early-fusion ball experiment

- [x] 9-channel RF-DETR Medium + ball head, trained on TrackNet games 1-7 (12 epochs, L40S, 109 min)
- [x] Compare with WASB on games 8-10 (5,675 frames): F1@4px 0.863 vs 0.902 (1-frame ablation 0.570). Decided: keep the ball head in the one model; WASB stays the ball labeler and fallback

## Phase 1: labelling set and gold set

- [x] Data layout (`dsa.data.paths`), label schema (`dsa.data.schema`), source registry (`dsa.data.sources`)
- [x] Converters for the 8 public sources into data/labels/ (`python -m dsa.data.convert`), with spans marking exhaustively labelled event ranges
- [x] YouTube collection: 150 diverse videos x 3 one-minute segments, manifest
- [x] Broadcast event data: F3Set 110 of 114 matches (262 h; 3 have no YouTube id, 1 removed) and E2E-Spot 28 of 28 (73 h), full videos at the labelled fps (`dsa.data.broadcast`)
- [x] Pre-fill (`dsa.label.prefill`): Astra names the players from numbered boxes; view is about the camera only; feet deferred
- [x] Clip labeling: labelers over 10 s clips, people tracked, Astra on keyframes, court snapped to lines (`dsa.label.clips`): clips_v1, 100 clips, 45.6k frames
- [x] Consistency filters mask bad labels and record why, incl. the Codex-free player rule (`dsa.label.consistency`)
- [x] Video review tool (`dsa.label.review_video`)
- [ ] Video review by Peter: watch clips with labels overlaid, flag what looks wrong; error rate per labeler per footage type
- [x] Test set suggested: 147 random view frames + 15 non-view + 30 hard, 67 videos held out (`dsa.label.testset`, data/gold/v1)
- [ ] Test set confirmed by hand in the static tool
- [ ] Re-score the phase-0 choices on the test set

No per-frame human review of training labels (decided 2026-09-23): the
specialists + Astra + consistency filters label training data; the human time
goes into a video spot-check and a small confirmed test set where no public
ground truth exists (amateur and own footage).

## Phases 2-3: footage at scale and label factory

- [ ] Grow YouTube toward 100+ hours where the gold set shows gaps; normalize to 30 fps
- [ ] Label factory on Modal; spot-check samples

## Phases 4-6: model

- [ ] Training set: fill-in labels on broadcast sources, batch mix ([DATA.md](DATA.md#batch-mix-proposed), [MODEL.md](MODEL.md#training-strategy-proposed-2026-09-26))
- [ ] v0 per frame (people, court, ball, view, in play)
- [ ] v1 temporal (events, stroke, identity, audio)
- [ ] Self-training rounds: Astra only on doubtful keyframes
