# Data

No dataset labels everything at once. Training merges three kinds of label:

1. **Public datasets**, each labelling a few heads ([DATASETS.md](../DATASETS.md)).
2. **Labeler output** on unlabelled footage: the best available model per task.
3. **Astra labels**, via the Codex CLI.

## Ground truth by task

Ground truth means labelled by people. Our own model outputs (RF-DETR boxes,
ViTPose joints, the USO view manifest, audio hit candidates) are **not**
ground truth; they are labeler output to be scored against it.

| Task | Human ground truth, local now | Human ground truth to download | Gap: must be generated | Next step |
|---|---|---|---|---|
| Person boxes | TennisSegmentation: 197 broadcast frames, both players (masks to boxes). Tennis Player Actions: 2,000 self-recorded images, one player each. | COCO person; TennisExpert boxes | Our phone footage, far player | Score Astra on both local sets; RF-DETR already scored (100% recall on the 197). Every person is labelled; no player / not-player label: players are the tracks the event head names as hitters. |
| Body joints (17) | Tennis Player Actions: 2,000 images, 17 COCO joints + neck, ~200 px players | COCO keypoints (not tennis) | Far player and broadcast-size players: no public tennis labels | Done: ViTPose confirmed (0.818 vs Astra 0.743 OKS). Hand-label far players in the gold set. |
| Neck, head top | Tennis Player Actions: neck | Halpe-FullBody (neck, head top) | Tennis head top | Neck = shoulder midpoint of the ViTPose labels. Head top learned from Halpe only; checked in the gold set. |
| Feet (big toe, small toe, heel, both sides) | COCO-WholeBody val (CC BY-NC; everyday photos, not tennis) | COCO-WholeBody train, Halpe-FullBody | Any tennis footage | Score Astra on COCO-WholeBody val; a pose model only if Astra is weak. Feet need only sparse labels, so Astra's speed is acceptable. Hand-label feet in the gold set. |
| Racket (5 points) | none | RacketVision: ~150k tennis broadcast frames, racket box + 5 keypoints, not linked to a player | Handheld footage; which player holds it | Download; attach each racket to the nearest wrist; train a racket labeler; score it and Astra (pick among candidates). |
| Court points | TennisSegmentation court mask (surface, not points; corners only approximate) | TennisCourtDetector: 8,841 frames, 14 points; Roboflow sets | Handheld / phone angles | Download TCD; score Astra (pick among candidate points) and a court model. Hand-label a few hundred own frames. |
| Ball position | TennisSegmentation ball mask, where visible | TrackNet tennis 20k frames; RacketVision 64k | Handheld footage | Download TrackNet; score WASB and Astra (stop after 50 if poor). |
| Hit / bounce timing, hitter | none | TrackNet (hit, bounce); E2E-Spot (serve, swing contact, bounce, near / far; videos via YouTube ids) | Handheld footage | Download TrackNet; score Astra on frame strips and the event labeler. |
| Stroke type | Tennis Player Actions: forehand / backhand / serve / ready, single images | F3Set (fine-grained, videos via YouTube ids); THETIS (12 classes, indoor clips); TennisExpert | Clips from phone footage | Score Astra on Tennis Player Actions now, F3Set later. |
| View ok | none exact. USO `shots.json` is method output (keeps only the high camera) | none | Everything but the USO clip | **Definition:** a camera behind one baseline, any height, whole court and both players in view. Astra agrees with USO on all shots except the low camera, which this definition counts as usable. Checked in the gold set. |
| In play | Richard / Dylan cut lists: hand-reviewed but padded activity edits, not strict points | F3Set / E2E-Spot / TennisExpert rally clips | Everything, strictly | **Definition:** serve toss to end of rally; walking, ball bouncing and breaks are out. Cut windows are strict spans plus padding. Astra agrees 82% with the padded windows; scored strictly in the gold set. |
| Track identity | none | none for tennis | Everything | Measure id switches with `track_metrics`; Astra checks track continuity on sampled pairs. |
| Contact point | not labelled directly: derived from hit frame + wrist / ball + court homography | | | Evaluated through its inputs. |

Three tasks can be evaluated today without downloads: person boxes, joints,
coarse stroke type (plus provisional view and in play).

## Footage

Self-recorded footage is the main source, because that is the target domain
and there is a lot of it. Look for variety: phone behind the baseline, fence
mounts, side views, high and low cameras, hard / clay / grass / indoor,
day and night, singles and doubles, coaching sessions and matches. Start with
100 hours; grow when the gold-set numbers stop improving.

Store video ids and timestamps, not redistributed video. Downloads stay in the
ignored `data/` tree. Broadcast and YouTube footage is for training only
(see the note at the top of DATASETS.md); check each platform's terms before
bulk downloading.

## Labelers

Chosen in phase 0 on public ground truth; scores in
[RESULTS-training.md](../RESULTS-training.md), sign-offs in
[PROGRESS.md](PROGRESS.md).

| Head | Labeler | Why | Status |
|---|---|---|---|
| person boxes | RF-DETR Medium @ 1152 + ByteTrack (`dsa.pose`) | ties Astra (100% recall), 1000x faster | confirmed |
| body joints | ViTPose-Plus-Huge on each box | 0.818 OKS vs Astra 0.743 | confirmed |
| neck | shoulder midpoint of ViTPose | no model needed | confirmed |
| head top | Halpe training data only | no labeler; checked in the gold set | confirmed |
| feet | Astra on sparse keyframes, left / right taken from the nearest ViTPose ankle | 0.870 OKS; no open 6-foot-point model we run | confirmed |
| racket | RacketVision RTMDet-M + RTMPose-M (released weights); racket assigned to the nearest ViTPose wrist; Astra where the detector misses | PCK@0.1 81% end to end vs Astra 58%; wrist / person boxes worse | confirmed |
| ball | WASB tennis weights; Astra checks frames where they disagree | F1@4px 0.886 vs Astra 0.691 | confirmed |
| stroke type, coarse | Astra on the frame | 99 / 100 | confirmed |
| view | Astra, one frame | 99 / 100 under the agreed definition | provisional: gold set scores it |
| in play | Astra, 4-frame sheet | 83% vs padded cut lists | provisional: gold set scores it |
| court | Astra on one keyframe per camera shot, and again wherever the camera moves (court lines tracked frame to frame) | ties TennisCourtDetector (96.7% vs 95.6% within 7 px), better on odd angles; no court model needed | confirmed |
| events | audio onsets + wrist speed + ball direction change vs Astra; E2E-Spot as second opinion | | to test |
| stroke type, fine | Astra on a strip around each hit vs F3Set | | to test |

Labels from two labelers that agree get high confidence; disagreements get low
weight and go into the review queue.

## Astra via Codex

Astra (`gpt-6-astra`, the default model in `~/.codex/config.toml`) runs
headless through the local Codex CLI with images and a JSON schema:

```bash
codex exec -m gpt-6-astra -s read-only \
  -i sheet.png --output-schema schemas/stroke.json -o label.json \
  "Frames 1-8 surround one hit, 33 ms apart. Which player hits, and which stroke?"
```

- **Evaluate it on every label type, not only categorical ones.** Score
  Astra against the gold set with the same metric as the model: pixel error
  for court points and ball, OKS for joints, ±2 frames for events, accuracy
  for stroke, view and in play. Per task, keep whichever of Astra and the
  specialist labeler scores better. Expect it to be strongest on categories
  and weakest on a 3 px ball and far-player wrists, but measure.
- **Prefer choosing over drawing for coordinates.** Give it candidates from
  a vision labeler (numbered points or boxes drawn on the image, or zoomed
  crops) and ask which is right or what is wrong, rather than asking it to
  produce coordinates from nothing.
- **Gold labels stay human.** If Astra pre-fills gold-set labels to speed up
  review, record that, since it biases its own score on those items.
- **Batch into contact sheets**: a grid of frames with timestamps and numbered
  boxes drawn in, several questions per call.
- **Cache by (video, window, prompt version).** Changing a prompt re-labels
  only what depends on it.
- **Measure before trusting.** Score Astra per task against the gold set; a
  task where it is unreliable does not use it.
- **Spend calls where they matter.** A subscription has usage limits, and 100
  hours of play holds tens of thousands of hits. Label a sample first, train
  the head, then ask Astra only about low-confidence or disagreeing cases.
- `-s read-only`: Astra answers questions; it does not run commands.

## Label format

One file per clip window, keyed by source frame, every label with a
`source` (dataset, labeler + version, `astra` + prompt version, or `human`)
and a `confidence`. Missing means unlabelled, not negative. Source timestamps
are preserved, as in the preprocessing manifest ([PREPROCESSING.md](../PREPROCESSING.md)).

```
video, fps, frame
view_ok, in_play                        per frame
people[]: track, box, keypoints[30][x, y, vis]   17 body, neck, head top, 6 feet, 5 racket
court[14][x, y, vis]
ball: x, y, vis                          optional
events[]: frame, kind (hit|bounce|serve), hitter, stroke
```

## Gold eval set

Hand-checked, held out by recording, never used for training or for tuning
prompts: two of our own sessions plus two or three YouTube recordings chosen
to differ in court and camera. Joints and court points on about 500 frames;
events, stroke types, view and in play over about 30 minutes.
