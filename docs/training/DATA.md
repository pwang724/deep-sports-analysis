# Data

No dataset labels everything at once. Training merges three kinds of label:

1. **Public datasets**, each labelling a few heads ([DATASETS.md](../DATASETS.md)).
2. **Labeler output** on unlabelled footage: the best available model per task.
3. **Astra labels**, via the Codex CLI.

## Ground truth by task

Ground truth means labelled by people. Our own model outputs (RF-DETR boxes,
ViTPose joints, the USO view manifest, audio hit candidates) are **not**
ground truth; they are labeler output to be scored against it.

| Task | Human ground truth, local now | Human ground truth to download | Gap: must be generated | Status / next step |
|---|---|---|---|---|
| Person boxes | TennisSegmentation: 197 broadcast frames, both players (masks to boxes). Tennis Player Actions: 2,000 self-recorded images, one player each. | COCO person; TennisExpert boxes | Our phone footage, far player | Done: RF-DETR (ties Astra, 100% recall). Every person is labelled; no player / not-player label: players are the tracks the event head names as hitters. |
| Body joints (17) | Tennis Player Actions: 2,000 images, 17 COCO joints + neck, ~200 px players | COCO keypoints (not tennis) | Far player and broadcast-size players: no public tennis labels | Done: ViTPose confirmed (0.818 vs Astra 0.743 OKS). Hand-label far players in the gold set. |
| Neck, head top | Tennis Player Actions: neck | Halpe-FullBody (neck, head top) | Tennis head top | Neck = shoulder midpoint of the ViTPose labels. Head top: Astra, in the feet call; checked in the gold set. |
| Feet (big toe, small toe, heel, both sides) | COCO-WholeBody val (CC BY-NC; everyday photos, not tennis) | COCO-WholeBody train, Halpe-FullBody | Any tennis footage | Deferred: Astra scored 0.870 OKS on COCO photos but is several px off on broadcast players; left unlabelled for now. |
| Racket (5 points) | none | RacketVision: ~150k tennis broadcast frames, racket box + 5 keypoints, not linked to a player | Handheld footage; which player holds it | Done: RacketVision's released RTMDet + RTMPose (81% vs Astra 58%); no training. Each racket attached to the nearest ViTPose wrist. |
| Court points | TennisSegmentation court mask (surface, not points; corners only approximate) | TennisCourtDetector: 8,841 frames, 14 points; Roboflow sets | Handheld / phone angles | Done: Astra per camera shot (ties TennisCourtDetector, 96.7% vs 95.6% within 7 px). Hand-label a few hundred own frames. |
| Ball position | TennisSegmentation ball mask, where visible | TrackNet tennis 20k frames; RacketVision 64k | Handheld footage | Done: WASB (F1 0.886 vs Astra 0.691 at 4 px). |
| Hit / bounce timing, hitter | none | TrackNet (hit, bounce); E2E-Spot (serve, swing contact, bounce, near / far; videos via YouTube ids) | Handheld footage | Hitter done: Astra (100% when found). Hit timing deferred: Astra 93% near, 40% far within 2 frames; later a ball + audio + wrist labeler, scored on E2E-Spot / F3Set. |
| Stroke type | Tennis Player Actions: forehand / backhand / serve / ready, single images | F3Set (fine-grained, videos via YouTube ids); THETIS (12 classes, indoor clips); TennisExpert | Clips from phone footage | Coarse done: Astra 99%. Forehand / backhand: Astra 93% on F3Set. Direction: computed from WASB + court. Technique (slice etc.): open. |
| View ok | none exact. USO `shots.json` is method output (keeps only the high camera) | none | Everything but the USO clip | **Definition:** a camera behind one baseline, any height, whole court and both players in view. Astra agrees with USO on all shots except the low camera, which this definition counts as usable. Checked in the gold set. |
| In play | Richard / Dylan cut lists: hand-reviewed but padded activity edits, not strict points | F3Set / E2E-Spot / TennisExpert rally clips | Everything, strictly | **Definition:** serve toss to end of rally; walking, ball bouncing and breaks are out. Cut windows are strict spans plus padding. Astra agrees 82% with the padded windows; scored strictly in the gold set. |
| Track identity | none | none for tennis | Everything | Measure id switches with `track_metrics`; Astra checks track continuity on sampled pairs. |
| Contact point | not labelled directly: derived from hit frame + wrist / ball + court homography | | | Evaluated through its inputs. |

Phase 0 is scored for every task above except track identity; results in
[RESULTS-training.md](../RESULTS-training.md), decisions in
[DECISIONS.md](DECISIONS.md).

## Footage

The labelling set is broad, not our own sessions: every public dataset with
human labels (converted as they are) plus a large, diverse collection of
YouTube tennis recordings, with our own footage as one slice. Diversity
matters more than volume: hard / clay / grass / indoor courts; cameras behind
the baseline high and low, fence-mounted, side-on, corner, drone and
broadcast; day, night and indoor light; singles, doubles, drills, lessons
and matches; juniors, club, college and pro; phone, GoPro and broadcast
quality. Start with ~150 videos x 3 one-minute segments; grow by hours when
the gold-set numbers stop improving.

YouTube videos are tracked in `data/videos/youtube/manifest.csv` (id, url,
channel, licence, duration, category, camera, surface, level, segments) and
fetched with `python -m dsa.data.youtube`. Downloads stay in the ignored
`data/` tree and are used for training only (see DATASETS.md).

## Data layout

`data/` (on PW_SSD) is organised by role; paths come from `dsa.data.paths`,
the list of datasets from `dsa.data.sources` (`python -m dsa.data.sources`
shows what is on disk and converted).

```
data/
  sources/<dataset>/     public datasets exactly as released (read only)
  code/<repo>/           third-party code we run: WASB, RacketVision, TennisCourtDetector
  videos/broadcast/      E2E-Spot / F3Set test matches
  videos/youtube/        <video_id>/seg<k>_<start>.mp4 + manifest.csv
  raw/                   our own recordings (preprocess outputs point here)
  labels/<source>/       every source in the one label format (below)
  gold/                  human-reviewed frames
  scratch/               throwaway images
```

## Labelers

Chosen in phase 0 on public ground truth; scores in
[RESULTS-training.md](../RESULTS-training.md), sign-offs in
[PROGRESS.md](PROGRESS.md).

| Head | Labeler | Why | Status |
|---|---|---|---|
| person boxes | RF-DETR Medium @ 1152 + ByteTrack (`dsa.pose`) | ties Astra (100% recall), 1000x faster | confirmed |
| body joints | ViTPose-Plus-Huge on each box | 0.818 OKS vs Astra 0.743 | confirmed |
| neck | shoulder midpoint of ViTPose | no model needed | confirmed |
| head top | Astra, in the same call as the feet | a simple point; not scored, checked in the gold set | confirmed |
| feet | deferred: left unlabelled | 0.870 OKS on COCO photos, several px off on broadcast players | deferred |
| racket | RacketVision RTMDet-M + RTMPose-M (released weights); racket assigned to the nearest ViTPose wrist; Astra where the detector misses | PCK@0.1 81% end to end vs Astra 58%; wrist / person boxes worse | confirmed |
| ball | WASB tennis weights; Astra checks frames where they disagree | F1@4px 0.886 vs Astra 0.691 | confirmed |
| stroke type, coarse | Astra on the frame | 99 / 100 | confirmed |
| view | Astra, one frame | 99 / 100 under the agreed definition | provisional: gold set scores it |
| in play | Astra, 4-frame sheet | 83% vs padded cut lists | provisional: gold set scores it |
| court | Astra on one keyframe per camera shot, and again wherever the camera moves (court lines tracked frame to frame) | ties TennisCourtDetector (96.7% vs 95.6% within 7 px), better on odd angles; no court model needed | confirmed |
| events (hit timing) | later: WASB ball direction change + audio onsets + ViTPose wrist speed | Astra finds contact within 2 frames 93% for the near player, 40% for the far one | deferred |
| hitter, forehand / backhand | Astra on a 12-frame strip around each hit, plus the far half of the court enlarged | hitter 100% when the hit is found; forehand / backhand 93% (98% near, 87% far) | confirmed |
| shot direction | computed: WASB ball track mapped onto the court (cross-court, down the line, ...) | Astra 61% by eye | confirmed, to build |
| technique (topspin / slice / volley / ...) | open: ball path after the hit, or a small model on F3Set | Astra found 1 of 11 slices | open |

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

Defined in `src/dsa/data/schema.py`; converters in `src/dsa/data/convert/`.
Each source is a directory of Parquet tables keyed by
`sample = <source>/<media id>/<frame>`:

```
frames    sample, source, split, media, frame, fps, width, height
people    sample, person, track, box, kp[30][x, y, vis], stroke, labeler
rackets   sample, racket, box, kp[5][x, y, vis], person, labeler    before attaching to a wrist
ball      sample, x, y, visible, labeler
court     sample, kp[14][x, y, vis], labeler
scene     sample, view, in_play, labeler
events    source, media, frame, fps, type, side, hand, technique, direction, outcome, labeler
```

Keypoints: 17 COCO body, neck, head top, 6 feet, 5 racket. Every row names
its `labeler` (a dataset for human labels, or a model / Astra with version).
A missing row or NaN means unlabelled and is masked from the loss; absence is
explicit (`visible = False`, `vis = 0`).

## Gold eval set

Hand-checked, never used for training or for tuning prompts, and held out by
video: about 500 frames sampled across the labelling set so every camera,
surface, level and play type is represented (YouTube, broadcast, our own
sessions), plus 30 minutes of events, strokes, view and in play. Frames are
pre-filled by the chosen labelers and corrected by hand in the review tool.
It re-scores every phase-0 choice outside broadcast footage and scores the
model later.

The 500 are analysis-view frames: a camera behind a baseline with the court in
view, whatever the players are doing. Frames that are not (close-ups, replays,
cut-off courts; about half of broadcast) get no pose, ball or court labels;
10% as many are kept, labelled only for view and in play, so the view head is
still scored on what it must reject. Frames are oversampled about 2x to get
there, and Astra's view flag (99/100 in phase 0) does the filtering.
