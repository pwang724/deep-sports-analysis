# Peter / Richard / Dylan recordings

This is a separate preprocessing recipe for the two September phone recordings.
It does not use the USO camera-view classifier. **Unscored play cuts are independent
of scoring:** they need retained source windows, not point winners or drill labels.
The optional scored workflow can later render elevens and matches with a
scoreboard once point outcomes and rules are annotated.

**Current data status:** both sources have been normalized. Unscored cut profiles
cover activity throughout both recordings: Richard's revised cut has 78 retained
windows (33:22.5), and Dylan's revised cut has 106 retained windows (24:20.0).
The earlier exports truncated live points and have been moved to Trash.
Original audio is retained without overlays.
Scoring is deferred. Bundled scoring profiles
contain draft grouping suggestions, not completed point/score logs. Dylan's
first five point windows have been inspected and entered as drafts. No point
winners have been verified. Four opening Richard rally windows are also logged
as drafts; verify whether they are warmup or scored play. Richard's proposed
four elevens and subsequent served play still require complete point annotation.
These are not finished scored edits.

## Unscored cuts

```bash
python -m dsa.preprocess.desktop_tennis export-play \
  --out output/preprocess/desktop_tennis/richard \
  --destination 'output/preprocess/desktop_tennis/exports/Richard - tennis play - revised.mp4'

python -m dsa.preprocess.desktop_tennis export-play \
  --out output/preprocess/desktop_tennis/dylan \
  --destination 'output/preprocess/desktop_tennis/exports/Dylan - tennis play - revised.mp4'
```

`cuts.json` is the editable decision list. Its half-open `start_frame` / `end_frame`
ranges refer to the normalized 30 fps master. It has its own source hash and
requires no `annotations.json`, initial score, point winner, or game type.
If absent, `export-play` initializes it from the bundled `*_cuts.json` profile
only when the source hash matches. Existing cut lists are never replaced by a
bundled profile. Ranges must be ordered, non-overlapping and within the source.

The source-matched profiles were made from a full-recording acoustic scan,
visual review of candidate starts/ends, and focused checks of ambiguous events.
Player-pose checks helped choose lead-in padding and locate a quiet far-end serve.
Obvious collection, breaks, caught tosses and false neighboring-court events
were removed; identified missed serves were kept. These are **approximate
activity edits**, not an exhaustive frame-by-frame ground-truth annotation:
quiet/off-camera activity may be missed and brief resets can remain. Clip
counts are not point counts. Complete sources remain available for refinement.

Dylan revision 2 restores footage after inspecting temporal context beyond all
122 original endpoints, including half-second samples for ambiguous endings.
The revised endings were checked again, and uncertain short gaps were joined to
preserve quiet serves and returns. It retains every frame from the first edit,
adds conservative tails, and deliberately leaves some short resets. This is a
visual boundary correction, not an exhaustive ball-track annotation.

Richard revision 2 likewise reviews all 135 original endings and starts, with
denser checks of ambiguous continuations and quiet serve attempts. The new
endings were checked in source context. It restores cut-off returns and missed
serve attempts, joins nearby exchanges, and retains some short resets. One
original false-positive window showing an empty court during a bench break was
removed. All other originally retained footage remains.

Audio proposals are **not point boundaries**. The initial proposal algorithm
trimmed weak hits more than 2.5 seconds beyond the last strong hit, discarding
quiet rally continuations. It now retains the entire connected impact group
and marks `point_end_verified: false`. Inspect footage beyond both ends and
confirm play has stopped before accepting a cut; a fixed silence threshold or
padding duration alone cannot establish that a point ended.

For a fresh audio proposal pass (requires the `preprocess` extra):

```bash
python -m dsa.preprocess.desktop_tennis detect-play \
  --out output/preprocess/desktop_tennis/dylan --start 30 --end 3860
```

This writes `cuts.candidates.json`, preserving the reviewed `cuts.json`.
Thresholds are tuned to these recordings and require visual review. Single
struck serves are eligible even without a return. Padding needs extra care
when the first audible hit is a return of a quiet far-end serve.

`export-play` writes an MP4 and JSON mapping output spans back to source frames.
It checks the final frame count and audio duration. The separate `play_shots.json`
manifest partitions the clean master into kept activity and discarded gaps,
resetting tracking at each cut. It preserves the scoring workflow's files.

## Sources and timeline

The originals were moved from `~/Desktop/tennis_videos/` into the repo's ignored
`data/raw/tennis_videos/` directory:

- `peter_richard_202609.MOV` — about 62 minutes; phone rotation is 180 degrees.
- `peter_dylan_202609.MOV` — about 69 minutes; Dylan wears black and a cap.

`prepare` retains these originals and produces an upright 1280×720 SDR H.264
`analysis.mp4` with audio at a constant 30 fps. On macOS, VideoToolbox converts
the source HDR color and encodes the master; on other systems, FFmpeg needs
`zscale` and `tonemap` for these HDR sources.

All annotation times are seconds from the original recording's start. The
analysis master preserves elapsed time but **does not preserve original phone
frame IDs**. `source.json` stores the original probe, SHA-256 and normalization
settings. `shots.json` deliberately references the clean analysis master.
Do not pass the original MOV as `--video` with this manifest.

## Commands

From the repository root, using its Python environment:

```bash
python -m dsa.preprocess.desktop_tennis prepare \
  data/raw/tennis_videos/peter_richard_202609.MOV \
  --opponent richard --out output/preprocess/desktop_tennis/richard

python -m dsa.preprocess.desktop_tennis prepare \
  data/raw/tennis_videos/peter_dylan_202609.MOV \
  --opponent dylan --out output/preprocess/desktop_tennis/dylan

python -m dsa.preprocess.desktop_tennis review \
  --out output/preprocess/desktop_tennis/richard --port 8765

python -m dsa.preprocess.desktop_tennis review \
  --out output/preprocess/desktop_tennis/dylan --port 8766
```

Open the printed localhost URL. The server binds only to `127.0.0.1`. Each save
updates `annotations.json`, `scores.json`, and `shots.json`, and preserves the
previous point log in `annotations.previous.json`. Preparation never overwrites
an existing point log. The bundled starting profiles apply only when both the
filename and source SHA-256 match.

## Point edits

Each segment is one `eleven` or a continuous `match`. Each point contains one
or more retained source clips:

1. Start a clip just before the actual feed or struck serve (`I`).
2. For a missed first serve, retain the attempt, then close that clip (`F`).
3. Skip dead time or a caught/aborted toss. Resume a new clip before the second
   struck serve (`I`). Both attempts remain inside **one point**.
4. At the point's end, select the winner (`1` or `2`), or unknown (`U`). The
   editor adds a 0.7 second tail so the point outcome remains visible.

`O` closes an ordinary retained clip without awarding the point. Edit existing
point windows, result times and roles with the point's **Edit** button. A double
fault is one lost point; mark its final clip `double_fault` if desired. A caught
toss never changes the score. The exported clip list is authoritative: footage
outside those windows is omitted, including setup, collection and breaks.

Point and clip times must be ordered and non-overlapping. `time` is when the
point outcome becomes known; it must lie in the point's last retained clip.
The score changes there, including during the tail. `edit_timeline` records
the exact source-to-output frame mapping after cuts.

## Scoring and verification

Eleven scoring is first to 11, win by two. Match scoring supports love / 15 /
30 / 40, deuce, advantage, games and sets. The default set convention is a
7-point, win-by-two tiebreak at 6–6. Best-of-match termination is not inferred.
Initial scores are explicit; a recording can start mid-game. Raw match point
counts of `[3, 3]` mean deuce and `[4, 3]` mean player 0 has advantage.

Unknown initial scores or unknown winners produce `?` instead of invented
numbers. A later known winner cannot repair an earlier unknown outcome;
resolve the earlier point or split at a verified score checkpoint with an
explicit initial score. Winning a first-serve-fault clip does not award a point:
only its parent point has a winner.

Before marking a segment verified, check its initial score, every point winner,
and complete point coverage. Final export refuses unverified segments. An
explicit `--draft` export carries a visible draft badge and keeps unknown scores
unknown. A draft with five logged points exports **only those five points**, not
the remainder of the segment.

```bash
python -m dsa.preprocess.desktop_tennis build \
  --out output/preprocess/desktop_tennis/dylan

python -m dsa.preprocess.desktop_tennis export \
  --out output/preprocess/desktop_tennis/dylan --segment match --draft

# After completing and verifying its point log:
python -m dsa.preprocess.desktop_tennis export \
  --out output/preprocess/desktop_tennis/richard
```

## Downstream analysis

The shared manifest covers every analysis frame exactly once. Verified point
clips are `keep`; gaps inside segments with verified coverage are `discard`;
unreviewed footage is `review`. Point/attempt boundaries remain separate so
tracking resets across cuts. Score-overlay exports are for watching; tracking
reads the clean master and its manifest:

```bash
modal run src/dsa/cloud/modal_app.py::run \
  --video output/preprocess/desktop_tennis/richard/analysis.mp4 \
  --shots output/preprocess/desktop_tennis/richard/shots.json \
  --run-name richard_points
```

The scored workflow does not infer point outcomes automatically. The activity
proposals above need no such labels. Local speech-recognition trials with
Whisper Small English and Large v3 Turbo
on Richard's opening ten minutes failed to recover dependable scores; repeated
phrases appeared during play, so those transcripts were not used as labels.
The recordings still need visual point review for scoring. The test
suite verifies scoring, rejected invalid logs, first-serve retention, dead-time
removal, frame mappings, audio retention and export frame counts.
