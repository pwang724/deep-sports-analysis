# USO 2026 highlights cleanup

This recipe is only for `uso2026_final_highlights.mp4`. Keep the high end-on
court view; remove crowd, close-ups, graphics and other camera angles. It is
calibrated on this clip and is not a general tennis preprocessing pipeline.

## Run from the repository root

```bash
uv pip install -e ".[preprocess]"
python -m dsa.preprocess.uso2026_highlights \
  data/raw/uso2026_final_highlights.mp4 --export
```

Output goes to `output/preprocess/uso2026_highlights/`. Use `--out` to change it,
`--device` to select inference hardware, or `--export-only` to export an existing
`shots.json` without inference. Previous results in `output/view_filter/` remain
usable and have not been deleted.

```bash
modal run src/dsa/cloud/modal_app.py::run \
  --video data/raw/uso2026_final_highlights.mp4 \
  --shots output/preprocess/uso2026_highlights/shots.json --run-name aerial_shots
```

## This clip's method

`pipeline.py` contains the whole selection algorithm. `profile.json` holds this
clip's 12 example timestamps, similarity margin, sampling rate and cut thresholds.
The bundled profile is checked against the source filename and SHA-256.

Frozen DINOv2 Small compares full-frame patch embeddings with keep/discard
references. Samples at 2 fps plus shot samples locate changes in view; detected
camera cuts are retained, and observed label changes are refined between samples.
Uncertain samples are withheld as `review`. Reference snapshots, decision contact
sheets and cached embeddings are saved alongside `shots.json`.

The calibrated result has 30 kept intervals / 393.3263 seconds, 96 discarded /
308.9587 seconds, and 53 review / 30.8475 seconds, covering all 43,944 frames.
This was inspected and tuned on this clip, not tested on a separate match.
Dissolve tails can pass, brief inserts can be missed, and replays from the desired
angle pass. Player identity and pose analysis are separate downstream stages.

A new video's method belongs in its own directory. It need not use any of these
models, thresholds, references, or refinement steps; only the output manifest
must follow the shared contract.
