# Deep Sports Analysis

Tennis video analysis from broadcast footage. See [docs/PLAN.md](docs/PLAN.md)
for the pipeline, [docs/DECISIONS.md](docs/DECISIONS.md) for the settled
design choices, and [docs/RESULTS-pose.md](docs/RESULTS-pose.md) for the
measurements behind them.

## Layout

```
src/dsa/            importable package
  pose/
    detectors.py    person detectors behind one interface (RF-DETR Medium, YOLO)
    backends.py     detector + ViTPose loading, ViTPose inference
    tracking.py     track_segment: detect, ByteTrack, pose; video in, joints table out
    segments.py     split a clip for parallel runs, merge results
    skeleton.py     COCO-17 names, edges, drawing
    bench.py        detector scoring against labelled boxes
    track_metrics.py tracker scoring on an unlabelled clip (coverage, id switches)
    eval_pose.py    pose scoring against labelled joints
  data/             dataset loaders (TennisSegmentation)
  video.py          ffmpeg helpers
  preprocess/
    uso2026_highlights/  this video's cleanup code and settings
    desktop_tennis/     phone-video normalization, point edits and scored exports
    media.py            shared video I/O
    manifest.py         retained source intervals consumed by tracking
  cloud/
    modal_app.py        run the tracker on Modal GPUs (docs/MODAL.md)
    bench_detectors.py  detector benchmark on labelled frames
    bench_tracking.py   tracker benchmark on a clip
    eval_pose.py        pose evaluation on labelled joints
scripts/            command-line entry points (joints, shots, eval_gt)
tests/              unit tests, no models or GPU needed
```

## Setup

```bash
uv venv --python 3.11 && uv pip install -e ".[cloud,dev]"
```

Download ViTPose-Plus-Huge into `models/vitpose-plus-huge`. RF-DETR weights
download themselves on first use. For the benchmarks, put TennisSegmentation
(Hugging Face, `julia-wenkmann/TennisSegmentation`) under
`data/sources/tennis_segmentation` and Tennis Player Actions (Mendeley) under
`data/sources/tennis_player_actions`; see docs/DATASETS.md. The layout of
`data/` is in `src/dsa/data/paths.py`.

`data/`, `output/` and `models/` are large and git-ignored. On Peter's Mac
they are symlinks into `/Volumes/PW_SSD/deep-sports-analysis/` (exFAT, so no
symlinks or virtualenvs inside), and the SSD must be connected to run
anything.

## Use

Each video owns its preprocessing code: cut away footage that is not useful
before running analysis. See [preprocessing](docs/PREPROCESSING.md) and the
[USO highlights recipe](src/dsa/preprocess/uso2026_highlights/README.md). The plan for replacing
the pipeline with one trained model is in [docs/training](docs/training/README.md).

```bash
python scripts/joints.py data/raw/clip.mp4 --start 60 --duration 20 --stride 2   # local (MPS/CPU)
modal run src/dsa/cloud/modal_app.py::run --video data/raw/clip.mp4 --start 60 --duration 20   # GPU
modal run src/dsa/cloud/bench_detectors.py   # detector benchmark, labelled frames
modal run src/dsa/cloud/bench_tracking.py    # tracker benchmark, unlabelled clip
modal run src/dsa/cloud/eval_pose.py         # pose evaluation, labelled joints
pytest
```
