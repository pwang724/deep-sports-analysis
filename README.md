# Deep Sports Analysis

Tennis video analysis from broadcast footage. See [docs/PLAN.md](docs/PLAN.md)
for the pipeline and [docs/RESULTS-pose.md](docs/RESULTS-pose.md) for what
the pose backends score.

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
    eval_pose.py    pose scoring against labelled joints
  data/             dataset loaders (TennisSegmentation)
  video.py          ffmpeg helpers
  cloud/
    modal_app.py        run the tracker on Modal GPUs (docs/MODAL.md)
    bench_detectors.py  detector benchmark on labelled frames
    eval_pose.py        pose evaluation on labelled joints
scripts/            command-line entry points (joints, shots, eval_gt)
tests/              unit tests, no models or GPU needed
```

## Setup

```bash
uv venv --python 3.11 && uv pip install -e ".[cloud,dev]"
```

Download ViTPose-Plus-Huge into `models/vitpose-plus-huge`. RF-DETR weights
download themselves on first use.

## Use

```bash
python scripts/joints.py data/raw/clip.mp4 --start 60 --duration 20 --stride 2   # local (MPS/CPU)
modal run src/dsa/cloud/modal_app.py::run --video data/raw/clip.mp4 --start 60 --duration 20   # GPU
modal run src/dsa/cloud/bench_detectors.py   # detector benchmark
modal run src/dsa/cloud/eval_pose.py         # pose evaluation
pytest
```
