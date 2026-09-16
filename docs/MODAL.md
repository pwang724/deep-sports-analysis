# Running on Modal

The pose tracker runs on Modal GPUs with no local hardware. One Volume holds
weights, source clips and outputs. Each video segment runs in its own
container, so a match processes in minutes.

## One-time setup

1. Log in with the personal account (peterwang724@gmail.com), saved as its own profile:

   ```bash
   modal token new --profile personal && modal profile activate personal
   ```

2. Upload the model weights (about 3.8 GB, one time):

   ```bash
   modal run src/dsa/cloud/modal_app.py::setup
   ```

   ViTPose comes from `models/vitpose-plus-huge`; RF-DETR from `~/.roboflow/models`.

3. For the benchmarks, upload the labelled datasets from `data/` (see README):

   ```bash
   modal run src/dsa/cloud/modal_app.py::datasets
   ```

## Track a clip

```bash
modal run src/dsa/cloud/modal_app.py::run --video data/raw/uso2026_final_highlights.mp4 --start 100 --duration 60
```

The clip is uploaded on first use. AV1 clips (the usual YouTube download)
are transcoded to H.264 once on Modal, because OpenCV's Linux wheel cannot
decode AV1; expect a few extra minutes the first time. Options mirror `scripts/joints.py`
(`--stride`, `--threshold`) plus:

| flag | default | meaning |
|---|---|---|
| `--detector` | `rfdetr` | person detector: `rfdetr` (RF-DETR Medium) or `yolo` |
| `--imgsz` | 1152 | detector input size; 1152 for rfdetr, 1280 for yolo |
| `--segment-len` | 15 | seconds per container; more segments means more GPUs in parallel |
| `--run-name` | `<clip>_<start>_<duration>` | output folder name |
| `--out` | `output` | local folder for the merged tables |

Merged `joints.parquet` and `summary.json` land in `output/<run-name>/`. The
annotated video stays on the Volume; the command to fetch it is printed at
the end.

GPUs are tried in order until one has capacity, default `L4,A10,L40S,A100-40GB`.
Override with a comma-separated list:

```bash
DSA_GPU=L40S,A100-40GB modal run src/dsa/cloud/modal_app.py::run --video ...
```

## Benchmarks

Each is its own Modal app and prints a table; results are recorded in
[RESULTS-pose.md](RESULTS-pose.md).

```bash
modal run src/dsa/cloud/bench_detectors.py   # person detectors vs labelled boxes
modal run src/dsa/cloud/bench_tracking.py    # detector + ByteTrack on a clip: coverage, id switches, cost
modal run src/dsa/cloud/eval_pose.py         # pose backends vs labelled joints
```

## How it works

- `src/dsa/cloud/modal_app.py` defines the image (pinned deps from
  `pyproject.toml`), the Volume layout, a `PoseTracker` class that loads the
  detector and ViTPose once per container in fp16, and CPU functions
  `prepare_video` (transcode if needed) and `merge_run`.
- Per frame: detector finds people, ByteTrack assigns IDs, ViTPose estimates
  joints per box. See `dsa/pose/tracking.py`.
- `dsa.pose.segments.plan_segments` splits the requested range; each segment
  is one `PoseTracker.track` call, fanned out with `.map()`.
- Track IDs are offset per segment so they never collide. ByteTrack does not
  carry identity across segment boundaries; stitch by position downstream if
  needed.
- Outputs are committed to the Volume by each container, then merged.

## Volume layout

```
/vol/models/vitpose-plus-huge   ViTPose weights
/vol/models/rfdetr              RF-DETR weights (RF_HOME); downloaded on first use
/vol/models/yolo                YOLO weights, only if the yolo detector is used
/vol/videos/<name>.mp4          source clips (plus <name>.h264.mp4 when transcoded)
/vol/datasets/<name>/           labelled evaluation data
/vol/runs/<run>/segNNNN/        per-segment joints.parquet, summary.json, annotated.mp4
/vol/runs/<run>/                merged outputs
```

Browse or fetch with `modal volume ls deep-sports /runs` and
`modal volume get deep-sports <remote> <local>`.

## Cost

Billing is per second while containers run. Rough L4 rates: about $0.80 per
GPU-hour, and a 60 fps clip at stride 2 processes at roughly real time, so a
10 minute clip costs well under a dollar. The starter plan includes $30 of
compute a month.
