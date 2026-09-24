"""Run the pose tracker on Modal GPUs, one container per video segment.

Everything persistent lives in one Modal Volume:

    /vol/models/vitpose-plus-huge   ViTPose weights (uploaded once by `setup`)
    /vol/models/rfdetr              RF-DETR weights (downloaded on first use, via RF_HOME)
    /vol/models/yolo                YOLO weights, only if that detector is used
    /vol/videos/<name>.mp4          source clips (plus <name>.h264.mp4 when transcoded)
    /vol/datasets/<name>/           labelled data for the benchmarks
    /vol/runs/<run>/segNNNN/        per-segment outputs
    /vol/runs/<run>/                merged joints.parquet, summary.json, annotated.mp4

First time:   modal run src/dsa/cloud/modal_app.py::setup       (model weights)
              modal run src/dsa/cloud/modal_app.py::datasets    (labelled data, for the benchmarks)
Track a clip: modal run src/dsa/cloud/modal_app.py::run --video data/raw/clip.mp4 --start 100 --duration 60
"""

import json
import os
import tomllib
from pathlib import Path

import modal

# Local repo root; only meaningful on the machine that launches the app.
REPO = Path(__file__).resolve().parents[3] if modal.is_local() else None
VOLUME_NAME = "deep-sports"
VOL = Path("/vol")
MODELS_DIR, VIDEOS_DIR, RUNS_DIR = VOL / "models", VOL / "videos", VOL / "runs"
VITPOSE_DIR = MODELS_DIR / "vitpose-plus-huge"
# First choice, then fallbacks if Modal has no capacity. Override with DSA_GPU=L40S or DSA_GPU=L4,A10.
GPU = os.environ.get("DSA_GPU", "L4,A10,L40S,A100-40GB").split(",")


def _pinned_dependencies() -> list[str]:
    """Runtime deps from pyproject.toml so local and cloud environments match. Read locally only."""
    if not modal.is_local():
        return []
    return tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["dependencies"]


volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Dependencies only; other apps extend this. Local source must be added last, so each app does that itself.
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "ffmpeg")
    .uv_pip_install(*_pinned_dependencies())
    .env({"RF_HOME": str(MODELS_DIR / "rfdetr"), "HF_HUB_OFFLINE": "1"})
)
image = base_image.add_local_python_source("dsa")

app = modal.App("deep-sports-pose", image=image)


@app.cls(gpu=GPU, volumes={str(VOL): volume}, cpu=4, memory=16_384, timeout=60 * 60)
class PoseTracker:
    """Detector and ViTPose, loaded once per container in fp16 and reused across segments."""

    detector: str = modal.parameter(default="rfdetr")
    imgsz: int = modal.parameter(default=1152)

    @modal.enter()
    def load(self):
        import torch

        from dsa.pose.backends import load_models

        self.models = load_models(self.detector, self.imgsz, VITPOSE_DIR, device="cuda", dtype=torch.float16)

    @modal.method()
    def track(self, video_name: str, run: str, segment: dict, config: dict) -> dict:
        from dsa.pose.segments import Segment
        from dsa.pose.tracking import TrackConfig, track_segment

        seg = Segment(**segment)
        out_dir = RUNS_DIR / run / seg.name
        summary = track_segment(VIDEOS_DIR / video_name, seg.start, seg.duration, out_dir,
                                self.models, TrackConfig(**config), seg.track_id_offset,
                                (seg.start_frame, seg.end_frame) if seg.start_frame is not None else None)
        volume.commit()
        return summary


@app.function(volumes={str(VOL): volume}, cpu=8, timeout=60 * 60)
def prepare_video(name: str) -> str:
    """Make sure a decodable copy of the clip exists on the volume; return its file name.

    OpenCV's Linux wheel cannot decode AV1 (what YouTube downloads usually
    are), so such clips are transcoded to H.264 once with the system ffmpeg.
    """
    from dsa.video import CV2_DECODABLE, codec_of, transcode_h264

    src = VIDEOS_DIR / name
    if codec_of(src) in CV2_DECODABLE:
        return name
    dst = src.with_name(f"{src.stem}.h264.mp4")
    if not dst.exists():
        print(f"transcoding {name} to H.264...", flush=True)
        transcode_h264(src, dst)
        volume.commit()
    return dst.name


@app.function(volumes={str(VOL): volume}, cpu=2, timeout=30 * 60)
def merge_run(run: str, segment_names: list[str]) -> dict:
    """Merge per-segment outputs into the run directory on the volume."""
    from dsa.pose.segments import merge_tables
    from dsa.video import concat_videos

    volume.reload()
    run_dir = RUNS_DIR / run
    seg_dirs = [run_dir / n for n in segment_names]
    summary = merge_tables(seg_dirs, run_dir)
    concat_videos([d / "annotated.mp4" for d in seg_dirs], run_dir / "annotated.mp4")
    volume.commit()
    return summary


def _rel(path: Path) -> str:
    """Container mount path -> path relative to the volume root, as the client API expects."""
    return "/" + str(path.relative_to(VOL))


def _volume_has(path: Path) -> bool:
    try:
        volume.listdir(_rel(path))
        return True
    except modal.exception.NotFoundError:
        return False


def _upload(local: Path, remote: Path) -> None:
    """Copy a local file or directory into the volume at the given mount path."""
    with volume.batch_upload() as batch:
        if local.is_dir():
            batch.put_directory(str(local), _rel(remote))
        else:
            batch.put_file(str(local), _rel(remote))


@app.local_entrypoint()
def setup(vitpose: str = "models/vitpose-plus-huge", rfdetr_cache: str = "~/.roboflow/models"):
    """Upload model weights to the volume. Safe to rerun; existing files are skipped."""
    if _volume_has(VITPOSE_DIR):
        print(f"{VITPOSE_DIR} already on volume")
    else:
        print(f"uploading {vitpose} (3.6 GB)...")
        _upload(REPO / vitpose, VITPOSE_DIR)
    for pth in Path(rfdetr_cache).expanduser().glob("*.pth"):
        remote = MODELS_DIR / "rfdetr" / pth.name
        if _volume_has(remote):
            print(f"{remote} already on volume")
        else:
            print(f"uploading {pth.name}...")
            _upload(pth, remote)
    print("done")


@app.local_entrypoint()
def datasets(root: str = "data/sources"):
    """Upload the labelled datasets used by the benchmarks. Safe to rerun; existing ones are skipped."""
    for name, subdir in (("tennis_segmentation", ""), ("tennis_player_actions", "Tennis Player Actions Dataset for Human Pose Estimation")):
        local = REPO / root / name / subdir
        remote = VOL / "datasets" / name
        if not local.exists():
            print(f"{local} not found locally, skipping")
        elif _volume_has(remote):
            print(f"{remote} already on volume")
        else:
            print(f"uploading {name}...")
            _upload(local, remote)
    print("done")


@app.local_entrypoint()
def run(video: str, start: float = 0.0, duration: float = 30.0, segment_len: float = 15.0,
        detector: str = "rfdetr", imgsz: int = 1152, stride: int = 2, threshold: float = 0.4,
        run_name: str | None = None, out: str = "output", shots: str = ""):
    """Track players over [start, start + duration), fanned out across GPUs by segment.

    The video is uploaded to the volume if missing. Merged joints.parquet and
    summary.json are downloaded into `<out>/<run_name>/`; the annotated video
    stays on the volume (see the printed command to fetch it).
    With --shots, use every kept shot in the manifest instead of start/duration/
    segment_len. Each shot gets a fresh tracker and retains source timestamps.
    """
    from dsa.pose.detectors import DETECTORS
    from dsa.pose.segments import plan_kept_shots, plan_segments
    from dsa.pose.tracking import TrackConfig

    if detector not in DETECTORS:
        raise SystemExit(f"unknown detector {detector!r}; choose from {sorted(DETECTORS)}")

    local_video = Path(video)
    manifest = json.loads(Path(shots).read_text()) if shots else None
    segments = plan_kept_shots(manifest, local_video) if manifest else plan_segments(start, duration, segment_len)
    name = local_video.name
    if not _volume_has(VIDEOS_DIR / name):
        print(f"uploading {name}...")
        _upload(local_video, VIDEOS_DIR / name)

    name = prepare_video.remote(name)

    run_name = run_name or (f"{local_video.stem}_kept" if manifest else f"{local_video.stem}_{int(start)}_{int(duration)}")
    cfg = TrackConfig(stride=stride, threshold=threshold)
    print(f"run {run_name}: {len(segments)} {'kept shots' if manifest else 'segments'} on {GPU[0]} "
          f"(fallbacks {GPU[1:]}), detector {detector}@{imgsz}")

    tracker = PoseTracker(detector=detector, imgsz=imgsz)
    for s in tracker.track.map([name] * len(segments), [run_name] * len(segments),
                               [vars(seg) for seg in segments], [vars(cfg)] * len(segments)):
        spf = s["sec_per_frame"]
        print(f"  {s['start']:7.1f}s  {s['frames_processed']:5d} frames  detect {spf['detect']:.3f}s/f  vitpose {spf['vitpose']:.3f}s/f")

    summary = merge_run.remote(run_name, [seg.name for seg in segments])

    out_dir = Path(out) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("joints.parquet", "summary.json"):
        with open(out_dir / fname, "wb") as f:
            for chunk in volume.read_file(_rel(RUNS_DIR / run_name / fname)):
                f.write(chunk)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_dir}/joints.parquet and summary.json")
    print(f"annotated video: modal volume get {VOLUME_NAME} {_rel(RUNS_DIR / run_name / 'annotated.mp4')} {out_dir}/annotated.mp4")
