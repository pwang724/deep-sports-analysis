"""Train and score the early-fusion ball model (dsa.train.ball_fusion) on Modal.

TrackNet tennis lives on the volume as the released Dataset.zip
(/vol/datasets/tracknet_tennis/Dataset.zip), fetched from Google Drive by
`fetch` (a 2.6 GB upload from here kept dropping) and unzipped to local disk
in each container. Each run
writes last.pt, history.json, per_frame.parquet, summary.json and a few
visualisations to /vol/runs/ball_fusion/<run>/; `main` copies all but the
checkpoint to output/train/ball_fusion/<run>/.

    modal run src/dsa/cloud/train_ball.py::fetch                  # once
    modal run src/dsa/cloud/train_ball.py                         # 9-ch and 3-ch runs in parallel
    modal run src/dsa/cloud/train_ball.py --variants 9ch --limit 256 --epochs 1   # smoke test

RacketVision (tennis subset) for the 9ch_rv run:

    modal run src/dsa/cloud/train_ball.py::fetch_rv       # Hugging Face -> /vol/datasets/racketvision/tennis
    modal run src/dsa/cloud/train_ball.py::prepare_rv     # labelled frames +- neighbours at 1280 x 720, zipped
    modal run src/dsa/cloud/train_ball.py::main --variants 9ch --rv --tag _rv --epochs 24 --gpu L40S
    modal run src/dsa/cloud/train_ball.py::main --variants 9ch --rv --no-rv-train --tag _24ep --epochs 24 --gpu L40S
    (add --resume to either to continue from its last finished epoch on the volume)
    modal run src/dsa/cloud/train_ball.py::bench --gpus H100,L40S,A100-80GB   # s/step, 60 steps each
    modal run src/dsa/cloud/train_ball.py::evaluate --run 9ch   # an old checkpoint on both test sets
"""
import json
import os
import time
from pathlib import Path

import modal

from dsa.cloud.modal_app import MODELS_DIR, REPO, RUNS_DIR, VOL, _rel, base_image, volume

TARS = VOL / "datasets" / "tracknet_tennis"
ZIP = TARS / "Dataset.zip"
DRIVE_ID = "1DQ3ZbvokTsgOq6x-ay6O8U2W4a8e3LFw"   # TrackNet tennis Dataset.zip
OUT = RUNS_DIR / "ball_fusion"
RV = VOL / "datasets" / "racketvision"   # RacketVision tennis subset, laid out as on Hugging Face (tennis/...)
VARIANTS = {"9ch": 3, "3ch": 1}      # run name -> frames per sample

app = modal.App("deep-sports-train-ball", image=base_image.add_local_python_source("dsa"))


def _unpack(rv: bool) -> Path:
    """TrackNet (and RacketVision frames under <root>/rv/) onto the container's local disk; returns root."""
    import subprocess
    import zipfile

    local = Path("/tmp/tracknet")
    subprocess.run(["mkdir", "-p", str(local)], check=True)
    zipfile.ZipFile(ZIP).extractall(local)
    root = next(local.rglob("game1")).parent       # Dataset/, wherever the zip put it
    if rv:
        (root / "rv").mkdir(exist_ok=True)
        for z in sorted((RV / "frames720").glob("*.zip")):
            zipfile.ZipFile(z).extractall(root / "rv")
        (root / "rv" / "index.csv").write_bytes((RV / "frames720" / "index.csv").read_bytes())
    return root


@app.function(gpu=os.environ.get("DSA_GPU", "L40S,A100-40GB").split(","), volumes={str(VOL): volume},
              cpu=16, memory=65_536, timeout=16 * 60 * 60)
def train_remote(run: str, frames: int, epochs: int, batch: int, lr: float, limit: int | None, rv: bool = False,
                 max_steps: int | None = None, workers: int = 12, rv_train: bool = True,
                 resume: bool = False) -> dict:
    import threading

    from dsa.train.ball_fusion import train

    t0 = time.time()
    root = _unpack(rv)
    out = OUT / run
    lock = threading.Lock()

    def log(msg: str):
        print(f"[{run}] {msg}", flush=True)
        with lock, open(out / "log.txt", "a") as f:
            f.write(msg + "\n")

    out.mkdir(parents=True, exist_ok=True)
    log(f"unpacked in {time.time() - t0:.0f}s")
    summary = train(root, out, frames=frames, epochs=epochs, batch=batch, lr=lr,
                    weights=str(MODELS_DIR / "rfdetr" / "rf-detr-medium.pth"), limit=limit, rv=rv,
                    rv_train=rv_train, max_steps=max_steps, workers=workers, resume=resume, log=log,
                    on_checkpoint=volume.commit)
    summary["container_minutes"] = (time.time() - t0) / 60
    volume.commit()
    return summary


@app.function(gpu=["L40S", "A100-40GB"], volumes={str(VOL): volume}, cpu=16, memory=65_536, timeout=2 * 60 * 60)
def evaluate_remote(run: str, frames: int = 3) -> dict:
    """A trained run's last.pt on TrackNet games 8-10 and RacketVision test -> <run>/eval_both/."""
    from dsa.train.ball_fusion import evaluate_checkpoint

    root = _unpack(True)
    s = evaluate_checkpoint(root, OUT / run / "last.pt", OUT / run / "eval_both", frames,
                            weights=str(MODELS_DIR / "rfdetr" / "rf-detr-medium.pth"))
    volume.commit()
    return s


@app.function(volumes={str(VOL): volume}, cpu=2, memory=4096, timeout=30 * 60, max_containers=64)
def extract_rally(match: str, rally: str, split: str) -> list[dict]:
    """One RacketVision rally: each labelled frame and its neighbours, 1280 x 720 JPEG, zipped to the volume.

    Neighbours are +-round(fps / 30) frames (1 at 25 / 29.97 fps, 2 at 60), about TrackNet's 30 fps motion.
    Labels are scaled from the video's size (1920 x 1080) to 1280 x 720. Returns the index rows.
    """
    import io
    import zipfile

    import cv2
    import pandas as pd

    t = RV / "tennis"
    csv, video = t / "all" / match / "csv" / f"{rally}_ball.csv", t / "videos" / f"{match}_{rally}.mp4"
    if not (csv.exists() and video.exists()):
        print(f"missing {match}_{rally}: csv {csv.exists()} video {video.exists()}")
        return []
    lab = pd.read_csv(csv)
    cap = cv2.VideoCapture(str(video))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vw, vh = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    d = max(1, round(fps / 30))
    clip = f"{match}_{rally}"
    want = {max(0, min(f + o, n - 1)) for f in lab.Frame for o in (-d, 0, d)}
    buf, got, i = io.BytesIO(), set(), 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        while i <= max(want):
            ok, im = cap.read()
            if not ok:
                break
            if i in want:
                im = cv2.resize(im, (1280, 720), interpolation=cv2.INTER_AREA)
                z.writestr(f"{clip}/{i:05d}.jpg", cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
                got.add(i)
            i += 1
    cap.release()
    (RV / "frames720").mkdir(parents=True, exist_ok=True)
    (RV / "frames720" / f"{clip}.zip").write_bytes(buf.getvalue())
    volume.commit()
    rows = []
    for r in lab.itertuples(index=False):
        f = int(r.Frame)
        p, q = max(0, min(f - d, n - 1)), max(0, min(f + d, n - 1))
        if not {f, p, q} <= got:
            continue
        vis = int(r.Visibility)
        rows.append({"clip": f"rv/{clip}", "prev": f"{p:05d}.jpg", "file": f"{f:05d}.jpg", "next": f"{q:05d}.jpg",
                     "vis": vis, "x": float(r.X) * 1280 / vw if vis else float("nan"),
                     "y": float(r.Y) * 720 / vh if vis else float("nan"), "split": split, "fps": fps, "step": d,
                     "src_w": vw, "src_h": vh, "labelled": len(lab)})
    return rows


@app.function(volumes={str(VOL): volume}, timeout=2 * 60 * 60)
def prepare_rv_remote() -> dict:
    """extract_rally over every rally in tennis/info/{train,val,test}.json; writes frames720/index.csv."""
    import pandas as pd

    info = RV / "tennis" / "info"
    jobs = [(m, r, s) for s in ("train", "val", "test") for m, r in json.load(open(info / f"{s}.json"))]
    rows = [row for rs in extract_rally.starmap(jobs) for row in rs]
    volume.reload()
    idx = pd.DataFrame(rows)
    idx.to_csv(RV / "frames720" / "index.csv", index=False)
    volume.commit()
    done = set(idx["clip"].str[3:])
    return {"rallies": len(jobs), "extracted": len(done),
            "missing": [f"{m}_{r}" for m, r, _ in jobs if f"{m}_{r}" not in done],
            "frames_by_split": idx.groupby("split").size().to_dict(),
            "visible_by_split": idx.groupby("split").vis.mean().to_dict(),
            "fps": idx.groupby("fps")["clip"].nunique().to_dict(),
            "size": idx.groupby(["src_w", "src_h"])["clip"].nunique().to_dict().__repr__()}


@app.function(image=base_image.uv_pip_install("gdown==5.2.0").add_local_python_source("dsa"), volumes={str(VOL): volume},
              timeout=60 * 60)
def fetch():
    """Download TrackNet's Dataset.zip (the Drive file in dsa.data.sources) straight to the volume."""
    import gdown

    TARS.mkdir(parents=True, exist_ok=True)
    if not ZIP.exists():
        gdown.download(id=DRIVE_ID, output=str(ZIP), quiet=False)
        volume.commit()
    print(ZIP, ZIP.stat().st_size)


def _hf_token() -> str:
    """The local Hugging Face token, if one is already saved (passed as an argument, never stored on Modal)."""
    f = Path.home() / ".cache/huggingface/token"
    return os.environ.get("HF_TOKEN") or (f.read_text().strip() if f.exists() else "")


@app.function(volumes={str(VOL): volume}, timeout=3 * 60 * 60)
def fetch_racketvision(token: str = "") -> dict:
    """RacketVision's tennis subset (videos, ball CSVs, rackets, splits) to /vol/datasets/racketvision.

    snapshot_download skips files already there, so a rerun resumes; a 429 waits and retries.
    """
    import time

    os.environ |= {"HF_HUB_OFFLINE": "0", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}   # the base image is offline
    from huggingface_hub import snapshot_download

    if token:
        os.environ["HF_TOKEN"] = token
    dst = RV
    dst.mkdir(parents=True, exist_ok=True)
    # what training needs first (splits, ball CSVs, videos), then the rest (7,400 racket files, 429-prone)
    stages = [["tennis/info/*", "tennis/all/*/csv/*"], ["tennis/videos/*"], ["tennis/*"]]
    for patterns in stages:
        for attempt in range(12):
            try:
                snapshot_download("linfeng302/RacketVision", repo_type="dataset", local_dir=str(dst),
                                  allow_patterns=patterns, max_workers=4)
                break
            except Exception as e:      # 429s surface as HfHubHTTPError / OSError after hub-side retries
                wait = min(60 * 2**attempt, 900)
                print(f"{patterns} attempt {attempt}: {type(e).__name__} {str(e)[:200]}; waiting {wait}s", flush=True)
                volume.commit()
                time.sleep(wait)
        volume.commit()
        print(f"done {patterns}", flush=True)
    t = dst / "tennis"
    sizes = {d.name: (sum(1 for _ in d.rglob("*") if _.is_file()),
                      sum(p.stat().st_size for p in d.rglob("*") if p.is_file())) for d in t.iterdir() if d.is_dir()}
    print(json.dumps(sizes, indent=1))
    return sizes


@app.local_entrypoint()
def fetch_rv():
    """modal run src/dsa/cloud/train_ball.py::fetch_rv  (RacketVision tennis to the volume)"""
    print(json.dumps(fetch_racketvision.remote(_hf_token()), indent=1))


def _copy_run(run: str, out: str, sub: str = "") -> Path:
    """Everything but checkpoints from /vol/runs/ball_fusion/<run>[/sub] to <out>/<run>[/sub]."""
    dst = REPO / out / run / sub
    dst.mkdir(parents=True, exist_ok=True)
    for e in volume.listdir(_rel(OUT / run / sub)):
        name = Path(e.path).name
        if not name.endswith(".pt") and e.type == modal.volume.FileEntryType.FILE:
            with open(dst / name, "wb") as f:
                for chunk in volume.read_file(e.path):
                    f.write(chunk)
    return dst


@app.local_entrypoint()
def prepare_rv():
    print(json.dumps(prepare_rv_remote.remote(), indent=1, default=str))


@app.local_entrypoint()
def bench(gpus: str = "H100,L40S,A100-80GB", steps: int = 60, workers: int = 12, rv: bool = True, cpu: int = 16):
    """~50 timed training steps (after 10 warm-up) of the 9-channel model per GPU type, in parallel."""
    calls = {g: train_remote.with_options(gpu=g, cpu=cpu).spawn(f"bench_{g}_w{workers}", 3, 24, 16, 1e-4, None, rv, steps,
                                                       workers) for g in gpus.split(",")}
    for g, c in calls.items():
        print(g, json.dumps(c.get(), default=float))


@app.local_entrypoint()
def evaluate(run: str = "9ch", out: str = "output/train/ball_fusion"):
    s = evaluate_remote.remote(run)
    print(json.dumps({"tracknet": s["peak>0.5"], "racketvision_test": s["racketvision_test"]["peak>0.5"]},
                     indent=1, default=float))
    print("wrote", _copy_run(run, out, "eval_both"))


@app.local_entrypoint()
def main(variants: str = "9ch,3ch", epochs: int = 12, batch: int = 16, lr: float = 1e-4,
         limit: int | None = None, tag: str = "", out: str = "output/train/ball_fusion", rv: bool = False,
         gpu: str = "", workers: int = 12, cpu: int = 16, rv_train: bool = True, resume: bool = False):
    """--rv scores RacketVision test as well; it also trains on RacketVision train unless --no-rv-train.

    --resume continues each run from its last finished epoch on the volume (same flags as the first launch).
    """
    runs = [(f"{v}{tag}", VARIANTS[v]) for v in variants.split(",")]
    fn = train_remote.with_options(**({"gpu": gpu} if gpu else {}), cpu=cpu)   # one GPU type, e.g. H100
    calls = [fn.spawn(r, f, epochs, batch, lr, limit, rv, None, workers, rv_train, resume) for r, f in runs]
    for (run, _), call in zip(runs, calls):
        summary = call.get()
        dst = _copy_run(run, out)
        print(run, json.dumps(summary["peak>0.5"], indent=2, default=float))
        if rv:
            print(run, "racketvision test", json.dumps(summary["racketvision_test"]["peak>0.5"], indent=2, default=float))
        print(f"wrote {dst}")
