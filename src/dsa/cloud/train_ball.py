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
"""
import json
import os
from pathlib import Path

import modal

from dsa.cloud.modal_app import MODELS_DIR, REPO, RUNS_DIR, VOL, _rel, base_image, volume

TARS = VOL / "datasets" / "tracknet_tennis"
ZIP = TARS / "Dataset.zip"
DRIVE_ID = "1DQ3ZbvokTsgOq6x-ay6O8U2W4a8e3LFw"   # TrackNet tennis Dataset.zip
OUT = RUNS_DIR / "ball_fusion"
VARIANTS = {"9ch": 3, "3ch": 1}      # run name -> frames per sample

app = modal.App("deep-sports-train-ball", image=base_image.add_local_python_source("dsa"))


@app.function(gpu=os.environ.get("DSA_GPU", "L40S,A100-40GB").split(","), volumes={str(VOL): volume},
              cpu=16, memory=65_536, timeout=6 * 60 * 60)
def train_remote(run: str, frames: int, epochs: int, batch: int, lr: float, limit: int | None) -> dict:
    import subprocess
    import threading
    import zipfile

    from dsa.train.ball_fusion import train

    local = Path("/tmp/tracknet")
    subprocess.run(["mkdir", "-p", str(local)], check=True)
    zipfile.ZipFile(ZIP).extractall(local)
    out = OUT / run
    lock = threading.Lock()

    def log(msg: str):
        print(f"[{run}] {msg}", flush=True)
        with lock, open(out / "log.txt", "a") as f:
            f.write(msg + "\n")

    out.mkdir(parents=True, exist_ok=True)
    root = next(local.rglob("game1")).parent       # Dataset/, wherever the zip put it
    summary = train(root, out, frames=frames, epochs=epochs, batch=batch, lr=lr,
                    weights=str(MODELS_DIR / "rfdetr" / "rf-detr-medium.pth"), limit=limit, log=log)
    volume.commit()
    return summary


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


@app.local_entrypoint()
def main(variants: str = "9ch,3ch", epochs: int = 12, batch: int = 16, lr: float = 1e-4,
         limit: int | None = None, tag: str = "", out: str = "output/train/ball_fusion"):
    runs = [(f"{v}{tag}", VARIANTS[v]) for v in variants.split(",")]
    calls = [train_remote.spawn(r, f, epochs, batch, lr, limit) for r, f in runs]
    for (run, _), call in zip(runs, calls):
        summary = call.get()
        dst = REPO / out / run
        dst.mkdir(parents=True, exist_ok=True)
        for e in volume.listdir(_rel(OUT / run)):
            name = Path(e.path).name
            if not name.endswith(".pt"):
                with open(dst / name, "wb") as f:
                    for chunk in volume.read_file(e.path):
                        f.write(chunk)
        print(run, json.dumps(summary["peak>0.5"], indent=2, default=float))
        print(f"wrote {dst}")
