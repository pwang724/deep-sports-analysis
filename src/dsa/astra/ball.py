"""Astra vs WASB as a ball labeler, scored on TrackNet tennis.

TrackNet tennis: 1280 x 720 broadcast clips at 30 fps, one ball label per frame
(visibility 0 = none, 1 = visible, 2 = hard to see, 3 = occluded). WASB's test
split, games 8-10, is sampled: 80 frames with the ball visible (1, 2) and 20
without (0). Both labelers see frames t-1, t, t+1 and locate the ball in t.

  wasb   WASB-SBDT (NTT, MIT) HRNet with its tennis weights, run here without
         its CUDA-only runner: frames resized to 512 x 288, ImageNet-normalised,
         stacked to 9 channels; the highest-scoring heatmap blob above 0.5 in
         the middle output frame, mapped back to 1280 x 720. No tracker.
  astra  the three full frames as three images.

A labelled-visible ball counts as found within 4 px (WASB's threshold) or
10 px. Precision, recall and F1 as in WASB: a detection far from the label is
both a false positive and a miss.

    python -m dsa.astra.ball --n 100
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from dsa.astra.codex import ask
from dsa.data.paths import SOURCES, CODE

ROOT = SOURCES / "tracknet/TrackNet/Dataset"
WASB = CODE / "wasb"
TEST_GAMES = ("game8", "game9", "game10")

SCHEMA = {
    "type": "object",
    "properties": {"visible": {"type": "boolean"}, "x": {"type": "number"}, "y": {"type": "number"}},
    "required": ["visible", "x", "y"],
    "additionalProperties": False,
}
PROMPT = (
    "The three attached images are consecutive frames (1280 x 720 pixels, 1/30 s apart) of a tennis "
    "broadcast. Find the tennis ball in the second image. It is small, a few pixels across, and may be "
    "blurred into a short streak; use the first and third images to see which small object is moving. "
    "Give the pixel coordinates of its centre (x from the left edge, y from the top edge) in the second "
    "image. Set visible to false if the ball is not in the second image (x and y then 0). Ignore balls "
    "held by ball kids or players and balls lying still on the ground. Look at the images directly; do "
    "not run any commands."
)


def sample(n: int, seed: int = 0) -> list[dict]:
    rows = []
    for game in TEST_GAMES:
        for clip in sorted((ROOT / game).iterdir()):
            labels = pd.read_csv(clip / "Label.csv")
            names = labels["file name"].tolist()
            for i in range(1, len(labels) - 1):
                r = labels.iloc[i]
                rows.append({"clip": str(clip), "prev": names[i - 1], "file": names[i], "next": names[i + 1],
                             "vis": int(r.visibility), "x": float(r["x-coordinate"] or 0), "y": float(r["y-coordinate"] or 0)})
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    vis = df[df.vis.isin([1, 2])]
    none = df[df.vis == 0]
    k = round(0.8 * n)
    return pd.concat([vis.iloc[rng.choice(len(vis), k, replace=False)],
                      none.iloc[rng.choice(len(none), n - k, replace=False)]]).to_dict("records")


class _AttrDict(dict):
    """The OmegaConf access WASB's HRNet expects: keys as attributes too."""

    __getattr__ = dict.__getitem__

    @classmethod
    def wrap(cls, v):
        return cls({k: cls.wrap(x) for k, x in v.items()}) if isinstance(v, dict) else v


class Wasb:
    def __init__(self, device: str):
        import torch

        sys.path.insert(0, str(WASB / "src"))
        from models import build_model

        cfg = {"model": _AttrDict.wrap(yaml.safe_load(open(WASB / "src/configs/model/wasb.yaml")))}
        self.model = build_model(cfg)
        state = torch.load(WASB / "pretrained_weights/wasb_tennis_best.pth.tar", map_location="cpu")["model_state_dict"]
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        self.model.to(device).eval()
        self.device, self.torch = device, torch
        self.mean, self.std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    def locate(self, frames: list[np.ndarray]) -> tuple[bool, float, float, float]:
        H, W = frames[0].shape[:2]
        x = np.concatenate([(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (512, 288)) / 255.0 - self.mean) / self.std
                            for f in frames], axis=2)
        t = self.torch.from_numpy(x.transpose(2, 0, 1)[None]).float().to(self.device)
        with self.torch.no_grad():
            hm = self.model(t)[0][0, 1].sigmoid().cpu().numpy()
        if hm.max() <= 0.5:
            return False, 0.0, 0.0, float(hm.max())
        n, labels = cv2.connectedComponents((hm > 0.5).astype(np.uint8))
        best = max(range(1, n), key=lambda m: hm[labels == m].sum())
        ys, xs = np.nonzero(labels == best)
        w = hm[ys, xs]
        return True, float((xs * w).sum() / w.sum() * W / 512), float((ys * w).sum() / w.sum() * H / 288), float(w.sum())


def outcome(truth: dict, visible: bool, x: float, y: float, px: float) -> dict:
    labelled = truth["vis"] in (1, 2)
    d = float(np.hypot(x - truth["x"], y - truth["y"])) if labelled and visible else np.nan
    hit = labelled and visible and d <= px
    return {"tp": hit, "fp": visible and not hit, "fn": labelled and not hit}


def prf(g: pd.DataFrame, px: int) -> dict:
    tp, fp, fn = g[f"tp{px}"].sum(), g[f"fp{px}"].sum(), g[f"fn{px}"].sum()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {f"precision@{px}": p, f"recall@{px}": r, f"f1@{px}": 2 * p * r / max(p + r, 1e-9)}


def main() -> None:
    import torch

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out", default="output/astra_eval/ball")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = sample(a.n)
    print(f"{len(items)} frames, {sum(it['vis'] in (1, 2) for it in items)} with a visible ball", flush=True)

    wasb = Wasb(a.device)
    rows = []
    for it in items:
        frames = [cv2.imread(str(Path(it["clip"]) / it[k])) for k in ("prev", "file", "next")]
        vis, x, y, score = wasb.locate(frames)
        rows.append({**it, "labeler": "wasb", "pred_visible": vis, "px": x, "py": y, "score": score})

    def ask_one(it):
        paths = [(Path(it["clip"]) / it[k]).resolve() for k in ("prev", "file", "next")]
        ans, sec = ask(PROMPT, paths, SCHEMA, out / "cache")
        return {**it, "labeler": "astra", "pred_visible": bool(ans["visible"]), "px": ans["x"], "py": ans["y"], "sec": sec}

    with ThreadPoolExecutor(a.workers) as pool:
        rows += list(pool.map(ask_one, items))

    df = pd.DataFrame(rows)
    for px in (4, 10):
        o = pd.DataFrame([outcome(r, r["pred_visible"], r["px"], r["py"], px) for r in rows])
        df[[f"tp{px}", f"fp{px}", f"fn{px}"]] = o.to_numpy()
    df["dist"] = np.where(df.vis.isin([1, 2]) & df.pred_visible, np.hypot(df.px - df.x, df.py - df.y), np.nan)
    df.to_parquet(out / "per_frame.parquet", index=False)

    summary = {}
    for lab, g in df.groupby("labeler"):
        summary[lab] = {**prf(g, 4), **prf(g, 10), "median_dist_px": float(g.dist.median()),
                        "said_visible_when_none": int((g.pred_visible & (g.vis == 0)).sum())}
    new = df[(df.labeler == "astra") & (df.get("sec", 0) > 0)].sec
    summary["astra"]["sec_per_call"] = float(new.mean()) if len(new) else None
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
