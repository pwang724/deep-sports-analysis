"""Astra vs TennisCourtDetector as a court-point labeler, scored on its val split.

TennisCourtDetector (yastrebksv, MIT): 1280 x 720 broadcast frames, 14 court
line intersections per frame (order below, from its court_reference.py).
data_val.json is held out from its training. 100 frames are sampled.

  tcd    its released model (TrackNet-style heatmaps at 640 x 360) with both
         of its post-processing steps, line-intersection refinement and a
         homography fit to the reference court: its best setting.
  astra  the frame plus a numbered diagram of the 14 points.

A labelled point inside the frame counts as found within 7 px (TCD's own
threshold). Also median error and within 15 px. A predicted point for a
labelled point outside the frame is ignored.

    python -m dsa.astra.court --n 100
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

from dsa.astra.codex import ask

ROOT = Path("data/tennis_court_detector/data")
TCD = Path("data/tcd_repo")
WEIGHTS = Path("models/tcd/model_tennis_court_det.pt")
W, H = 1280, 720

POINTS = [
    "far baseline, left doubles corner", "far baseline, right doubles corner",
    "near baseline, left doubles corner", "near baseline, right doubles corner",
    "left singles sideline at the far baseline", "left singles sideline at the near baseline",
    "right singles sideline at the far baseline", "right singles sideline at the near baseline",
    "far service line at the left singles sideline", "far service line at the right singles sideline",
    "near service line at the left singles sideline", "near service line at the right singles sideline",
    "far T (centre service line meets far service line)", "near T (centre service line meets near service line)",
]
# Reference-court coordinates of the 14 points (court_reference.py), for the diagram.
REF = [(286, 561), (1379, 561), (286, 2935), (1379, 2935), (423, 561), (423, 2935), (1242, 561),
       (1242, 2935), (423, 1110), (1242, 1110), (423, 2386), (1242, 2386), (832, 1110), (832, 2386)]

SCHEMA = {
    "type": "object",
    "properties": {"points": {"type": "array", "minItems": 14, "maxItems": 14, "items": {
        "type": "object",
        "properties": {"in_view": {"type": "boolean"}, "x": {"type": "number"}, "y": {"type": "number"}},
        "required": ["in_view", "x", "y"], "additionalProperties": False}}},
    "required": ["points"],
    "additionalProperties": False,
}
PROMPT = (
    "The first image is a 1280 x 720 frame of a tennis broadcast. The second is a diagram of a tennis "
    "court seen from above, the near baseline at the bottom, with 14 numbered points where court lines "
    "cross. Return the pixel coordinates (x from the left edge, y from the top edge) of each point in "
    "the first image, in order 1 to 14:\n"
    + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(POINTS))
    + "\nLeft and right are as seen from the near baseline, looking at the far one. Place each point at "
    "the centre of the crossing lines, as precisely as you can. Set in_view to false for a point outside "
    "the frame or hidden (x and y then your best estimate). Look at the images directly; do not run any "
    "commands."
)


def diagram(path: Path) -> None:
    s = 0.25
    img = np.full((int(3484 * s), int(1665 * s), 3), (60, 120, 60), np.uint8)
    ref = [(int(x * s), int(y * s)) for x, y in REF]
    lines = [(0, 1), (2, 3), (0, 2), (1, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]
    for a, b in lines:
        cv2.line(img, ref[a], ref[b], (255, 255, 255), 2)
    cv2.line(img, (ref[0][0], int(1748 * s)), (ref[1][0], int(1748 * s)), (200, 200, 200), 1)
    for i, p in enumerate(ref):
        cv2.circle(img, p, 6, (0, 0, 255), -1)
        cv2.putText(img, str(i + 1), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(str(path), img)


def sample(n: int, seed: int = 0) -> list[dict]:
    val = json.load(open(ROOT / "data_val.json"))
    have = {p.stem for p in (ROOT / "images").glob("*.png") if not p.name.startswith("._")}
    val = [v for v in val if v["id"] in have]
    rng = np.random.default_rng(seed)
    return [val[i] for i in sorted(rng.choice(len(val), min(n, len(val)), replace=False))]


class Tcd:
    def __init__(self, device: str):
        import torch

        sys.path.insert(0, str(TCD))
        from tracknet import BallTrackerNet

        self.model = BallTrackerNet(out_channels=15)
        self.model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
        self.model.to(device).eval()
        self.device, self.torch = device, torch

    def locate(self, bgr: np.ndarray) -> np.ndarray:
        """(14, 2) points in pixels, NaN where none is found."""
        from postprocess import postprocess, refine_kps

        inp = cv2.resize(bgr, (640, 360)).astype(np.float32) / 255.0
        t = self.torch.from_numpy(inp.transpose(2, 0, 1)[None]).to(self.device)
        with self.torch.no_grad():
            pred = self.torch.sigmoid(self.model(t)[0]).cpu().numpy()
        pts = []
        for k in range(14):
            x, y = postprocess((pred[k] * 255).astype(np.uint8), low_thresh=170, max_radius=25)
            if k not in (8, 9, 12) and x and y:
                x, y = refine_kps(bgr, int(y), int(x))
            pts.append((x, y))
        pts = np.array([(np.nan, np.nan) if x is None else (x, y) for x, y in pts], float)
        m = fit_court(pts)
        return pts if m is None else cv2.perspectiveTransform(np.float32(REF)[:, None], m).reshape(14, 2)


def fit_court(pts: np.ndarray) -> np.ndarray | None:
    """TCD's homography step (homography.get_trans_matrix, which breaks on NumPy 2 /
    new SciPy): fit each of its 12 four-point court configurations and keep the one
    whose projection lies closest, on average, to the other detected points."""
    sys.path.insert(0, str(TCD))
    from homography import court_conf_ind

    best, best_d = None, np.inf
    for inds in court_conf_ind.values():
        if np.isnan(pts[inds]).any():
            continue
        m, _ = cv2.findHomography(np.float32(REF)[inds], np.float32(pts[inds]), method=0)
        if m is None:
            continue
        proj = cv2.perspectiveTransform(np.float32(REF)[:, None], m).reshape(14, 2)
        others = [i for i in range(12) if i not in inds and not np.isnan(pts[i]).any()]
        d = np.mean(np.linalg.norm(proj[others] - pts[others], axis=1)) if others else np.inf
        if d < best_d:
            best, best_d = m, d
    return best


def score(name: str, xy: np.ndarray, item: dict) -> dict:
    gt = np.array(item["kps"], float)
    inside = (gt[:, 0] >= 0) & (gt[:, 0] < W) & (gt[:, 1] >= 0) & (gt[:, 1] < H)
    d = np.linalg.norm(xy - gt, axis=1)
    d = np.where(np.isnan(d), np.inf, d)[inside]
    return {"id": item["id"], "labeler": name, "n_points": int(inside.sum()), "median_err": float(np.median(d)),
            "within7": float((d <= 7).mean()), "within15": float((d <= 15).mean()),
            **{f"err{k}": float(e) for k, e in zip(np.flatnonzero(inside), d)}}


def main() -> None:
    import torch

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--out", default="output/astra_eval/court")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    diagram(out / "diagram.png")
    items = sample(a.n)
    print(f"{len(items)} frames", flush=True)

    tcd = Tcd(a.device)
    rows, preds = [], {}
    for it in items:
        xy = tcd.locate(cv2.imread(str(ROOT / "images" / f"{it['id']}.png")))
        preds[("tcd", it["id"])] = xy
        rows.append(score("tcd", xy, it))

    def ask_one(it):
        img = (ROOT / "images" / f"{it['id']}.png").resolve()
        ans, sec = ask(PROMPT, [img, (out / "diagram.png").resolve()], SCHEMA, out / "cache")
        xy = np.array([[q["x"], q["y"]] for q in ans["points"]], float)
        return it, xy, sec

    with ThreadPoolExecutor(a.workers) as pool:
        for it, xy, sec in pool.map(ask_one, items):
            preds[("astra", it["id"])] = xy
            rows.append({**score("astra", xy, it), "sec": sec})

    df = pd.DataFrame(rows)
    df.to_parquet(out / "per_frame.parquet", index=False)
    np.save(out / "preds.npy", {f"{k[0]}/{k[1]}": v for k, v in preds.items()}, allow_pickle=True)
    summary = {}
    for lab, g in df.groupby("labeler"):
        errs = pd.concat([g[f"err{k}"] for k in range(14) if f"err{k}" in g]).dropna()
        summary[lab] = {"frames": len(g), "points": int(len(errs)), "within7": float((errs <= 7).mean()),
                        "within15": float((errs <= 15).mean()), "median_err": float(errs.median()),
                        "frames_all_within15": float((g.within15 == 1).mean()),
                        "per_point_median": [float(g[f"err{k}"].median()) for k in range(14)]}
    new = df[(df.labeler == "astra") & (df.sec > 0)].sec
    summary["astra"]["sec_per_call"] = float(new.mean()) if len(new) else None
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
