"""Astra as a foot labeler, scored on COCO-WholeBody.

COCO-WholeBody (CC BY-NC 4.0) adds six hand-labelled foot points per person to
COCO: big toe, small toe and heel on each foot. A fixed sample of people with
all six visible and a box at least 150 px tall (about a near tennis player) is
cropped and upscaled as for joints; Astra returns both ankles and the six foot
points. Scored with OKS (COCO-WholeBody foot sigmas) and PCK against:

  ankle copy   every foot point placed on the labelled ankle of its side: the
               score of knowing where the foot is but nothing about its shape

Images are fetched one by one from images.cocodataset.org into
data/coco_wholebody/val2017. Annotations: coco_wholebody_val_v1.0.json from
github.com/jin-s13/COCO-WholeBody.

    python -m dsa.astra.feet --n 100
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dsa.astra.codex import ask
from dsa.astra.joints import upscaled_crop
from dsa.pose.eval_pose import crop_box

ROOT = Path("data/coco_wholebody")
FEET = ["l_big_toe", "l_small_toe", "l_heel", "r_big_toe", "r_small_toe", "r_heel"]
POINTS = ["l_ankle", "r_ankle"] + FEET
FOOT_SIGMAS = np.array([.068, .066, .066, .092, .094, .094])
ANKLES = (15, 16)  # COCO body indices

SCHEMA = {
    "type": "object",
    "properties": {j: {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                       "required": ["x", "y"], "additionalProperties": False} for j in POINTS},
    "required": POINTS,
    "additionalProperties": False,
}


def prompt(w: int, h: int) -> str:
    names = ", ".join(f"{j} ({j.replace('l_', 'left ').replace('r_', 'right ').replace('_', ' ')})" for j in POINTS)
    return (
        f"The attached image is {w} x {h} pixels and is centred on one person. Return pixel "
        "coordinates for that person's ankles and feet: x from the left edge, y from the top edge, "
        "origin at the top-left corner. Left and right mean the person's own left and right, not "
        "the viewer's. Ankle: centre of the ankle joint. Big toe: tip of the big toe. Small toe: tip "
        "of the little toe. Heel: back of the heel. Points are on the shoe when shoes are worn. "
        "If a point is hidden, give your best estimate. "
        f"Keys: {names}. Look at the image directly; do not run any commands."
    )


def sample(n: int, seed: int = 0) -> list[dict]:
    d = json.load(open(ROOT / "coco_wholebody_val_v1.0.json"))
    files = {im["id"]: im["file_name"] for im in d["images"]}
    ok = [a for a in d["annotations"]
          if a.get("foot_valid") and not a["iscrowd"] and a["bbox"][3] >= 150
          and (np.array(a["foot_kpts"]).reshape(6, 3)[:, 2] > 0).all()]
    rng = np.random.default_rng(seed)
    picked = [ok[i] for i in sorted(rng.choice(len(ok), min(n, len(ok)), replace=False))]
    for a in picked:
        a["file_name"] = files[a["image_id"]]
    return picked


def image_path(file_name: str) -> Path:
    path = ROOT / "val2017" / file_name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"http://images.cocodataset.org/val2017/{file_name}", path)
    return path


def score(name: str, feet: np.ndarray, gt: np.ndarray, box: np.ndarray, ann_id: int) -> dict:
    bw, bh = box[2] - box[0], box[3] - box[1]
    d2 = ((feet - gt) ** 2).sum(1)
    err = np.sqrt(d2) / bh
    r = {"id": ann_id, "backend": name,
         "oks": float(np.exp(-d2 / (2 * FOOT_SIGMAS ** 2 * bw * bh)).mean()),
         "pck05": float((err < 0.05).mean()), "pck10": float((err < 0.10).mean())}
    for k, j in enumerate(FEET):
        r[f"{j}_err"] = err[k]
    return r


def label_one(a: dict, crops_dir: Path, cache_dir: Path) -> list[dict]:
    x, y, w, h = a["bbox"]
    box = np.array([x, y, x + w, y + h], float)
    gt = np.array(a["foot_kpts"], float).reshape(6, 3)[:, :2]
    body = np.array(a["keypoints"], float).reshape(17, 3)
    crop_path = crops_dir / f"{a['id']}.png"
    scale, origin, (cw, ch) = upscaled_crop(cv2.imread(str(image_path(a["file_name"]))), box, crop_path)
    answer, sec = ask(prompt(cw, ch), [crop_path], SCHEMA, cache_dir)
    xy = np.array([[answer[j]["x"], answer[j]["y"]] for j in POINTS], float) / scale + origin
    astra = score("astra", xy[2:], gt, box, a["id"])
    astra["sec"] = sec
    for k, j in enumerate(POINTS):
        astra[f"{j}_x"], astra[f"{j}_y"] = xy[k]
    rows = [astra]
    if (body[list(ANKLES), 2] > 0).all():
        l, r = body[ANKLES[0], :2], body[ANKLES[1], :2]
        rows.append(score("ankle_copy", np.array([l, l, l, r, r, r]), gt, box, a["id"]))
    return rows


def worst_sheet(astra: pd.DataFrame, anns: dict, out: Path, n: int = 12) -> None:
    """The n lowest-OKS people, cropped: labelled feet green, Astra red, one line toe-heel-toe per foot."""
    tiles = []
    for _, r in astra.nsmallest(n, "oks").iterrows():
        a = anns[r.id]
        x, y, w, h = a["bbox"]
        bgr = cv2.imread(str(image_path(a["file_name"])))
        gt = np.array(a["foot_kpts"], float).reshape(6, 3)[:, :2]
        pred = np.array([[r[f"{j}_x"], r[f"{j}_y"]] for j in FEET])
        for pts, color in ((gt, (0, 200, 0)), (pred, (0, 0, 255))):
            for foot in (pts[:3], pts[3:]):
                cv2.polylines(bgr, [foot[[0, 2, 1]].astype(np.int32)], False, color, 2)
                for p in foot:
                    cv2.circle(bgr, tuple(p.astype(int)), 3, color, -1)
        x1, y1, x2, y2 = crop_box(bgr.shape[:2], np.array([x, y + 0.6 * h, x + w, y + h]), 0.3)
        tile = cv2.resize(bgr[y1:y2, x1:x2], (320, 200))
        cv2.putText(tile, f"{r.oks:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="output/astra_eval/feet")
    a = p.parse_args()

    out = Path(a.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    anns = sample(a.n)
    print(f"{len(anns)} people", flush=True)
    rows = []
    with ThreadPoolExecutor(a.workers) as pool:
        for i, r in enumerate(pool.map(lambda x: label_one(x, out / "crops", out / "cache"), anns), 1):
            rows += r
            if i % 10 == 0:
                print(f"{i}/{len(anns)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "per_person.parquet", index=False)
    astra = df[df.backend == "astra"]
    worst_sheet(astra, {x["id"]: x for x in anns}, out / "worst.jpg")

    summary = {}
    for b, g in df.groupby("backend"):
        summary[b] = {"n": len(g), "oks": float(g.oks.mean()), "pck05": float(g.pck05.mean()),
                      "pck10": float(g.pck10.mean()),
                      "point_err_pct": {j: float(100 * g[f"{j}_err"].median()) for j in FEET}}
    new = astra.sec[astra.sec > 0]
    summary["astra"]["sec_per_call"] = float(new.mean()) if len(new) else None
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'backend':12s} {'n':>4s} {'OKS':>7s} {'PCK@.05':>8s} {'PCK@.10':>8s}")
    for b, s in summary.items():
        print(f"{b:12s} {s['n']:4d} {s['oks']:7.3f} {s['pck05']:8.3f} {s['pck10']:8.3f}")
    print("\nmedian error, % of box height:")
    print(f"{'point':12s}" + "".join(f"{b:>12s}" for b in summary))
    for j in FEET:
        print(f"{j:12s}" + "".join(f"{summary[b]['point_err_pct'][j]:12.1f}" for b in summary))


if __name__ == "__main__":
    main()
