"""Astra as a joint labeler, scored on Tennis Player Actions.

A fixed sample per action is cropped to its labelled box (padded 20%) and
upscaled to 768 px on the long side: the view ViTPose gets. Astra returns
pixel coordinates of the 17 COCO joints in the crop; they are mapped back to
the image and scored with the OKS and PCK of dsa.pose.eval_pose, next to
ViTPose's stored result on the same images (output/eval_gt, run once with
scripts/eval_gt.py or src/dsa/cloud/eval_pose.py).

    python -m dsa.astra.joints --per-action 25
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dsa.astra.codex import ask
from dsa.pose.eval_pose import _score, crop_box, iter_labels, print_summary, summarize
from dsa.pose.skeleton import COCO17, SKELETON

ROOT = "data/tennis_player_actions/Tennis Player Actions Dataset for Human Pose Estimation"
LONG_SIDE = 768

SCHEMA = {
    "type": "object",
    "properties": {j: {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                       "required": ["x", "y"], "additionalProperties": False} for j in COCO17},
    "required": COCO17,
    "additionalProperties": False,
}


def prompt(w: int, h: int) -> str:
    names = ", ".join(f"{j} ({j.replace('l_', 'left ').replace('r_', 'right ')})" for j in COCO17)
    return (
        f"The attached image is {w} x {h} pixels and shows one tennis player. Return the pixel "
        "coordinates of the player's 17 body joints: x from the left edge, y from the top edge, "
        "origin at the top-left corner. Left and right mean the player's own left and right, not "
        "the viewer's. Place each joint at the centre of the joint (for eyes and nose, the centre "
        "of that feature). If a joint is hidden, give your best estimate of where it is. "
        f"Keys: {names}. Look at the image directly; do not run any commands."
    )


def sample(root: Path, per_action: int, seed: int = 0) -> list[tuple]:
    by_action: dict[str, list] = {}
    for item in iter_labels(root):
        by_action.setdefault(item[0], []).append(item)
    rng = np.random.default_rng(seed)
    return [items[i] for a, items in sorted(by_action.items())
            for i in sorted(rng.choice(len(items), min(per_action, len(items)), replace=False))]


def upscaled_crop(bgr: np.ndarray, box: np.ndarray, path: Path) -> tuple[float, np.ndarray, tuple[int, int]]:
    """Write the box padded 20% and upscaled to LONG_SIDE; return (scale, crop origin, crop w/h).

    Map crop pixels back to the image with xy / scale + origin.
    """
    x1, y1, x2, y2 = crop_box(bgr.shape[:2], box, 0.2)
    scale = LONG_SIDE / max(y2 - y1, x2 - x1)
    crop = cv2.resize(bgr[y1:y2, x1:x2], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(path), crop)
    return scale, np.array([x1, y1], float), (crop.shape[1], crop.shape[0])


def label_one(item: tuple, crops_dir: Path, cache_dir: Path) -> dict:
    action, file_name, path, gt, vis, box = item
    crop_path = crops_dir / f"{action}_{Path(file_name).stem}.png"
    scale, origin, (w, h) = upscaled_crop(cv2.imread(str(path)), box, crop_path)
    answer, sec = ask(prompt(w, h), [crop_path], SCHEMA, cache_dir)
    xy = np.array([[answer[j]["x"], answer[j]["y"]] for j in COCO17], float) / scale + origin
    row = _score("astra", xy, gt, vis, box, action, file_name)
    row["sec"] = sec
    for k, j in enumerate(COCO17):
        row[f"{j}_x"], row[f"{j}_y"] = xy[k]
    return row


def draw(bgr: np.ndarray, xy: np.ndarray, vis: np.ndarray, color) -> None:
    for a, b in SKELETON:
        if vis[a] > 0 and vis[b] > 0:
            cv2.line(bgr, tuple(xy[a].astype(int)), tuple(xy[b].astype(int)), color, 2)
    for k in range(17):
        if vis[k] > 0:
            cv2.circle(bgr, tuple(xy[k].astype(int)), 3, color, -1)


def worst_sheet(rows: pd.DataFrame, items: dict, out: Path, n: int = 12) -> None:
    """Grid of the n lowest-OKS crops: ground truth green, Astra red."""
    tiles = []
    for _, r in rows.nsmallest(n, "oks").iterrows():
        action, file_name, path, gt, vis, box = items[(r.action, r.image)]
        bgr = cv2.imread(str(path))
        xy = np.array([[r[f"{j}_x"], r[f"{j}_y"]] for j in COCO17])
        draw(bgr, gt, vis, (0, 200, 0))
        draw(bgr, xy, vis, (0, 0, 255))
        x1, y1, x2, y2 = crop_box(bgr.shape[:2], box, 0.2)
        tile = cv2.resize(bgr[y1:y2, x1:x2], (240, 320))
        cv2.putText(tile, f"{action} {r.oks:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        tiles.append(tile)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--per-action", type=int, default=25)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--vitpose", default="output/eval_gt/per_image.parquet")
    p.add_argument("--out", default="output/astra_eval/joints")
    a = p.parse_args()

    out = Path(a.out)
    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    items = sample(Path(a.root), a.per_action)
    print(f"{len(items)} images", flush=True)
    rows = []
    with ThreadPoolExecutor(a.workers) as pool:
        for i, row in enumerate(pool.map(lambda it: label_one(it, crops_dir, out / "cache"), items), 1):
            rows.append(row)
            if i % 10 == 0:
                print(f"{i}/{len(items)}", flush=True)
    astra = pd.DataFrame(rows)

    vp = pd.read_parquet(a.vitpose)
    keys = set(zip(astra.action, astra.image))
    vp = vp[(vp.backend == "vitpose") & np.array([k in keys for k in zip(vp.action, vp.image)], dtype=bool)]
    df = pd.concat([astra, vp], ignore_index=True)
    df.to_parquet(out / "per_image.parquet", index=False)
    new = astra.sec[astra.sec > 0]
    summary = summarize(df, {"astra": float(new.mean()) if len(new) else None})
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    worst_sheet(astra, {(it[0], it[1]): it for it in items}, out / "worst.jpg")
    print(f"\n{len(astra)} images, {len(vp)} with ViTPose results\n")
    print_summary(summary)


if __name__ == "__main__":
    main()
