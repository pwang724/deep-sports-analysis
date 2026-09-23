"""Astra as a coarse stroke classifier, scored on Tennis Player Actions.

Each image in the dataset is filed under one of four actions: forehand,
backhand, serve, ready_position. Astra sees the whole frame (one player) and
picks one. Same 100-image sample as dsa.astra.joints. Single frames, not
clips: fine-grained stroke type from video is scored later on F3Set.

    python -m dsa.astra.stroke --per-action 25
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
from dsa.astra.joints import ROOT, sample

ACTIONS = ["forehand", "backhand", "serve", "ready_position"]
SCHEMA = {
    "type": "object",
    "properties": {"action": {"type": "string", "enum": ACTIONS}},
    "required": ["action"],
    "additionalProperties": False,
}
PROMPT = (
    "The attached frame shows one tennis player. Which is the player doing? "
    "forehand: a groundstroke on the racket-hand side, any phase from backswing to follow-through. "
    "backhand: a groundstroke across the body on the non-racket side, one- or two-handed, any phase. "
    "serve: any phase of a serve, from toss to follow-through. "
    "ready_position: waiting for the ball, not swinging. "
    "Look at the image directly; do not run any commands."
)


def label_one(item: tuple, cache_dir: Path) -> dict:
    action, file_name, path, *_ = item
    answer, sec = ask(PROMPT, [path], SCHEMA, cache_dir)
    return {"action": action, "image": file_name, "pred": answer["action"], "sec": sec}


def mistakes_sheet(df: pd.DataFrame, paths: dict, out: Path, n: int = 12) -> None:
    tiles = []
    for _, r in df[df.pred != df.action].head(n).iterrows():
        tile = cv2.resize(cv2.imread(str(paths[(r.action, r.image)])), (320, 180))
        cv2.putText(tile, f"{r.action} -> {r.pred}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        tiles.append(tile)
    if not tiles:
        return
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=ROOT)
    p.add_argument("--per-action", type=int, default=25)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="output/astra_eval/stroke")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = sample(Path(a.root), a.per_action)
    print(f"{len(items)} images", flush=True)
    with ThreadPoolExecutor(a.workers) as pool:
        df = pd.DataFrame(pool.map(lambda it: label_one(it, out / "cache"), items))
    df.to_parquet(out / "per_image.parquet", index=False)
    mistakes_sheet(df, {(it[0], it[1]): it[2] for it in items}, out / "mistakes.jpg")

    confusion = pd.crosstab(df.action, df.pred).reindex(index=ACTIONS, columns=ACTIONS, fill_value=0)
    new = df.sec[df.sec > 0]
    summary = {"n": len(df), "accuracy": float((df.pred == df.action).mean()),
               "per_action": {k: float((g.pred == k).mean()) for k, g in df.groupby("action")},
               "sec_per_call": float(new.mean()) if len(new) else None,
               "confusion": confusion.to_dict()}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\naccuracy {summary['accuracy']:.1%} on {len(df)} images\n")
    print("rows: label, columns: Astra")
    print(confusion.to_string())


if __name__ == "__main__":
    main()
