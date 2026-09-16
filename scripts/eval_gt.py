"""Score pose backends against hand-labelled tennis joints, locally.

    python scripts/eval_gt.py --root "data/tennis_player_actions/Tennis Player Actions Dataset for Human Pose Estimation"

See dsa.pose.eval_pose for the backends and metrics. For a GPU run use
src/dsa/cloud/eval_pose.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dsa.pose.eval_pose import EvalModels, evaluate, print_summary, summarize


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", required=True)
    p.add_argument("--limit", type=int, default=None, help="images per action, for a quick run")
    p.add_argument("--vitpose", default="models/vitpose-plus-huge")
    p.add_argument("--out", default="output/eval_gt")
    p.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = p.parse_args()

    models = EvalModels(a.vitpose, a.device)
    df, timing = evaluate(a.root, models, a.limit)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "per_image.parquet", index=False)
    summary = summarize(df, timing)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary)


if __name__ == "__main__":
    main()
