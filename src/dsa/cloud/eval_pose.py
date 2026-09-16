"""Score pose backends against labelled joints on one GPU container.

    modal run src/dsa/cloud/eval_pose.py            # all ~2,000 images
    modal run src/dsa/cloud/eval_pose.py --limit 50 # 50 per action, smoke test
"""
import json
from pathlib import Path

import modal

from dsa.cloud.modal_app import GPU, VITPOSE_DIR, VOL, image, volume

DATASET_DIR = VOL / "datasets" / "tennis_player_actions"

app = modal.App("deep-sports-eval-pose", image=image)


@app.function(gpu=GPU, volumes={str(VOL): volume}, cpu=4, memory=16_384, timeout=2 * 60 * 60)
def evaluate_on_gpu(limit: int | None) -> tuple[bytes, dict]:
    import io

    import torch

    from dsa.pose.eval_pose import EvalModels, evaluate

    models = EvalModels(VITPOSE_DIR, device="cuda", dtype=torch.float16)
    df, timing = evaluate(DATASET_DIR, models, limit)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue(), timing


@app.local_entrypoint()
def main(limit: int | None = None, out: str = "output/eval_gt"):
    import io

    import pandas as pd

    from dsa.pose.eval_pose import print_summary, summarize

    raw, timing = evaluate_on_gpu.remote(limit)
    df = pd.read_parquet(io.BytesIO(raw))
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "per_image.parquet", index=False)
    summary = summarize(df, timing)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{df.image.nunique()} images, {GPU}\n")
    print_summary(summary)
