"""RacketVision's own racket model (RTMDet-M + RTMPose-M) on the rackets Astra was scored on.

The released checkpoints (linfeng302/RacketVision-Models) run on one GPU in
an OpenMMLab image of their own; the repo's pinned dependencies do not mix
with mmcv. Two settings:

  gt_box    RTMPose on the labelled racket box: the keypoint model alone.
  detector  RTMDet on the full frame, tennis rackets at score >= 0.3 (their
            defaults), RTMPose on each; the box with the highest IoU with the
            label is scored, and a racket with no box at IoU >= 0.1 is missed.

Both are scored exactly as Astra is (dsa.astra.racket.score). The clips are in
RacketVision's test split, so the model never trained on them.

    modal run src/dsa/cloud/racket_pose.py
"""
import json
from pathlib import Path

import modal

RP = Path(__file__).resolve().parents[3] / "data/racketvision_repo/source/RacketPose" if modal.is_local() else Path("/rp")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.1.2", "torchvision==0.16.2", index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("mmcv==2.1.0", find_links="https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html")
    .pip_install("numpy<2", "mmengine==0.10.7", "mmdet==3.3.0", "opencv-python-headless<4.11", "huggingface_hub",
                 "json_tricks", "munkres", "scipy", "matplotlib", "xtcocotools", "pillow")
    .pip_install("mmpose==1.3.2", extra_options="--no-deps")
    .add_local_dir(RP / "configs", "/rp/configs")
)
app = modal.App("deep-sports-racket-pose", image=image)


@app.function(gpu=["L4", "A10"], timeout=30 * 60)
def predict(frames: list[bytes], boxes: list[list[float]]) -> list[dict]:
    import functools

    import cv2
    import numpy as np
    import torch

    torch.load = functools.partial(torch.load, weights_only=False)
    from huggingface_hub import hf_hub_download
    from mmdet.apis import inference_detector, init_detector
    from mmengine.registry import DefaultScope
    from mmpose.apis import inference_topdown
    from mmpose.apis import init_model as init_pose_model

    ckpt = lambda f: hf_hub_download("linfeng302/RacketVision-Models", f"checkpoints/{f}")
    det = init_detector("/rp/configs/detection/rtmdet_m_racket_infer.py", ckpt("epoch_300.pth"), device="cuda")
    pose = init_pose_model("/rp/configs/pose/rtmpose_m_racket_infer.py", ckpt("best_PCK_epoch_90.pth"), device="cuda")

    out = []
    for raw, (x, y, w, h) in zip(frames, boxes):
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        gt = inference_topdown(pose, img, np.array([[x, y, x + w, y + h]]))[0].pred_instances
        with DefaultScope.overwrite_default_scope("mmdet"):
            d = inference_detector(det, img).pred_instances
        keep = (d.labels.cpu().numpy() == 2) & (d.scores.cpu().numpy() >= 0.3)
        dboxes = d.bboxes.cpu().numpy()[keep]
        dets = []
        if len(dboxes):
            for p, b, s in zip(inference_topdown(pose, img, dboxes), dboxes, d.scores.cpu().numpy()[keep]):
                dets.append({"box": b.tolist(), "score": float(s), "kp": p.pred_instances.keypoints[0].tolist()})
        out.append({"gt_box": gt.keypoints[0].tolist(), "detections": dets})
    return out


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


@app.local_entrypoint()
def main(n: int = 100, out: str = "output/astra_eval/racket"):
    import cv2
    import numpy as np
    import pandas as pd

    from dsa.astra.racket import POINTS, sample, score

    items = sample(n)
    frames = []
    for it in items:
        cap = cv2.VideoCapture(it["video"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, it["frame"])
        ok, bgr = cap.read()
        cap.release()
        frames.append(cv2.imencode(".png", bgr)[1].tobytes())
    preds = predict.remote(frames, [it["box"].tolist() for it in items])

    rows = []
    for it, p in zip(items, preds):
        rows.append(score("rtmpose_gt_box", np.array(p["gt_box"]), it))
        x, y, w, h = it["box"]
        best = max(p["detections"], key=lambda d: iou(d["box"], [x, y, x + w, y + h]), default=None)
        if best and iou(best["box"], [x, y, x + w, y + h]) >= 0.1:
            rows.append({**score("rtmdet_rtmpose", np.array(best["kp"]), it), "detected": True})
        else:
            rows.append({"id": f"{Path(it['video']).stem}_{it['frame']}_{it['k']}", "backend": "rtmdet_rtmpose",
                         "detected": False, "mean_err": np.nan, "pck10": 0.0, "pck20": 0.0, "pck10_any_side": 0.0})
    df = pd.DataFrame(rows)
    df.to_parquet(Path(out) / "per_racket_rtmpose.parquet", index=False)

    summary = {}
    for b, g in df.groupby("backend"):
        summary[b] = {"n": len(g), "mean_err": float(g.mean_err.mean()), "pck10": float(g.pck10.mean()),
                      "pck20": float(g.pck20.mean()), "pck10_any_side": float(g.pck10_any_side.mean()),
                      "point_err_median": {j: float(g[f"{j}_err"].median()) for j in POINTS}}
        if "detected" in g:
            summary[b]["detected"] = float(g.detected.fillna(True).mean())
    (Path(out) / "summary_rtmpose.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
