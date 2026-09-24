"""RacketVision's own racket model (RTMDet-M + RTMPose-M) on the rackets Astra was scored on.

The released checkpoints (linfeng302/RacketVision-Models) run on one GPU in
an OpenMMLab image of their own; the repo's pinned dependencies do not mix
with mmcv. RTMPose is given racket boxes from:

  gt_box            the labelled racket box: the keypoint model alone.
  detector          RTMDet on the full frame, tennis rackets at score >= 0.3
                    (their defaults).
  person            RF-DETR person boxes (conf >= 0.4), as the pipeline makes them.
  wrist, forearm    squares around each ViTPose wrist (see hand_boxes); per
                    person, the wrist whose racket RTMPose is surer of.
  detector+forearm  per person, a detection whose handle is within 0.25 x
                    height of a wrist, else the forearm box.

For each labelled racket the nearest predicted racket is scored exactly as
Astra is (dsa.astra.racket.score); none within 0.5 racket lengths is a miss.
RF-DETR and ViTPose run locally (cached in people.pkl), RTMPose on Modal. The clips are in
RacketVision's test split, so the model never trained on them.

    modal run src/dsa/cloud/racket_pose.py
"""
import json
from pathlib import Path

import modal

RP = Path(__file__).resolve().parents[3] / "data/code/racketvision/source/RacketPose" if modal.is_local() else Path("/rp")

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
def predict(frames: list[bytes], boxes: list[list[list[float]]]) -> list[dict]:
    """Per frame: RTMPose on each given xyxy box, and RTMDet + RTMPose on the full frame."""
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

    def keypoints(img, xyxy):
        if not len(xyxy):
            return []
        return [{"box": b.tolist(), "kp": p.pred_instances.keypoints[0].tolist(),
                 "kp_score": p.pred_instances.keypoint_scores[0].tolist()}
                for p, b in zip(inference_topdown(pose, img, np.asarray(xyxy, float)), np.asarray(xyxy, float))]

    out = []
    for raw, given in zip(frames, boxes):
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        with DefaultScope.overwrite_default_scope("mmdet"):
            d = inference_detector(det, img).pred_instances
        keep = (d.labels.cpu().numpy() == 2) & (d.scores.cpu().numpy() >= 0.3)
        out.append({"given": keypoints(img, given), "detections": keypoints(img, d.bboxes.cpu().numpy()[keep])})
    return out


def people(items: list[dict], frames: list, cache: Path) -> list[list[dict]]:
    """RF-DETR person boxes (conf >= 0.4) and ViTPose joints per frame, as the pipeline makes them."""
    import pickle

    if cache.exists():
        return pickle.loads(cache.read_bytes())
    import cv2
    import torch

    from dsa.pose.backends import load_models, run_vitpose

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    m = load_models("rfdetr", 1152, Path("models/vitpose-plus-huge"), device)
    out = []
    for bgr in frames:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        xyxy, conf = m.detector.detect(rgb)
        xyxy = xyxy[conf >= 0.4]
        xy, sc = run_vitpose(*m.vitpose, device, rgb, xyxy)
        out.append([{"box": b, "kp": k, "kp_score": s} for b, k, s in zip(xyxy, xy, sc)])
    cache.write_bytes(pickle.dumps(out))
    return out


def hand_boxes(person: dict) -> dict[str, list]:
    """Candidate racket boxes from one person: the person box, and per wrist a square
    on the wrist (side 0.8 x height, room for the racket in any direction) and a
    tighter one pushed along the forearm (side 0.5 x height, centre 0.2 x height out)."""
    import numpy as np

    b, kp = person["box"], person["kp"]
    h = b[3] - b[1]
    sq = lambda c, side: [c[0] - side / 2, c[1] - side / 2, c[0] + side / 2, c[1] + side / 2]
    boxes = {"person": [list(b)]}
    boxes["wrist"] = [sq(kp[w], 0.8 * h) for w in (9, 10)]
    fore = []
    for e, w in ((7, 9), (8, 10)):
        v = kp[w] - kp[e]
        fore.append(sq(kp[w] + 0.2 * h * v / max(np.linalg.norm(v), 1e-6), 0.5 * h))
    boxes["forearm"] = fore
    return boxes


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
        frames.append(cap.read()[1])
        cap.release()
    ppl = people(items, frames, Path(out) / "people.pkl")

    # Every candidate box goes to RTMPose in one list per frame; remember what each one is.
    given, index = [], []
    for it, ps in zip(items, ppl):
        x, y, w, h = it["box"]
        boxes, idx = [[x, y, x + w, y + h]], [("gt_box", -1, 0)]
        for k, p in enumerate(ps):
            for method, bs in hand_boxes(p).items():
                for side, b in enumerate(bs):
                    boxes.append(b)
                    idx.append((method, k, side))
        given.append(boxes)
        index.append(idx)
    preds = predict.remote([cv2.imencode(".png", f)[1].tobytes() for f in frames], given)

    rows = []
    for it, ps, idx, p in zip(items, ppl, index, preds):
        by = {}
        for (method, k, side), r in zip(idx, p["given"]):
            by.setdefault(method, {}).setdefault(k, []).append(r)
        # One racket per person per method: the wrist whose racket RTMPose is surer of.
        cands = {m: [max(rs, key=lambda r: np.mean(r["kp_score"])) for rs in per.values()] for m, per in by.items()}
        cands["detector"] = p["detections"]
        # Detector, with each person's racket taken from a detection whose handle is
        # near one of their wrists, else from the forearm box.
        both = []
        for k, person in enumerate(ps):
            h = person["box"][3] - person["box"][1]
            near = [d for d in p["detections"]
                    if min(np.linalg.norm(np.array(d["kp"][2]) - person["kp"][w]) for w in (9, 10)) < 0.25 * h]
            both.append(max(near, key=lambda d: np.mean(d["kp_score"])) if near else cands["forearm"][k])
        cands["detector+forearm"] = both
        for method, rs in cands.items():
            rid = f"{Path(it['video']).stem}_{it['frame']}_{it['k']}"
            if not rs:
                rows.append({"id": rid, "backend": method, "found": False, "mean_err": np.nan,
                             "pck10": 0.0, "pck20": 0.0, "pck10_any_side": 0.0})
                continue
            # The predicted racket closest to the labelled one is the match.
            best = min((score(method, np.array(r["kp"]), it) for r in rs), key=lambda r: r["mean_err"])
            found = best["mean_err"] < 0.5
            rows.append({**best, "found": found} if found else
                        {"id": rid, "backend": method, "found": False, "mean_err": np.nan,
                         "pck10": 0.0, "pck20": 0.0, "pck10_any_side": 0.0})
    df = pd.DataFrame(rows)
    df.to_parquet(Path(out) / "per_racket_boxes.parquet", index=False)

    summary = {}
    for b, g in df.groupby("backend"):
        summary[b] = {"n": len(g), "found": float(g.found.mean()), "mean_err_found": float(g.mean_err.mean()),
                      "pck10": float(g.pck10.mean()), "pck20": float(g.pck20.mean()),
                      "pck10_any_side": float(g.pck10_any_side.mean())}
    (Path(out) / "summary_boxes.json").write_text(json.dumps(summary, indent=2))
    print(pd.DataFrame(summary).T.round(3).to_string())
