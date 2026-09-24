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
import os
import time
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


@app.cls(gpu="L40S", cpu=2.0, memory=8192, timeout=30 * 60, max_containers=20, scaledown_window=60)
class Rackets:
    """RTMDet + RTMPose, loaded once per container.

    L40S is cheapest per frame (`bench`, 300 frames warm, incl. CPU and memory): L4 21 frames/s
    $0.013 / 1k frames, A10 35 $0.010, L40S 49 $0.005 (2026-09-24).
    """

    @modal.enter()
    def load(self):
        import functools

        import torch

        torch.load = functools.partial(torch.load, weights_only=False)
        from huggingface_hub import hf_hub_download
        from mmdet.apis import init_detector
        from mmpose.apis import init_model as init_pose_model

        ckpt = lambda f: hf_hub_download("linfeng302/RacketVision-Models", f"checkpoints/{f}")
        self.det = init_detector("/rp/configs/detection/rtmdet_m_racket_infer.py", ckpt("epoch_300.pth"), device="cuda")
        self.pose = init_pose_model("/rp/configs/pose/rtmpose_m_racket_infer.py", ckpt("best_PCK_epoch_90.pth"),
                                    device="cuda")
        self.gpu = torch.cuda.get_device_name(0)

    def _run(self, frames: list, boxes: list[list[list[float]]]) -> list[dict]:
        """frames: encoded images (bytes) or BGR arrays."""
        import cv2
        import numpy as np
        from mmdet.apis import inference_detector
        from mmengine.registry import DefaultScope
        from mmpose.apis import inference_topdown

        def keypoints(img, xyxy):
            if not len(xyxy):
                return []
            return [{"box": b.tolist(), "kp": p.pred_instances.keypoints[0].tolist(),
                     "kp_score": p.pred_instances.keypoint_scores[0].tolist()}
                    for p, b in zip(inference_topdown(self.pose, img, np.asarray(xyxy, float)), np.asarray(xyxy, float))]

        out = []
        for raw, given in zip(frames, boxes):
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if isinstance(raw, bytes) else raw
            with DefaultScope.overwrite_default_scope("mmdet"):
                d = inference_detector(self.det, img).pred_instances
            keep = (d.labels.cpu().numpy() == 2) & (d.scores.cpu().numpy() >= 0.3)
            out.append({"given": keypoints(img, given), "detections": keypoints(img, d.bboxes.cpu().numpy()[keep])})
        return out

    @modal.method()
    def predict(self, frames: list[bytes], boxes: list[list[list[float]]]) -> list[dict]:
        """Per frame: RTMPose on each given xyxy box, and RTMDet + RTMPose on the full frame."""
        return self._run(frames, boxes)

    @modal.method()
    def clip(self, video: bytes, offsets: list[int]) -> dict:
        """One call per clip: the clip as H.264 (dsa.label.clips.clip_video), the frames at `offsets`."""
        import tempfile

        import cv2

        t0, want, out, busy = time.time(), set(offsets), {}, 0.0
        with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
            f.write(video)
            f.flush()
            cap = cv2.VideoCapture(f.name)
            for i in range(max(offsets) + 1):
                ok, bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"clip has {i} frames, wanted offsets up to {max(offsets)}")
                if i in want:
                    t1 = time.time()
                    out[i] = self._run([bgr], [[]])[0]
                    busy += time.time() - t1
        return {"preds": [out[o] for o in offsets], "seconds": time.time() - t0, "decode": time.time() - t0 - busy,
                "gpu": self.gpu, "task": os.environ.get("MODAL_TASK_ID")}

    @modal.method()
    def timed(self, frames: list[bytes], boxes: list[list[list[float]]]) -> dict:
        """predict, plus the seconds spent and the GPU, for cost accounting."""
        t0 = time.time()
        out = self._run(frames, boxes)
        return {"preds": out, "seconds": time.time() - t0, "gpu": self.gpu, "task": os.environ.get("MODAL_TASK_ID")}


# The old function's name, so `predict.remote(frames, boxes)` keeps working.
predict = Rackets().predict


@app.local_entrypoint()
def bench(gpus: str = "L4,A10,L40S", n: int = 300):
    """Warm frames/s and $ per 1k frames of one clip on each GPU:  modal run src/dsa/cloud/racket_pose.py::bench"""
    from dsa.cloud.pose import CPU, MEMORY, PRICE
    from dsa.data.paths import SCRATCH
    from dsa.label.clips import clip_video

    c = json.loads((SCRATCH / "clips/clips_v1/clip_list.json").read_text())[2]
    video = clip_video(c, SCRATCH / "clips" / "bench")
    offsets = list(range(min(n, c["end"] - c["start"])))
    for g in gpus.split(","):
        R = Rackets.with_options(gpu=g)
        for k in ("cold", "warm"):
            t = time.time()
            r = R().clip.remote(video, offsets)
            rate = next(v for key, v in PRICE.items() if key in r["gpu"])
            usd = r["seconds"] / 3600 * (rate + CPU * PRICE["cpu"] + MEMORY / 1024 * PRICE["gib"])
            print(f"{g:5s} {k}: wall {time.time() - t:6.1f} s, busy {r['seconds']:6.1f} s (decode {r['decode']:.1f} s), "
                  f"{len(offsets) / r['seconds']:5.2f} frames/s, ${usd / len(offsets) * 1000:.3f} per 1k frames ({r['gpu']})",
                  flush=True)


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
