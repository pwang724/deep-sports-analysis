"""RF-DETR Medium @ 1152 + ViTPose-Plus-Huge on Modal GPUs, for dsa.label.clips.

The same models and code as dsa.label.clips.Local (fp32, conf >= 0.4), loaded
once per container from the deep-sports volume (dsa.cloud.modal_app setup).
Per frame the result is what Local.people returns: [{"box", "xy", "score"}] as
numpy arrays.

  people(jpegs)          a few frames as JPEG (keyframes)
  clip(video, offsets)   one call per clip: the clip as a near-lossless H.264
                         file (dsa.label.clips.clip_video, ~30x smaller than
                         JPEG frames), decoded here as it goes; the frames at `offsets`

Both return the GPU seconds and the container, for cost accounting.

    modal run src/dsa/cloud/pose.py --gpus L4,A10,L40S     # frames/s and $ per 1k frames per GPU
"""
import os
import time

import modal

from dsa.cloud.modal_app import VITPOSE_DIR, VOL, image, volume

# fp32 (to match the local labels): warm, 300 frames of ~9.5 people, incl. CPU and memory,
#   L4 2.96 frames/s $0.090 / 1k frames, A10 3.76 $0.093, L40S 7.03 $0.038 (2026-09-24).
GPU = "L40S"            # cheapest per frame
MAX_CONTAINERS = 20     # speed from more containers, not bigger GPUs
CPU, MEMORY = 2.0, 8192
# Modal list prices, USD per hour: GPU, and per core / per GiB of what a container reserves.
PRICE = {"L4": 0.80, "A10": 1.10, "L40S": 1.95, "A100": 2.10, "cpu": 0.0473, "gib": 0.0080}

app = modal.App("deep-sports-label-pose", image=image)


def frames_at(video: bytes, offsets: list[int]):
    """Yield (offset, BGR frame) for `offsets` (0 = first frame, ascending) of an mp4 given as bytes."""
    import tempfile

    import cv2

    want = set(offsets)
    with tempfile.NamedTemporaryFile(suffix=".mp4") as f:
        f.write(video)
        f.flush()
        cap = cv2.VideoCapture(f.name)
        for i in range(max(offsets) + 1):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"clip has {i} frames, wanted offsets up to {max(offsets)}")
            if i in want:
                yield i, bgr
        cap.release()


@app.cls(gpu=GPU, volumes={str(VOL): volume}, cpu=CPU, memory=MEMORY, timeout=30 * 60, max_containers=MAX_CONTAINERS,
         scaledown_window=60)
class Pose:
    @modal.enter()
    def load(self):
        import torch

        from dsa.pose.backends import load_models

        self.models = load_models("rfdetr", 1152, VITPOSE_DIR, "cuda", torch.float32)
        self.gpu = torch.cuda.get_device_name(0)

    def _people(self, bgr):
        import cv2

        from dsa.pose.backends import run_vitpose

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        xyxy, conf = self.models.detector.detect(rgb)
        xyxy = xyxy[conf >= 0.4]
        xy, sc = run_vitpose(*self.models.vitpose, "cuda", rgb, xyxy)
        return [{"box": b, "xy": k, "score": s} for b, k, s in zip(xyxy, xy, sc)]

    def _result(self, people, t0, decode_s=0.0):
        return {"people": people, "seconds": time.time() - t0, "decode": decode_s, "gpu": self.gpu,
                "task": os.environ.get("MODAL_TASK_ID")}

    @modal.method()
    def people(self, frames: list[bytes]) -> dict:
        import cv2
        import numpy as np

        t0 = time.time()
        return self._result([self._people(cv2.imdecode(np.frombuffer(r, np.uint8), cv2.IMREAD_COLOR)) for r in frames], t0)

    @modal.method()
    def clip(self, video: bytes, offsets: list[int]) -> dict:
        t0, out, busy = time.time(), {}, 0.0
        for o, bgr in frames_at(video, sorted(offsets)):
            t1 = time.time()
            out[o] = self._people(bgr)
            busy += time.time() - t1
        return self._result([out[o] for o in offsets], t0, time.time() - t0 - busy)


def usd(gpu_name: str, seconds: float) -> float:
    """Cost of `seconds` of one container: GPU plus the CPU and memory it reserves."""
    rate = next((v for k, v in PRICE.items() if k in gpu_name), PRICE["A100"])
    return seconds / 3600 * (rate + CPU * PRICE["cpu"] + MEMORY / 1024 * PRICE["gib"])


@app.local_entrypoint()
def main(gpus: str = "L4,A10,L40S", clip: str = "", n: int = 300):
    """Warm frames/s and $ per 1k frames of one clip on each GPU."""
    import json
    from pathlib import Path

    from dsa.data.paths import SCRATCH
    from dsa.label.clips import clip_video

    clips = json.loads((SCRATCH / "clips/clips_v1/clip_list.json").read_text())
    c = next(x for x in clips if x["clip"] == clip) if clip else clips[2]
    video = clip_video(c, SCRATCH / "clips" / "bench")
    offsets = list(range(min(n, c["end"] - c["start"])))
    print(f"{c['clip']}: {len(offsets)} frames, {len(video) / 1e6:.1f} MB")
    for g in gpus.split(","):
        P = Pose.with_options(gpu=g)
        for k in ("cold", "warm"):
            t = time.time()
            r = P().clip.remote(video, offsets)
            n_people = sum(map(len, r["people"]))
            fps = len(offsets) / r["seconds"]
            print(f"{g:5s} {k}: wall {time.time() - t:6.1f} s, busy {r['seconds']:6.1f} s (decode {r['decode']:.1f} s), "
                  f"{fps:5.2f} frames/s, {n_people / len(offsets):.1f} people/frame, "
                  f"${usd(r['gpu'], r['seconds']) / len(offsets) * 1000:.3f} per 1k frames ({r['gpu']})", flush=True)
