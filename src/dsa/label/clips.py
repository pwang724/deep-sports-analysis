"""Pre-fill labels over short video clips: every labeler, every frame, tracked.

Clips are 10 s windows (--seconds) drawn from the YouTube manifest (status
done), every video once before any twice, round-robin over camera x surface x
level, plus own recordings and broadcast matches. Singles from behind
the baseline only for now: manifest rows with play == doubles or camera
drone / fence / side are skipped.

Per clip:
  shots      cuts where the HSV histogram of consecutive frames jumps
             (CUT_BHATTACHARYYA); camera pan within a shot is the median
             Lucas-Kanade motion of background corners (frames.cam_dx / cam_dy,
             px from the shot's first frame).
  keyframes  every KEY_S seconds, plus the middle of any shot without one.
             Astra (dsa.label.prefill.scene_prompt) gives view, in play, the
             players (numbered boxes) and 14 court points, in the slim v1 call
             (keyframes answered before the switch keep their v0 answer).
  scope      a frame takes view / singles from the nearest keyframe of its
             shot. Out of scope (not a view, or Astra names > 2 players, i.e.
             doubles or crowd) -> scene only. Frames in a shot without a
             keyframe are not labelled at all.
  people     RF-DETR Medium @ 1152 + ViTPose-Plus-Huge on every --stride'th
             in-scope frame (keyframes always), tracked within a shot (centre
             distance in box heights + IoU, gaps up to 0.5 s). A track is a
             player if Astra named it on most keyframes it is on, or if it
             re-acquires a lost player unambiguously (link_players) (at most 2; people.named is Astra's own call
             on keyframes); head top on keyframes only.
  ball       WASB on t-1, t, t+1, every in-scope frame.
  court      per shot, the per-point median of Astra's keyframe answers,
             written on every in-scope frame of the shot (keyframe rows keep
             their own answer in court.kf_kp, for dsa.label.consistency).
  scene      every frame with a keyframe in its shot; in_play only where the
             keyframes around it agree (NaN across a change). scene.singles.
  rackets    RacketVision on Modal on the same frames as people (--no-rackets).

RF-DETR + ViTPose run on Modal L40S by default (--pose modal, dsa.cloud.pose;
fp32 as locally; one call per clip, the clip sent as near-lossless H.264, about
$0.04 per 1k frames); --pose local runs them here (MPS, ~1.5 s a frame). WASB
always runs here (0.09 s a frame). Pass-1 results are cached per clip in
data/scratch/clips/<out>/scan/, Astra answers in output/astra_cache/prefill,
so a rerun redoes only Modal. `kill -USR1 <pid>` prints every thread's stack.

frames gets extra columns clip, collection, video, camera, surface, level, play,
shot, keyframe, scope (view | nonview | doubles | none), heads (comma list of
heads labelled), cam_dx, cam_dy. The clips picked go to data/scratch/clips/<out>/clips.json
(--clip-list redoes them), timings to timings.json beside it.

    python -m dsa.label.clips --clips 60 --out clips_v1
    python -m dsa.label.clips --clips 5 --own 1 --broadcast 1 --seconds 6 --stride 2 --out clips_smoke
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import pickle
import signal
import subprocess
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from dsa.astra.court import diagram
from dsa.data import schema
from dsa.data.paths import DATA, RAW, SCRATCH, VIDEOS
from dsa.astra.codex import NotCached
from dsa.label.prefill import (ASTRA, PEOPLE, RACKETS, WASB, astra_scene, draw_boxes, head_top_rule, players,
                                  sheet_image)

KEY_S = 2.0                    # Astra keyframe spacing: in play changes on a scale of seconds, a call costs ~20 s
FLOW_W = 640                   # px width the camera shift is tracked at
CUT_BHATTACHARYYA = 0.35       # histogram distance of a hard cut; pans and players moving stay under ~0.15
MIN_SHOT_S = 0.3               # cuts closer than this are one transition (fades, flashes)
IOU_MATCH = 0.3                # a person box overlaps its previous self at least this much at >= 10 fps
MATCH_H = 0.5                  # box heights between a track's predicted centre and its next box
MATCH_H_PER_S = 1.5            # + this per second of gap
MATCH_RATIO = 0.6              # box heights of one person in neighbouring frames agree this well
TRACK_GAP_S = 0.5              # s a lost track may reappear within
REACQ_S = 1.0                  # s a lost player may be re-acquired within (dsa.label.consistency too)
REACQ_H_PER_S = 3.0            # box heights a second a player may move while lost
REACQ_MARGIN = 1.5             # the next nearest person must be this much further away
SCENE_VARIANT = "v1"           # slim scene call (dsa.astra.scene_cost): same accuracy, ~25% less of the Codex meter
MAX_PLAYERS = 2                # singles: more named players means doubles or bystanders, out of scope
EXCLUDED_CAMERAS = {"drone", "fence", "side"}
JPEG_Q = 95                    # keyframes sent to Modal; q95 moves ViTPose joints by well under a pixel
VIDEO_CRF = 4                  # clips sent to Modal as H.264: ~11 MB per 10 s of 720p60 (JPEG q95: ~180 MB)
CACHE = Path("output/astra_cache/prefill")


def log(*a) -> None:
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def media_id(media: str) -> str:
    return Path(media).with_suffix("").as_posix().replace("/", "__")


def probe(path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    out = (cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
           int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return out


def window(path: Path, seconds: float, rng: np.random.Generator, meta: dict) -> dict | None:
    fps, count, w, h = probe(path)
    n, edge = int(round(seconds * fps)), int(2 * fps)
    if count - 2 * edge - n <= 0:
        return None
    start = int(rng.integers(edge, count - edge - n))
    media = str(path.relative_to(DATA))
    return {"clip": f"{media_id(media)}@{start}", "media": media, "start": start, "end": start + n, "fps": fps,
            "width": w, "height": h, **meta}


def pick_clips(n: int, own: int, broadcast: int, seconds: float, seed: int) -> list[dict]:
    """YouTube clips (each video once before any twice, strata in turn), then own and broadcast."""
    rng = np.random.default_rng(seed)
    clips = []
    m = pd.read_csv(VIDEOS / "youtube" / "manifest.csv")
    ok = (m["status"].fillna("") == "done") & m["selected"].fillna(False).astype(bool) \
        & (m["play"].fillna("") != "doubles") & ~m["camera"].fillna("unknown").isin(EXCLUDED_CAMERAS)
    m = m[ok].sample(frac=1, random_state=seed)
    strata = [g.to_dict("records") for _, g in m.fillna("unknown").groupby(["camera", "surface", "level"])]
    rng.shuffle(strata)
    # Every video once before any twice (more videos, more diversity, a cleaner hold-out by video);
    # within each pass, one video per stratum in turn so small strata are not crowded out.
    order, want = [], max(n - own - broadcast, 0)
    while strata and len(order) < want:
        queues = [list(s) for s in strata]
        while any(queues) and len(order) < want:
            for q in queues:
                if q and len(order) < want:
                    order.append(q.pop(0))
    for v in order:
        segs = [p for p in sorted((VIDEOS / "youtube" / v["video_id"]).glob("seg*.mp4")) if not p.name.startswith("._")]
        if not segs:
            continue
        c = window(segs[rng.integers(len(segs))], seconds, rng, {
            "collection": "youtube", "video": v["video_id"], **{k: v[k] for k in ("camera", "surface", "level", "play")}})
        if c:
            clips.append(c)
    for k, (paths, meta) in enumerate((
            (sorted((RAW / "tennis_videos").glob("*.MOV")), {"collection": "own", "camera": "baseline_low",
                                                              "surface": "hard", "level": "rec", "play": "practice"}),
            (sorted((VIDEOS / "broadcast").glob("*.mp4")), {"collection": "broadcast", "camera": "broadcast",
                                                             "surface": "hard", "level": "pro", "play": "match"}))):
        paths = [p for p in paths if not p.name.startswith("._")]
        for j in range((own, broadcast)[k]):
            p = paths[j % len(paths)]
            c = window(p, seconds, rng, {**meta, "video": p.stem})
            if c:
                clips.append(c)
    return clips


def read_range(media: str, a: int, b: int):
    """Yield (frame, bgr) for frames a..b-1 of a video, decoding sequentially."""
    cap = cv2.VideoCapture(str(DATA / media))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(a, 0))
    for f in range(max(a, 0), b):
        ok, bgr = cap.read()
        if not ok:
            break
        yield f, bgr
    cap.release()


def clip_video(clip: dict, work: Path) -> bytes:
    """The clip re-encoded to near-lossless H.264 (CRF VIDEO_CRF), frame 0 = clip start: what Modal decodes.

    Encoded from the frames read_range decodes (piped raw), so frame i is exactly local frame start + i;
    seeking with ffmpeg instead drops frames of some variable-frame-rate sources. Near-lossless: already at
    CRF 10 (6 MB) it is closer to the source than JPEG q95 (PSNR 45 vs 44 dB).
    """
    out = work / "video" / f"{clip['clip']}.mp4"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.mp4")
        w, h = clip["width"], clip["height"]
        enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                                "-r", f"{clip['fps']}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                                "-crf", str(VIDEO_CRF), "-pix_fmt", "yuv420p", str(tmp)], stdin=subprocess.PIPE)
        n = 0
        for _, bgr in read_range(clip["media"], clip["start"], clip["end"]):
            enc.stdin.write(np.ascontiguousarray(bgr).tobytes())
            n += 1
        enc.stdin.close()
        if enc.wait() or n != clip["end"] - clip["start"]:
            raise RuntimeError(f"clip_video {clip['clip']}: ffmpeg {enc.returncode}, {n} frames")
        tmp.replace(out)
    return out.read_bytes()


class Timer:
    def __init__(self):
        self.sec, self.n = defaultdict(float), defaultdict(int)

    def add(self, name: str, sec: float, n: int = 1):
        self.sec[name] += sec
        self.n[name] += n

    def report(self) -> dict:
        return {k: {"seconds": round(self.sec[k], 1), "items": self.n[k],
                    "s_per_item": round(self.sec[k] / max(self.n[k], 1), 3)} for k in self.sec}


class Local:
    """WASB, and RF-DETR + ViTPose when pose runs locally; loaded once, timed separately."""

    def __init__(self, timer: Timer, pose: bool = True):
        import torch

        from dsa.astra.ball import Wasb
        from dsa.data.paths import MODELS
        from dsa.pose.backends import load_models

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pose = load_models("rfdetr", 1152, MODELS / "vitpose-plus-huge", self.device) if pose else None
        self.wasb = Wasb(self.device)
        self.t = timer

    def people(self, bgr: np.ndarray) -> list[dict]:
        from dsa.pose.backends import run_vitpose

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        xyxy, conf = self.pose.detector.detect(rgb)
        xyxy = xyxy[conf >= 0.4]
        t1 = time.time()
        xy, sc = run_vitpose(*self.pose.vitpose, self.device, rgb, xyxy)
        self.t.add("rfdetr", t1 - t0)
        self.t.add("vitpose", time.time() - t1)
        self.t.add("vitpose_people", 0, len(xyxy))
        return [{"box": b, "xy": k, "score": s} for b, k, s in zip(xyxy, xy, sc)]

    def submit(self, frames: list[np.ndarray]) -> "Done":
        return Done([self.people(f) for f in frames])

    def ball(self, frames: list[np.ndarray]) -> tuple[bool, float, float]:
        t0 = time.time()
        vis, x, y, _ = self.wasb.locate(frames)
        self.t.add("wasb", time.time() - t0)
        return vis, x, y


class Done:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def jpeg(bgr: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])[1].tobytes()


class Cost:
    """GPU seconds per Modal labeler and the containers used, from what the calls report."""

    def __init__(self):
        self.sec, self.frames, self.tasks, self.lock = defaultdict(float), defaultdict(int), defaultdict(set), threading.Lock()

    def add(self, name: str, r: dict, n: int):
        with self.lock:
            self.sec[(name, r["gpu"])] += r["seconds"]
            self.frames[name] += n
            self.tasks[name].add(r.get("task"))

    def report(self) -> dict:
        from dsa.cloud.pose import usd

        out = {}
        for (name, gpu), sec in self.sec.items():
            out[f"{name} on {gpu}"] = {"busy_seconds": round(sec, 1), "frames": self.frames[name],
                                       "containers": len(self.tasks[name]), "usd_busy": round(usd(gpu, sec), 3),
                                       "usd_per_1k_frames": round(usd(gpu, sec) / max(self.frames[name], 1) * 1000, 4)}
        return out


class ModalPose:
    """RF-DETR + ViTPose on Modal (dsa.cloud.pose): frames go up as JPEG in chunks, calls run in parallel."""

    def __init__(self, cost: Cost):
        from dsa.cloud.pose import Pose

        self.pose, self.chunk, self.cost = Pose(), 32, cost

    def submit_clip(self, video: bytes, offsets: list[int]) -> "Pending":
        return Pending([(len(offsets), self.pose.clip.spawn(video, offsets))] if offsets else [], self.cost)

    def submit(self, frames: list) -> "Pending":
        data = [f if isinstance(f, bytes) else jpeg(f) for f in frames]
        calls = [(len(data[i:i + self.chunk]), self.pose.people.spawn(data[i:i + self.chunk]))
                 for i in range(0, len(data), self.chunk)]
        return Pending(calls, self.cost)


class Offline:
    """Stands in for the pose and racket models in --offline: anything not cached is an error (clip skipped)."""

    def __getattr__(self, name):
        raise RuntimeError(f"--offline: not cached ({name})")


class Pending:
    def __init__(self, calls, cost: Cost):
        self.calls, self.cost, self.value, self.lock = calls, cost, None, threading.Lock()

    def get(self) -> list:
        with self.lock:                   # keyframe results are read from several Astra threads
            if self.value is None:
                value = []
                for n, call in self.calls:
                    r = call.get()
                    self.cost.add("pose", r, n)
                    value += r["people"]
                self.value = value
        return self.value


def scan(clip: dict, local: Local, timer: Timer) -> dict:
    """Pass 1: shots, camera shift and WASB on every frame of the clip.

    Camera shift: CameraTrack, from the shot's first frame.
    """
    a, b = clip["start"], clip["end"]
    hist_prev, cuts, shift, ball, camera = None, [a], {}, {}, CameraTrack()
    buf, t0, wasb0 = [], time.time(), timer.sec["wasb"]
    scale = clip["width"] / FLOW_W
    for f, bgr in read_range(clip["media"], a - 1, b + 1):
        if a <= f < b:
            small = cv2.resize(bgr, (FLOW_W, round(FLOW_W * bgr.shape[0] / bgr.shape[1])), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            hist = cv2.normalize(cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]), None).flatten()
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if hist_prev is not None and cv2.compareHist(hist_prev, hist, cv2.HISTCMP_BHATTACHARYYA) > CUT_BHATTACHARYYA \
                    and f - cuts[-1] >= MIN_SHOT_S * clip["fps"]:
                cuts.append(f)
            shift[f] = tuple(camera.step(gray, f == cuts[-1]) * scale)
            hist_prev = hist
        buf = (buf + [(f, bgr)])[-3:]
        if len(buf) == 3 and a <= buf[1][0] < b:
            ball[buf[1][0]] = local.ball([x for _, x in buf])
    timer.add("decode+cuts", time.time() - t0 - (timer.sec["wasb"] - wasb0), b - a)
    shots = [(c, n) for c, n in zip(cuts, cuts[1:] + [b])]
    fps = clip["fps"]
    grid = [a + round((KEY_S / 2 + KEY_S * k) * fps) for k in range(int((b - a) / fps / KEY_S) + 1)]
    keys = [g for g in grid if g < b]
    for s0, s1 in shots:
        if not any(s0 <= k < s1 for k in keys):
            keys.append((s0 + s1) // 2)
    return {"shots": shots, "keys": sorted(keys), "shift": shift, "ball": ball, "camera": CameraTrack.VERSION}


class CameraTrack:
    """Camera shift (pan as a 2D translation, FLOW_W pixels) since the shot's first frame, summed frame to frame:
    400 fresh corners of the previous frame tracked by Lucas-Kanade, kept if they track back to within
    0.5 px, and the median motion of those within 1 px of the median (players are a minority of corners).
    Fresh corners each frame: corners tracked from the shot start end up on the players as the background
    ones are lost, and their median jumps."""
    VERSION = 2

    def __init__(self):
        self.prev, self.cam = None, np.zeros(2)

    def step(self, gray: np.ndarray, cut: bool) -> np.ndarray:
        if self.prev is None or cut:
            self.cam = np.zeros(2)
        else:
            p0 = cv2.goodFeaturesToTrack(self.prev, 400, 0.01, 8)
            if p0 is not None and len(p0) >= 8:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, p0, None, winSize=(21, 21), maxLevel=3)
                pb, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None, winSize=(21, 21), maxLevel=3)
                ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < 0.5)
                d = (p1 - p0).reshape(-1, 2)[ok]
                if len(d) >= 8:
                    m = np.median(d, 0)
                    inl = np.linalg.norm(d - m, axis=1) < 1.0
                    self.cam = self.cam + (np.median(d[inl], 0) if inl.sum() >= 8 else m)
        self.prev = gray
        return self.cam.copy()


def camera_shift(clip: dict, shots: list[tuple[int, int]]) -> dict[int, tuple[float, float]]:
    """CameraTrack over a clip whose shots are known (to update a cached scan)."""
    cuts, camera, shift = {s0 for s0, _ in shots}, CameraTrack(), {}
    for f, bgr in read_range(clip["media"], clip["start"], clip["end"]):
        small = cv2.resize(bgr, (FLOW_W, round(FLOW_W * bgr.shape[0] / bgr.shape[1])), interpolation=cv2.INTER_AREA)
        shift[f] = tuple(camera.step(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), f in cuts) * clip["width"] / FLOW_W)
    return shift


def shot_of(shots, f: int) -> int:
    return next(i for i, (s0, s1) in enumerate(shots) if s0 <= f < s1)


def keyframe_items(clip: dict, keys: list[int], pose, work: Path) -> dict[int, dict]:
    """Keyframe images, 2 x 2 sheets 0.5 s apart, numbered-box images and people, for the Astra calls."""
    fps, need = clip["fps"], {}
    for k in keys:
        for o in (0, -0.75, -0.25, 0.25, 0.75):
            need.setdefault(k + round(o * fps), []).append(k)
    imgs = {}
    for f, bgr in read_range(clip["media"], min(need), max(need) + 1):
        if f in need:
            imgs[f] = bgr
    items = {}
    for k in keys:
        bgr = imgs[k]
        h, w = bgr.shape[:2]
        stem = f"{clip['clip']}__{k}"
        around = [imgs.get(k + round(o * fps), bgr) for o in (-0.75, -0.25, 0.25, 0.75)]
        paths = {n: work / f"{stem}{s}.jpg" for n, s in (("frame_path", ""), ("sheet_path", "_sheet"), ("boxes_path", "_boxes"),
                                                         ("sheet_v1_path", "_sheet_v1"), ("boxes_v1_path", "_boxes_v1"))}
        cv2.imwrite(str(paths["frame_path"]), bgr)
        cv2.imwrite(str(paths["sheet_path"]), sheet_image(around, "v0"))
        cv2.imwrite(str(paths["sheet_v1_path"]), sheet_image(around, "v1"))
        items[k] = {"row": {"sample": schema.sample_id("k", clip["clip"].replace("@", "_"), k)}, "bgr": bgr, **paths}
    # Cached: the numbered-box images, hence Astra's cache keys, must not move between reruns.
    cached = work / "keyframes" / f"{clip['clip']}.pkl"
    if cached.exists():
        calls = Done(pickle.loads(cached.read_bytes()))
    else:
        calls = pose.submit([items[k]["bgr"] for k in keys])
        calls.save = cached
    for i, k in enumerate(keys):
        items[k]["pending"] = (calls, i)
    return items


def astra_keyframe(item: dict, work: Path, timer: Timer) -> dict:
    """Numbered-box image, scene call, then head top for the chosen players (at most MAX_PLAYERS) if in scope."""
    calls, i = item.pop("pending")
    item["people"] = people = calls.get()[i]
    if getattr(calls, "save", None) and not calls.save.exists():
        calls.save.parent.mkdir(exist_ok=True)
        tmp = calls.save.with_suffix(f".{threading.get_ident()}.tmp")
        tmp.write_bytes(pickle.dumps(calls.get()))
        tmp.replace(calls.save)
    cv2.imwrite(str(item["boxes_path"]), draw_boxes(item["bgr"], people, "v0"))
    cv2.imwrite(str(item["boxes_v1_path"]), draw_boxes(item["bgr"], people, "v1"))
    t0 = time.time()
    try:                                  # keyframes labelled before the switch keep their paid v0 answer
        sc = astra_scene(item, work / "diagram.png", CACHE, "v0", cache_only=True)
        sc["variant"] = "v0"
    except NotCached:
        sc = astra_scene({**item, "sheet_path": item["sheet_v1_path"], "boxes_path": item["boxes_v1_path"]},
                         work / "diagram.png", CACHE, SCENE_VARIANT)
        sc["variant"] = SCENE_VARIANT
    timer.add("astra_scene", time.time() - t0)
    sc["singles"] = len(sc["players"]) <= MAX_PLAYERS
    sc["chosen"] = (sc["players"] or [int(i) for i in players(item["people"], sc["court"])][:MAX_PLAYERS]) \
        if sc["singles"] else []
    item.pop("bgr")                       # 100 clips of keyframes would hold GBs
    return sc


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (n, 4) and (m, 4) xyxy boxes."""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def track(frames: list[int], people: dict[int, list[dict]], start_id: int, fps: float
          ) -> tuple[dict[tuple[int, int], int], int]:
    """Tracker over the labelled frames of one shot. Returns {(frame, person): track id}.

    A track predicts its box centre from its velocity; a detection matches it when the centre is within
    MATCH_H (+ MATCH_H_PER_S per second of gap) box heights of the prediction and the heights agree
    (Hungarian on distance minus IoU, so big overlapping boxes still pair by IoU). Distance in box heights
    keeps small, fast far players; a lost track waits TRACK_GAP_S for its person to come back."""
    live: dict[int, dict] = {}
    out, nxt, wait = {}, start_id, max(1, round(TRACK_GAP_S * fps))
    for f in frames:
        boxes = np.array([p["box"] for p in people[f]], float).reshape(-1, 4)
        cen = (boxes[:, :2] + boxes[:, 2:]) / 2
        hts = boxes[:, 3] - boxes[:, 1]
        ids = list(live)
        assigned = {}
        if ids and len(boxes):
            cost = np.full((len(ids), len(boxes)), 1e6)
            ov = iou(np.array([live[t]["box"] for t in ids]), boxes)
            for r, t in enumerate(ids):
                L = live[t]
                gap = f - L["f"]
                d = np.linalg.norm(L["c"] + L["v"] * gap - cen, axis=1) / np.maximum(L["h"], hts)
                ratio = np.minimum(L["h"], hts) / np.maximum(L["h"], hts)
                ok = ((d <= MATCH_H + MATCH_H_PER_S * gap / fps) & (ratio >= MATCH_RATIO)) | (ov[r] >= IOU_MATCH)
                cost[r, ok] = d[ok] - ov[r, ok]
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] < 1e6:
                    assigned[c] = ids[r]
        for c in range(len(boxes)):
            if c not in assigned:
                assigned[c], nxt = nxt, nxt + 1
            t = assigned[c]
            if t in live:
                L = live[t]
                v = (cen[c] - L["c"]) / (f - L["f"])
                live[t] = {"box": boxes[c], "c": cen[c], "h": hts[c], "v": 0.5 * L["v"] + 0.5 * v, "f": f}
            else:
                live[t] = {"box": boxes[c], "c": cen[c], "h": hts[c], "v": np.zeros(2), "f": f}
            out[(f, c)] = t
        live = {t: L for t, L in live.items() if f - L["f"] <= wait}
    return out, nxt


def reacquire_gate(gap_frames: int, fps: float) -> float:
    """Box heights a player may move over a gap: a sprint is ~3 heights a second."""
    return MATCH_H + REACQ_H_PER_S * gap_frames / fps


def link_players(frames: list[int], people: dict[int, list[dict]], tracks: dict[tuple[int, int], int],
                 named: set[int], keyframes: set[int], court: np.ndarray | None, fps: float) -> None:
    """Re-acquire lost players: a player track that ends (or starts) mid-shot continues as the track that
    starts (or ends) within REACQ_S, nearest its last (first) box, in the playing area, when no other person
    there is nearly as close. Tracks Astra saw on a keyframe and did not name are never players. In place."""
    from dsa.label.consistency import box_in_area

    rows = defaultdict(list)                   # track -> [(frame, person)]
    for f in frames:
        for i in range(len(people[f])):
            rows[tracks[(f, i)]].append((f, i))
    seen_on_key = {t for t, r in rows.items() if any(f in keyframes for f, _ in r)}
    box = lambda fi: np.asarray(people[fi[0]][fi[1]]["box"], float)
    cen = lambda b: (b[:2] + b[2:]) / 2
    area_ok = lambda b: court is None or box_in_area(b, court)
    changed = True
    while changed:
        changed = False
        for p in sorted(named):
            if p not in rows:
                continue
            for end, ahead in ((rows[p][-1], True), (rows[p][0], False)):
                b0 = box(end)
                cands = []
                for t, r in rows.items():
                    if t in named or t in seen_on_key:
                        continue
                    first = r[0] if ahead else r[-1]
                    gap = first[0] - end[0] if ahead else end[0] - first[0]
                    if not 0 < gap <= REACQ_S * fps:
                        continue
                    b = box(first)
                    d = np.linalg.norm(cen(b) - cen(b0)) / max(b[3] - b[1], b0[3] - b0[1])
                    if d <= reacquire_gate(gap, fps) and area_ok(b):
                        cands.append((d, t, first[0]))
                if not cands:
                    continue
                d, t, f = min(cands)
                # anyone else in the playing area at that frame nearly as close: ambiguous, leave it
                others = [np.linalg.norm(cen(np.asarray(q["box"], float)) - cen(b0)) / max(q["box"][3] - q["box"][1], b0[3] - b0[1])
                          for i, q in enumerate(people[f]) if tracks[(f, i)] not in (t, p) and tracks[(f, i)] not in named
                          and area_ok(np.asarray(q["box"], float))]
                if any(o < REACQ_MARGIN * d + 0.1 for o in others):
                    continue
                moved = rows.pop(t)
                for fi in moved:
                    tracks[fi] = p
                rows[p] = sorted(rows[p] + moved)
                changed = True
                break
            if changed:
                break


def submit_clip(clip: dict, info: dict, scenes: dict[int, dict], items: dict[int, dict], pose, stride: int,
                rackets, work: Path) -> dict:
    """Pass 2, first half: scope per frame; people (and rackets) sent off for the in-scope frames."""
    shots, keys = info["shots"], info["keys"]
    kshot = {k: shot_of(shots, k) for k in keys}
    nearest, scope = {}, {}
    for f in range(clip["start"], clip["end"]):
        s = shot_of(shots, f)
        ks = [k for k in keys if kshot[k] == s]
        if not ks:
            scope[f] = "none"
            continue
        k = min(ks, key=lambda k: abs(k - f))
        nearest[f] = (k, [k for k in ks if k <= f][-1:] + [k for k in ks if k >= f][:1])
        sc = scenes[k]
        scope[f] = "nonview" if not sc["view"] else "doubles" if not sc["singles"] else "view"
    labelled = [f for f in range(clip["start"], clip["end"]) if scope[f] == "view"
                and ((f - clip["start"]) % stride == 0 or f in keys)]
    todo = [f for f in labelled if f not in keys]
    remote, start = isinstance(pose, ModalPose), clip["start"]
    cached = work / "modal" / f"{clip['clip']}.pkl"
    if cached.exists():
        c = pickle.loads(cached.read_bytes())
        if c["todo"] == todo and c["labelled"] == labelled and (rackets is None) == (c["dets"] is None):
            return {"clip": clip, "info": info, "scenes": scenes, "items": items, "scope": scope, "nearest": nearest,
                    "labelled": labelled, "todo": todo, "people": Done(c["people"]), "racket_calls": [],
                    "dets": c["dets"], "cache": cached}
    if isinstance(pose, Offline):
        raise RuntimeError(f"--offline: no Modal results cached for {clip['clip']}")
    video = clip_video(clip, work) if labelled and (remote or rackets is not None) else None
    if remote:
        people = pose.submit_clip(video, [f - start for f in todo])
    else:                                       # locally, run now rather than hold a clip of raw frames
        want, got = set(todo), {}
        if todo:
            for f, bgr in read_range(clip["media"], min(todo), max(todo) + 1):
                if f in want:
                    got[f] = pose.people(bgr)
        people = Done([got[f] for f in todo])
    racket_calls = [(labelled, rackets.clip.spawn(video, [f - start for f in labelled]))] \
        if rackets is not None and labelled else []
    return {"clip": clip, "info": info, "scenes": scenes, "items": items, "scope": scope, "nearest": nearest,
            "labelled": labelled, "todo": todo, "people": people, "racket_calls": racket_calls,
            "dets": None if rackets is None else {}, "cache": cached}


def finish_clip(st: dict, cost: Cost) -> tuple[dict[str, list], dict[str, list]]:
    """Pass 2, second half: tracks, players and the rows of every table for one clip; racket detections."""
    clip, info, scenes, items, scope, nearest, labelled = (st[k] for k in (
        "clip", "info", "scenes", "items", "scope", "nearest", "labelled"))
    shots, keys, fps = info["shots"], info["keys"], clip["fps"]
    kshot = {k: shot_of(shots, k) for k in keys}
    people = {k: items[k]["people"] for k in keys if scope.get(k) == "view"}
    people.update(zip(st["todo"], st["people"].get()))
    by_frame = dict(st["dets"] or {})
    for chunk, call in st["racket_calls"]:
        r = call.get()
        cost.add("rackets", r, len(chunk))
        by_frame.update({f: pred["detections"] for f, pred in zip(chunk, r["preds"])})
    if not st["cache"].exists():
        st["cache"].parent.mkdir(exist_ok=True)
        st["cache"].write_bytes(pickle.dumps({"todo": st["todo"], "labelled": labelled, "people": st["people"].get(),
                                              "dets": None if st["dets"] is None else by_frame}))
    dets = {schema.sample_id(clip["source"], media_id(clip["media"]), f): d for f, d in by_frame.items()}
    # Tracks within each shot; players are the tracks Astra named on a keyframe of the shot.
    tracks, nxt = {}, 0
    for s in range(len(shots)):
        fs = [f for f in labelled if shot_of(shots, f) == s]
        t, nxt = track(fs, people, nxt, fps)
        tracks.update(t)
    # A track is a player when Astra named it on most of the keyframes it is on (a whole-shot track would
    # otherwise take one keyframe's slip everywhere; dsa.label.consistency masks the close votes).
    votes = defaultdict(lambda: [0, 0])
    for k in keys:
        if scope.get(k) == "view":
            for i in range(len(people[k])):
                votes[(kshot[k], tracks[(k, i)])][i in scenes[k]["chosen"]] += 1
    player_tracks = defaultdict(set)
    for (sh, t), (no, yes) in votes.items():
        if yes > no:
            player_tracks[sh].add(t)
    for s in range(len(shots)):
        fs = [f for f in labelled if shot_of(shots, f) == s]
        ks = [j for j in keys if kshot[j] == s and scope.get(j) == "view"]
        courts = [scenes[j]["court"] for j in ks if scenes[j]["court"] is not None]
        court = None
        if courts:
            stack = np.stack(courts)
            court = np.column_stack([np.median(stack[:, :, :2], 0), np.where((stack[:, :, 2] > 0).mean(0) >= 0.5, 2.0, 0.0)])
        if fs and player_tracks[s]:
            link_players(fs, people, tracks, player_tracks[s], set(ks), court, fps)
    src = clip["source"]
    rows = defaultdict(list)
    for f in range(clip["start"], clip["end"]):
        s = schema.sample_id(src, media_id(clip["media"]), f)
        sh = shot_of(shots, f)
        heads = []
        if f in nearest:
            k, around = nearest[f]
            sc = scenes[k]
            plays = {scenes[j]["in_play"] for j in around}
            rows["scene"].append({"sample": s, "view": sc["view"], "in_play": plays.pop() if len(plays) == 1 else None,
                                  "labeler": ASTRA, "singles": sc["singles"]})
            heads.append("scene")
        if scope[f] == "view":
            vis, x, y = info["ball"].get(f, (False, np.nan, np.nan))
            rows["ball"].append({"sample": s, "x": x if vis else np.nan, "y": y if vis else np.nan, "visible": vis,
                                 "labeler": WASB})
            heads.append("ball")
            ks = [j for j in keys if kshot[j] == sh and scope.get(j) == "view" and scenes[j]["court"] is not None]
            if ks:
                stack = np.stack([scenes[j]["court"] for j in ks])
                court = np.column_stack([np.median(stack[:, :, :2], 0),
                                         np.where((stack[:, :, 2] > 0).mean(0) >= 0.5, 2.0, 0.0)])
                rows["court"].append({"sample": s, "kp": schema.flat(court), "labeler": ASTRA,
                                      "kf_kp": schema.flat(scenes[f]["court"]) if f in ks else None})
                heads.append("court")
        if f in labelled:
            heads.append("people")
            for i, p in enumerate(people[f]):
                kp = schema.empty_kp()
                vis = np.where(p["score"] >= 0.3, 2.0, 1.0)
                kp[:17] = np.column_stack([p["xy"], vis])
                kp[schema.KP["neck"]] = [*((p["xy"][5] + p["xy"][6]) / 2), min(vis[5], vis[6])]
                kp[schema.KP["head_top"]] = [*head_top_rule(np.asarray(p["box"]), p["xy"], p["score"]), 2.0]
                tid = tracks[(f, i)]
                rows["people"].append({"sample": s, "person": i, "track": f"{clip['clip']}#{tid}",
                                       "box": schema.flat(p["box"]), "kp": kp, "stroke": None,
                                       "player": tid in player_tracks[sh],
                                       "named": (i in scenes[f]["chosen"]) if f in keys else None,
                                       "labeler": PEOPLE + "; head top: rule",
                                       "_score": p["score"]})
            if st["dets"] is not None:
                heads.append("rackets")
        dx, dy = info["shift"].get(f, (np.nan, np.nan))
        rows["frames"].append({"sample": s, "source": src, "split": "train", "media": clip["media"], "frame": f,
                               "fps": fps, "width": clip["width"], "height": clip["height"], "clip": clip["clip"],
                               **{k: clip.get(k) for k in ("collection", "video", "camera", "surface", "level", "play")},
                               "shot": sh, "keyframe": f in keys, "scope": scope[f], "heads": ",".join(heads),
                               "cam_dx": dx, "cam_dy": dy})
    return rows, dets


def attach_rackets(rows: dict[str, list], dets: dict[str, list]) -> None:
    """Racket rows, and each racket's points onto the player whose wrist is nearest its handle."""
    by_sample = defaultdict(list)
    for r in rows["people"]:
        by_sample[r["sample"]].append(r)
    for s, ds in dets.items():
        for n, det in enumerate(ds):
            rkp = np.column_stack([np.array(det["kp"]), np.where(np.array(det["kp_score"]) >= 0.3, 2.0, 1.0)])
            owner, best = None, np.inf
            for r in by_sample[s]:
                if not r["player"]:
                    continue
                h = r["box"][3] - r["box"][1]
                d = min(np.linalg.norm(rkp[2, :2] - r["kp"][w, :2]) for w in (9, 10))
                if d < 0.25 * h and d < best:
                    owner, best = r, d
            rows["rackets"].append({"sample": s, "racket": n, "box": schema.flat(det["box"]), "kp": schema.flat(rkp),
                                    "person": np.nan if owner is None else owner["person"], "labeler": RACKETS})
            if owner is not None:
                owner["kp"][schema.KP["racket_tip"]:] = rkp
                owner["labeler"] += "; racket: " + RACKETS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="labels source name, e.g. clips_v1")
    p.add_argument("--clips", type=int, default=60, help="clips in total, own and broadcast included")
    p.add_argument("--own", type=int, default=4)
    p.add_argument("--broadcast", type=int, default=4)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--stride", type=int, default=1, help="people and rackets on every n-th frame (ball every frame)")
    p.add_argument("--workers", type=int, default=4, help="concurrent Astra calls")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--clip-list", help="clips.json of an earlier run, to redo the same clips (the manifest grows)")
    p.add_argument("--no-rackets", action="store_true", help="skip the Modal racket step")
    p.add_argument("--pose", choices=["modal", "local"], default="modal",
                   help="RF-DETR + ViTPose on Modal GPUs (dsa.cloud.pose) or on this machine")
    p.add_argument("--offline", action="store_true",
                   help="rebuild the tables from the caches only (scan, keyframes, Astra, Modal): no Modal, no "
                        "Astra calls; clips missing a cache are skipped")
    a = p.parse_args()
    faulthandler.register(signal.SIGUSR1, all_threads=True)      # kill -USR1 <pid> prints every thread's stack

    work = SCRATCH / "clips" / a.out
    work.mkdir(parents=True, exist_ok=True)
    clips = json.loads(Path(a.clip_list).read_text()) if a.clip_list else \
        pick_clips(a.clips, a.own, a.broadcast, a.seconds, a.seed)
    for c in clips:
        c["source"] = a.out
    (work / "clips.json").write_text(json.dumps(clips, indent=1, default=str))
    print(f"{len(clips)} clips, {sum(c['end'] - c['start'] for c in clips)} frames -> {work / 'clips.json'}", flush=True)
    diagram(work / "diagram.png")
    timer, cost = Timer(), Cost()
    t_start = time.time()
    if a.offline:
        import os
        os.environ["DSA_ASTRA_CACHE_ONLY"] = "1"         # dsa.astra.codex.ask: a miss raises instead of calling
        a.pose, a.no_rackets = "offline", True
    local = Local(timer, pose=a.pose == "local")
    timer.add("load_models", time.time() - t_start)

    import contextlib
    apps, rackets = [], None
    if a.pose == "modal":
        from dsa.cloud.pose import app as pose_app
        apps.append(pose_app)
    if not a.no_rackets:
        from dsa.cloud.racket_pose import Rackets, app as racket_app
        apps.append(racket_app)
    rows, dets = defaultdict(list), {}
    with contextlib.ExitStack() as stack, ThreadPoolExecutor(a.workers) as pool:
        for app in apps:
            stack.enter_context(app.run())
        pose = ModalPose(cost) if a.pose == "modal" else Offline() if a.pose == "offline" else local
        if not a.no_rackets:
            rackets = Rackets()
        elif a.offline:
            rackets = Offline()                          # the cached racket detections are kept
        # Pass 1 on the main thread (WASB on the local GPU); each clip's pass 2 goes to Modal from a second
        # thread as soon as its Astra answers are in, so Astra, uploads and Modal overlap the scan.
        # With local pose, pass 2 runs after the scan instead: the local GPU is busy with WASB until then.
        t0, remote = time.time(), a.pose == "modal"

        failed = {}

        def second(c, info, items, futs):
            try:
                scenes = {k: f.result() for k, f in futs.items()}
            except Exception as e:                 # e.g. Astra out of credits: rerun later, caches keep the rest
                failed[c["clip"]] = str(e).strip().splitlines()[-1][:200]
                log(f"skip {c['clip']}: {failed[c['clip']]}")
                return None
            try:
                st = submit_clip(c, info, scenes, items, pose, a.stride, rackets, work)
            except Exception as e:                 # e.g. Modal spend limit: keep what was already paid for
                failed[c["clip"]] = str(e).strip().splitlines()[-1][:200]
                log(f"skip {c['clip']}: {failed[c['clip']]}")
                return None
            log(f"submit {c['clip']}: {sum(s['view'] for s in scenes.values())}/{len(scenes)} view keyframes, "
                f"{len(st['labelled'])} frames for people")
            return st

        with ThreadPoolExecutor(1) as submitter:
            later = []
            for c in clips:
                cached = work / "scan" / f"{c['clip']}.pkl"
                if cached.exists():
                    info = pickle.loads(cached.read_bytes())
                    if info.get("camera") != CameraTrack.VERSION:     # an older camera track: redo just that
                        info["shift"], info["camera"] = camera_shift(c, info["shots"]), CameraTrack.VERSION
                        cached.write_bytes(pickle.dumps(info))
                else:
                    info = scan(c, local, timer)
                    cached.parent.mkdir(exist_ok=True)
                    cached.write_bytes(pickle.dumps(info))
                real_end = max(info["shift"], default=c["start"] - 1) + 1
                if real_end < c["end"]:           # the container's frame count overestimates: the video ends early
                    log(f"{c['clip']}: video ends at frame {real_end}, not {c['end']}")
                    c["end"] = real_end
                    info["shots"] = [(s0, min(s1, real_end)) for s0, s1 in info["shots"] if s0 < real_end]
                    info["keys"] = [k for k in info["keys"] if k < real_end]
                try:
                    items = keyframe_items(c, info["keys"], pose, work)
                except Exception as e:             # e.g. Modal spend limit
                    failed[c["clip"]] = str(e).strip().splitlines()[-1][:200]
                    log(f"skip {c['clip']}: {failed[c['clip']]}")
                    continue
                futs = {k: pool.submit(astra_keyframe, it, work, timer) for k, it in items.items()}
                job = (c, info, items, futs)
                later.append(submitter.submit(second, *job) if remote else job)
                log(f"scan {c['clip']}: {len(info['shots'])} shots, {len(info['keys'])} keyframes")
            timer.add("pass1_wall", time.time() - t0, len(clips))
            states = [f.result() for f in later] if remote else [second(*job) for job in later]
            states = [st for st in states if st is not None]
            (work / "failed.json").write_text(json.dumps(failed, indent=1))
            if failed:
                log(f"{len(failed)} clips skipped (see {work / 'failed.json'}); rerun the same command to finish them")
        for st in states:
            try:
                r, d = finish_clip(st, cost)
            except Exception as e:                 # a failed Modal call: the other clips still get written
                log(f"skip {st['clip']['clip'] if isinstance(st.get('clip'), dict) else '?'}: {str(e).strip().splitlines()[-1][:200]}")
                continue
            for k, v in r.items():
                rows[k] += v
            dets.update(d)
        timer.add("pass2_wall", time.time() - t0, sum(len(st["labelled"]) for st in states))
    attach_rackets(rows, dets)
    timer.add("total_wall", time.time() - t_start)

    for r in rows["people"]:
        r["kp"] = schema.flat(r["kp"])
        r.pop("_score")
    tables = {"frames": pd.DataFrame(rows["frames"]),
              "people": pd.DataFrame(rows["people"], columns=schema.TABLES["people"] + ["player", "named"]),
              "rackets": pd.DataFrame(rows["rackets"], columns=schema.TABLES["rackets"]),
              "ball": pd.DataFrame(rows["ball"], columns=schema.TABLES["ball"]),
              "court": pd.DataFrame(rows["court"], columns=schema.TABLES["court"] + ["kf_kp"]),
              "scene": pd.DataFrame(rows["scene"], columns=schema.TABLES["scene"] + ["singles"])}
    for t, c in (("scene", "view"), ("scene", "in_play"), ("scene", "singles"), ("people", "named")):
        tables[t][c] = tables[t][c].astype("boolean")
    out = schema.write(a.out, tables)
    report = timer.report()
    (work / "timings.json").write_text(json.dumps({"local": report, "modal": cost.report()}, indent=1))
    print(json.dumps({k: len(v) for k, v in tables.items()}), "->", out)
    print(pd.DataFrame(report).T.to_string())
    if cost.sec:
        print(pd.DataFrame(cost.report()).T.to_string())


if __name__ == "__main__":
    main()
