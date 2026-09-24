"""Pre-fill labels over short video clips: every labeler, every frame, tracked.

Clips are 10 s windows (--seconds) drawn from the YouTube manifest, stratified
round-robin over camera x surface x level with one clip per video before any
video repeats, plus own recordings and broadcast matches. Singles from behind
the baseline only for now: manifest rows with play == doubles or camera
drone / fence / side are skipped.

Per clip:
  shots      cuts where the HSV histogram of consecutive frames jumps
             (CUT_BHATTACHARYYA); camera pan within a shot is the median
             Lucas-Kanade motion of background corners (frames.cam_dx / cam_dy,
             px from the shot's first frame).
  keyframes  every KEY_S seconds, plus the middle of any shot without one.
             Astra (dsa.label.prefill.scene_prompt) gives view, in play, the
             players (numbered boxes) and 14 court points; then head top for
             the players (feet deferred as in prefill).
  scope      a frame takes view / singles from the nearest keyframe of its
             shot. Out of scope (not a view, or Astra names > 2 players, i.e.
             doubles or crowd) -> scene only. Frames in a shot without a
             keyframe are not labelled at all.
  people     RF-DETR Medium @ 1152 + ViTPose-Plus-Huge on every --stride'th
             in-scope frame (keyframes always), tracked by IoU + Hungarian
             within a shot. A track is a player if Astra named it on any
             keyframe of the shot (at most 2; people.named is Astra's own call
             on keyframes); head top on keyframes only.
  ball       WASB on t-1, t, t+1, every in-scope frame.
  court      per shot, the per-point median of Astra's keyframe answers,
             written on every in-scope frame of the shot (keyframe rows keep
             their own answer in court.kf_kp, for dsa.label.consistency).
  scene      every frame with a keyframe in its shot; in_play only where the
             keyframes around it agree (NaN across a change). scene.singles.
  rackets    RacketVision on Modal on the same frames as people (--no-rackets).

frames gets extra columns clip, collection, video, camera, surface, level, play,
shot, keyframe, scope (view | nonview | doubles | none), heads (comma list of
heads labelled), cam_dx, cam_dy. The clips picked go to data/scratch/clips/<out>/clips.json
(--clip-list redoes them), timings to timings.json beside it.

    python -m dsa.label.clips --clips 60 --out clips_v1
    python -m dsa.label.clips --clips 5 --own 1 --broadcast 1 --seconds 6 --stride 2 --out clips_smoke
"""
from __future__ import annotations

import argparse
import json
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
from dsa.label.prefill import ASTRA, PEOPLE, RACKETS, WASB, astra_feet, astra_scene, players

KEY_S = 2.0                    # Astra keyframe spacing: in play changes on a scale of seconds, a call costs ~20 s
FLOW_W = 640                   # px width the camera shift is tracked at
CUT_BHATTACHARYYA = 0.35       # histogram distance of a hard cut; pans and players moving stay under ~0.15
MIN_SHOT_S = 0.3               # cuts closer than this are one transition (fades, flashes)
IOU_MATCH = 0.3                # a person box overlaps its previous self at least this much at >= 10 fps
TRACK_GAP = 5                  # labelled frames a lost track may reappear within
MAX_PLAYERS = 2                # singles: more named players means doubles or bystanders, out of scope
EXCLUDED_CAMERAS = {"drone", "fence", "side"}
RACKET_CHUNK = 64              # frames per Modal call: ~20 MB of JPEG
CACHE = Path("output/astra_cache/prefill")


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
    """YouTube clips stratified round-robin over camera x surface x level, then own and broadcast."""
    rng = np.random.default_rng(seed)
    clips = []
    m = pd.read_csv(VIDEOS / "youtube" / "manifest.csv")
    ok = (m["status"].fillna("") == "done") & m["selected"].fillna(False).astype(bool) \
        & (m["play"].fillna("") != "doubles") & ~m["camera"].fillna("unknown").isin(EXCLUDED_CAMERAS)
    m = m[ok].sample(frac=1, random_state=seed)
    strata = [g.to_dict("records") for _, g in m.fillna("unknown").groupby(["camera", "surface", "level"])]
    rng.shuffle(strata)
    used: dict[str, int] = defaultdict(int)
    want = max(n - own - broadcast, 0)
    for rnd in range(100):
        for s in strata:
            if len(clips) >= want:
                break
            v = min(s, key=lambda r: used[r["video_id"]])
            if used[v["video_id"]] > rnd:
                continue
            segs = [p for p in sorted((VIDEOS / "youtube" / v["video_id"]).glob("seg*.mp4")) if not p.name.startswith("._")]
            used[v["video_id"]] += 1
            if not segs:
                continue
            c = window(segs[rng.integers(len(segs))], seconds, rng, {
                "collection": "youtube", "video": v["video_id"],
                **{k: v[k] for k in ("camera", "surface", "level", "play")}})
            if c:
                clips.append(c)
        if len(clips) >= want:
            break
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
    """RF-DETR + ViTPose and WASB, loaded once, timed separately."""

    def __init__(self, timer: Timer):
        import torch

        from dsa.astra.ball import Wasb
        from dsa.data.paths import MODELS
        from dsa.pose.backends import load_models

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pose = load_models("rfdetr", 1152, MODELS / "vitpose-plus-huge", self.device)
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

    def ball(self, frames: list[np.ndarray]) -> tuple[bool, float, float]:
        t0 = time.time()
        vis, x, y, _ = self.wasb.locate(frames)
        self.t.add("wasb", time.time() - t0)
        return vis, x, y


def scan(clip: dict, local: Local, timer: Timer) -> dict:
    """Pass 1: shots, camera shift and WASB on every frame of the clip.

    Camera shift is the median motion of up to 400 corners tracked by
    Lucas-Kanade from the shot's first frame (re-seeded when fewer than 20
    survive): players are a minority of corners, so the median is the camera.
    """
    a, b = clip["start"], clip["end"]
    hist_prev, cuts, shift, ball = None, [a], {}, {}
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
            if f == cuts[-1] or len(pts) < 20:
                base = np.zeros(2) if f == cuts[-1] else cam
                pts0 = cv2.goodFeaturesToTrack(gray, 400, 0.01, 8)
                pts0 = np.zeros((0, 2), np.float32) if pts0 is None else pts0.reshape(-1, 2)
                pts = pts0.copy()
            else:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray_prev, gray, pts.reshape(-1, 1, 2), None, winSize=(21, 21),
                                                      maxLevel=3)
                ok = st.ravel() == 1
                pts0, pts = pts0[ok], nxt.reshape(-1, 2)[ok]
            cam = base + (np.median(pts - pts0, 0) if len(pts) else 0)
            shift[f] = tuple(cam * scale)
            hist_prev, gray_prev = hist, gray
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
    return {"shots": shots, "keys": sorted(keys), "shift": shift, "ball": ball}


def shot_of(shots, f: int) -> int:
    return next(i for i, (s0, s1) in enumerate(shots) if s0 <= f < s1)


def keyframe_items(clip: dict, keys: list[int], local: Local, work: Path) -> dict[int, dict]:
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
        sheet = [cv2.resize(imgs.get(k + round(o * fps), bgr), (640, round(640 * h / w))) for o in (-0.75, -0.25, 0.25, 0.75)]
        paths = {n: work / f"{stem}{s}.jpg" for n, s in (("frame_path", ""), ("sheet_path", "_sheet"), ("boxes_path", "_boxes"))}
        cv2.imwrite(str(paths["frame_path"]), bgr)
        cv2.imwrite(str(paths["sheet_path"]), np.vstack([np.hstack(sheet[:2]), np.hstack(sheet[2:])]))
        people = local.people(bgr)
        boxed = bgr.copy()
        for i, p in enumerate(people):
            x1, y1, x2, y2 = (int(v) for v in p["box"])
            cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(boxed, str(i), (x1, max(y1 - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(str(paths["boxes_path"]), boxed)
        items[k] = {"row": {"sample": schema.sample_id("k", clip["clip"].replace("@", "_"), k)}, "bgr": bgr,
                    "people": people, **paths}
    return items


def astra_keyframe(item: dict, work: Path, timer: Timer) -> dict:
    """Scene call, then head top for the chosen players (at most MAX_PLAYERS) if in scope."""
    t0 = time.time()
    sc = astra_scene(item, work / "diagram.png", CACHE)
    timer.add("astra_scene", time.time() - t0)
    sc["singles"] = len(sc["players"]) <= MAX_PLAYERS
    sc["chosen"] = (sc["players"] or [int(i) for i in players(item["people"], sc["court"])][:MAX_PLAYERS]) \
        if sc["singles"] else []
    sc["head"] = {}
    if sc["view"] and sc["singles"] and sc["chosen"]:
        t0 = time.time()
        sc["head"] = astra_feet(item, sc["chosen"], work, CACHE)
        timer.add("astra_head", time.time() - t0)
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


def track(frames: list[int], people: dict[int, list[dict]], start_id: int) -> tuple[dict[tuple[int, int], int], int]:
    """IoU + Hungarian tracker over the labelled frames of one shot. Returns {(frame, person): track id}."""
    live: dict[int, tuple[np.ndarray, int]] = {}      # id -> (last box, frames since seen)
    out, nxt = {}, start_id
    for f in frames:
        boxes = np.array([p["box"] for p in people[f]], float).reshape(-1, 4)
        ids = list(live)
        assigned = {}
        if ids and len(boxes):
            m = iou(np.array([live[i][0] for i in ids]), boxes)
            for r, c in zip(*linear_sum_assignment(-m)):
                if m[r, c] >= IOU_MATCH:
                    assigned[c] = ids[r]
        for i in list(live):
            live[i] = (live[i][0], live[i][1] + 1)
        for c in range(len(boxes)):
            if c not in assigned:
                assigned[c], nxt = nxt, nxt + 1
            live[assigned[c]] = (boxes[c], 0)
            out[(f, c)] = assigned[c]
        live = {i: v for i, v in live.items() if v[1] <= TRACK_GAP}
    return out, nxt


def label_clip(clip: dict, info: dict, scenes: dict[int, dict], items: dict[int, dict], local: Local, stride: int,
               racket_calls: list, predict, timer: Timer) -> dict[str, list]:
    """Pass 2: people on in-scope frames, tracks, players, and the rows of every table for one clip."""
    shots, keys, fps = info["shots"], info["keys"], clip["fps"]
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
    people = {k: items[k]["people"] for k in keys if scope.get(k) == "view"}
    jpegs = {}
    todo = set(labelled)
    if todo:
        for f, bgr in read_range(clip["media"], min(todo), max(todo) + 1):
            if f not in todo:
                continue
            if f not in people:
                people[f] = local.people(bgr)
            if predict is not None:
                jpegs[f] = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
    if predict is not None and jpegs:
        fs = sorted(jpegs)
        for i in range(0, len(fs), RACKET_CHUNK):
            chunk = fs[i:i + RACKET_CHUNK]
            racket_calls.append((clip["clip"], chunk, time.time(), predict.spawn([jpegs[f] for f in chunk], [[] for _ in chunk])))
    # Tracks within each shot; players are the tracks Astra named on a keyframe of the shot.
    tracks, nxt = {}, 0
    for s in range(len(shots)):
        fs = [f for f in labelled if shot_of(shots, f) == s]
        t, nxt = track(fs, people, nxt)
        tracks.update(t)
    player_tracks = defaultdict(set)
    for k in keys:
        if scope.get(k) == "view":
            player_tracks[kshot[k]] |= {tracks[(k, i)] for i in scenes[k]["chosen"] if (k, i) in tracks}
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
                head = scenes[f]["head"].get(i) if f in keys else None
                if head is not None:
                    kp[schema.KP["head_top"]] = [*head["head_top"], 2.0]
                tid = tracks[(f, i)]
                rows["people"].append({"sample": s, "person": i, "track": f"{clip['clip']}#{tid}",
                                       "box": schema.flat(p["box"]), "kp": kp, "stroke": None,
                                       "player": tid in player_tracks[sh],
                                       "named": (i in scenes[f]["chosen"]) if f in keys else None,
                                       "labeler": PEOPLE + (f"; head top: {ASTRA}" if head is not None else ""),
                                       "_score": p["score"]})
            if predict is not None:
                heads.append("rackets")
        dx, dy = info["shift"].get(f, (np.nan, np.nan))
        rows["frames"].append({"sample": s, "source": src, "split": "train", "media": clip["media"], "frame": f,
                               "fps": fps, "width": clip["width"], "height": clip["height"], "clip": clip["clip"],
                               **{k: clip.get(k) for k in ("collection", "video", "camera", "surface", "level", "play")},
                               "shot": sh, "keyframe": f in keys, "scope": scope[f], "heads": ",".join(heads),
                               "cam_dx": dx, "cam_dy": dy})
    return rows


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
    a = p.parse_args()

    work = SCRATCH / "clips" / a.out
    work.mkdir(parents=True, exist_ok=True)
    clips = json.loads(Path(a.clip_list).read_text()) if a.clip_list else \
        pick_clips(a.clips, a.own, a.broadcast, a.seconds, a.seed)
    for c in clips:
        c["source"] = a.out
    (work / "clips.json").write_text(json.dumps(clips, indent=1, default=str))
    print(f"{len(clips)} clips, {sum(c['end'] - c['start'] for c in clips)} frames -> {work / 'clips.json'}", flush=True)
    diagram(work / "diagram.png")
    timer = Timer()
    t_start = time.time()
    local = Local(timer)
    timer.add("load_models", time.time() - t_start)

    import contextlib
    predict, ctx = None, contextlib.nullcontext()
    if not a.no_rackets:
        from dsa.cloud.racket_pose import app, predict
        ctx = app.run()
    rows, racket_calls = defaultdict(list), []
    with ctx, ThreadPoolExecutor(a.workers) as pool:
        # Pass 1 for every clip first, so Astra works while the GPU does.
        pending = []
        for c in clips:
            info = scan(c, local, timer)
            items = keyframe_items(c, info["keys"], local, work)
            futs = {k: pool.submit(astra_keyframe, it, work, timer) for k, it in items.items()}
            pending.append((c, info, items, futs))
            print(f"scan {c['clip']}: {len(info['shots'])} shots, {len(info['keys'])} keyframes", flush=True)
        for c, info, items, futs in pending:
            scenes = {k: f.result() for k, f in futs.items()}
            for k, v in label_clip(c, info, scenes, items, local, a.stride, racket_calls, predict, timer).items():
                rows[k] += v
            print(f"label {c['clip']}: {sum(s['view'] for s in scenes.values())}/{len(scenes)} view keyframes",
                  flush=True)
        dets = {}
        if racket_calls:
            t0, n = min(r[2] for r in racket_calls), 0
            for clip, chunk, _, call in racket_calls:
                src = next(c for c in clips if c["clip"] == clip)
                for f, pred in zip(chunk, call.get()):
                    dets[schema.sample_id(a.out, media_id(src["media"]), f)] = pred["detections"]
                    n += 1
            timer.add("rackets_modal_wall", time.time() - t0, n)
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
    (work / "timings.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: len(v) for k, v in tables.items()}), "->", out)
    print(pd.DataFrame(report).T.to_string())


if __name__ == "__main__":
    main()
