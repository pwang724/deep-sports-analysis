"""Pre-fill labels on sampled frames with every labeler chosen in phase 0.

Input is a frames table (dsa.data.schema) pointing into videos; output is a
labels source with the same samples, ready for the review tool:

  people   RF-DETR Medium @ 1152 (conf >= 0.4) + ViTPose-Plus-Huge: box and 17
           body joints for every person; neck = shoulder midpoint.
  court    Astra, the frame plus a numbered court diagram (dsa.astra.court).
  scene    Astra in the same call: view (behind a baseline, any height, whole
           court) and in play (serve toss to end of rally), with a 2 x 2 sheet
           of frames 0.5 s apart for in play.
  head top Astra, one call per frame with a crop per player. The same call
           returns 6 foot points (side from the nearest ViTPose ankle); they
           are deferred (FEET = False) and left unlabelled.
           Players are the boxes Astra names in the scene call (shown the frame
           with numbered boxes); if it names none, up to 4 people in the playing
           area (court + 8 m behind the baselines, 3.5 m beside, not the
           umpire's band at the net), else the two largest.
  ball     WASB on frames t-1, t, t+1.
  rackets  RacketVision RTMDet + RTMPose on Modal; each racket goes to the
           player whose wrist is nearest its handle (within 0.25 x height).

    python -m dsa.label.prefill data/gold/v1/frames.parquet --out gold_v1_prefill
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dsa.astra.codex import MODEL, EFFORT, ask
from dsa.astra.court import POINTS as COURT_NAMES, REF, diagram
from dsa.astra.joints import upscaled_crop
from dsa.data import schema
from dsa.data.paths import DATA, SCRATCH

ASTRA = f"astra:{MODEL}:{EFFORT}"
PEOPLE = "rfdetr-medium@1152+vitpose-plus-huge"
WASB = "wasb-tennis"
RACKETS = "racketvision-rtmdet-m+rtmpose-m"
FOOT_KEYS = ["l_ankle", "r_ankle", "l_big_toe", "l_small_toe", "l_heel", "r_big_toe", "r_small_toe", "r_heel",
             "head_top"]
# Feet are deferred: off by several pixels on broadcast-size players, so they stay unlabelled (NaN).
# Astra still returns them in the head-top call and they are cached, so turning this on costs nothing.
FEET = False
METRE = 1117 / 10.97            # reference-court units per metre (dsa.astra.court REF)

POINT = {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
         "required": ["x", "y"], "additionalProperties": False}
SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "view": {"type": "boolean"}, "in_play": {"type": "boolean"}, "court_visible": {"type": "boolean"},
        "players": {"type": "array", "items": {"type": "integer"}},
        "points": {"type": "array", "minItems": 14, "maxItems": 14, "items": {
            "type": "object", "properties": {"in_view": {"type": "boolean"}, "x": {"type": "number"},
                                              "y": {"type": "number"}},
            "required": ["in_view", "x", "y"], "additionalProperties": False}}},
    "required": ["view", "in_play", "court_visible", "players", "points"], "additionalProperties": False,
}
FEET_SCHEMA = {
    "type": "object",
    "properties": {"people": {"type": "array", "items": {
        "type": "object", "properties": {k: POINT for k in FOOT_KEYS}, "required": FOOT_KEYS,
        "additionalProperties": False}}},
    "required": ["people"], "additionalProperties": False,
}


def scene_prompt(w: int, h: int, variant: str = "v0") -> str:
    if variant != "v0":
        return slim_scene_prompt([(w, h)]) if variant == "v1" else slim_scene_prompt([(w, h)] * 4)
    return (
        f"The first image is a {w} x {h} frame of a tennis video. The second is a diagram of a tennis court "
        "seen from above, near baseline at the bottom, with 14 numbered points where lines cross. The third "
        "shows four frames 0.5 s apart around the first image's moment, in reading order. The fourth is the "
        "first image with every detected person boxed and numbered.\n"
        "view: is the first image a usable analysis view: a camera behind one baseline, at any height, "
        "looking down the court, with the whole court (or all of it but the near corners) in view? This is "
        "about the camera only: True even when players are off court, changing ends or out of frame. False "
        "for side-on or other angles, close-ups where the court is cut off, player close-ups, replays, crowd "
        "and graphics.\n"
        "in_play: on the main court (the one whose baseline is nearest the camera), is a point being played "
        "at this moment, from the serve toss to the end of the rally, warm-up rallies included? False when "
        "players are collecting balls, walking, bouncing the ball before a serve, resting or the court is "
        "empty.\n"
        "players: the numbers of the boxes (fourth image) around people playing on the main court: the "
        "singles or doubles players, not umpires, line judges, ball kids, coaches, spectators or players on "
        "other courts. Empty if nobody is playing.\n"
        "court_visible: is a tennis court visible at all? If so, give the pixel coordinates in the first "
        "image of the 14 points of the main court, in order:\n"
        + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(COURT_NAMES))
        + "\nLeft and right as seen from the near baseline. Set in_view false for a point outside the frame "
        "or hidden, with your best estimate. If no court is visible, return 14 points at 0, 0 with in_view "
        "false. Look at the images directly; do not run any commands."
    )


# Scene-call variants (dsa.astra.scene_cost compares them; clips uses v0):
#   v0  keyframe, diagram, 2 x 2 sheet of 640 px tiles, keyframe with numbered boxes.
#   v1  slim: diagram, 320 px sheet, keyframe with thin numbered corner boxes; trimmed prompt.
#   v2  v1 for 4 keyframes in one call: diagram once, then sheet + boxed frame per moment.
VARIANTS = ("v0", "v1", "v2")
SHEET_WIDTH = {"v0": 640, "v1": 320, "v2": 320}
BATCH = 4
BATCH_SCHEMA = {"type": "object", "properties": {"moments": {"type": "array", "minItems": BATCH, "maxItems": BATCH,
                                                             "items": SCENE_SCHEMA}},
                "required": ["moments"], "additionalProperties": False}


def slim_scene_prompt(sizes: list[tuple[int, int]]) -> str:
    """Trimmed scene prompt for one moment (v1) or len(sizes) moments (v2)."""
    if len(sizes) == 1:
        head = ("Image 1: a numbered diagram of a tennis court from above, near baseline at the bottom, 14 points "
                "where lines cross. Image 2: four frames 0.5 s apart in reading order, around the moment of image 3. "
                f"Image 3: the {sizes[0][0]} x {sizes[0][1]} frame at that moment, each detected person marked by "
                "numbered corner brackets.\n")
        frame = "image 3"
    else:
        head = ("Image 1: a numbered diagram of a tennis court from above, near baseline at the bottom, 14 points "
                f"where lines cross. Then {len(sizes)} moments, possibly from different videos, two images each: "
                "first four frames 0.5 s apart in reading order (labelled with the moment number), then the frame at that moment, each "
                "detected person marked by numbered corner brackets. "
                + "; ".join(f"Moment {k + 1}: images {2 * k + 2}-{2 * k + 3}, frame {w} x {h}"
                            for k, (w, h) in enumerate(sizes))
                + f". Return `moments`: {len(sizes)} answers in moment order, each judged on its own images only.\n")
        frame = "that moment's frame"
    return (
        head
        + "view: a camera behind one baseline (any height) looking down the court, with the whole court (bar the "
        "near corners) in view? Camera only: true even if players are off court. False for side-on or other "
        "angles, court cut off, close-ups, replays, crowd, graphics.\n"
        "in_play: on the main court (baseline nearest the camera), is a point under way, from serve toss to end "
        "of rally, warm-ups included? False when collecting balls, walking, bouncing the ball before a serve, "
        "resting, or the court is empty.\n"
        "players: numbers of the people playing on the main court (not umpires, line judges, ball kids, coaches, "
        "spectators, other courts); empty if none.\n"
        f"court_visible: is a tennis court visible? If so, pixel coordinates in {frame} of the main court's 14 "
        "points, in order:\n"
        + "\n".join(f"{i + 1}. {p}" for i, p in enumerate(COURT_NAMES))
        + "\nLeft and right as seen from the near baseline. in_view false for a point outside the frame or hidden "
        "(best estimate). No court: 14 points at 0, 0, in_view false. Look at the images directly; do not run "
        "any commands."
    )


def draw_boxes(bgr: np.ndarray, people: list[dict], variant: str = "v0") -> np.ndarray:
    """The frame with every person's numbered box: v0 full 2 px boxes, else 1 px corner brackets that leave
    most of each side (and the court lines under it) clear."""
    boxed = bgr.copy()
    for i, person in enumerate(people):
        x1, y1, x2, y2 = (int(v) for v in person["box"])
        if variant == "v0":
            cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(boxed, str(i), (x1, max(y1 - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            continue
        lx, ly = max(4, (x2 - x1) // 4), max(4, (y2 - y1) // 5)
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(boxed, (cx, cy), (cx + dx * lx, cy), (0, 255, 255), 1)
            cv2.line(boxed, (cx, cy), (cx, cy + dy * ly), (0, 255, 255), 1)
        cv2.putText(boxed, str(i), (x1, max(y1 - 4, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return boxed


def sheet_image(frames: list[np.ndarray], variant: str = "v0", label: str | None = None) -> np.ndarray:
    """2 x 2 sheet of the four frames at SHEET_WIDTH[variant] per tile, optionally labelled top-left."""
    h, w = frames[0].shape[:2]
    tw = SHEET_WIDTH[variant]
    tiles = [cv2.resize(f, (tw, round(tw * h / w))) for f in frames]
    sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    if label:
        cv2.putText(sheet, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(sheet, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return sheet


def feet_prompt(sizes: list[tuple[int, int]]) -> str:
    return (
        f"The {len(sizes)} attached images are crops, each centred on one tennis player "
        f"({', '.join(f'{w} x {h}' for w, h in sizes)} pixels). For each image in order, return pixel "
        "coordinates within that image (x from the left, y from the top) of the player's ankles, feet and "
        "head top. Left and right are the player's own. Ankle: centre of the joint. Big toe / small toe: "
        "tips of the toes of the shoe. Heel: back of the heel. Head top: the top of the skull (top of the "
        "hair or cap). Give a best estimate for hidden points. Look at the images directly; do not run any "
        "commands."
    )


def read_frames(media: str, frames: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(DATA / media))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(np.clip(f, 0, n - 1)))
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read {media} frame {f}")
        out.append(bgr)
    cap.release()
    return out


def court_homography(kp: np.ndarray) -> np.ndarray | None:
    ok = kp[:, 2] > 0
    if ok.sum() < 4:
        return None
    m, _ = cv2.findHomography(np.float32(kp[ok, :2]), np.float32(np.array(REF, float)[ok]), cv2.RANSAC, 10.0)
    return m


def players(people: list[dict], court: np.ndarray | None) -> list[int]:
    """Indexes of up to 4 people standing in the playing area, tallest first, else the two largest.

    The playing area is the court plus 8 m behind each baseline (pros stand
    well back) and 3.5 m beside each doubles sideline, minus the band within
    2 m of the net outside the sidelines, where the chair umpire and net
    judges sit.
    """
    heights = [p["box"][3] - p["box"][1] for p in people]
    m = court_homography(court) if court is not None else None
    if m is not None and people:
        feet = np.array([[(p["box"][0] + p["box"][2]) / 2, p["box"][3]] for p in people], np.float32)
        ref = cv2.perspectiveTransform(feet[:, None], m).reshape(-1, 2)
        left, right, far, near, net = 286, 1379, 561, 2935, 1748
        inside = []
        for i, (x, y) in enumerate(ref):
            in_area = left - 3.5 * METRE <= x <= right + 3.5 * METRE and far - 8 * METRE <= y <= near + 8 * METRE
            by_net = abs(y - net) < 2 * METRE and not left <= x <= right
            if in_area and not by_net:
                inside.append(i)
        if inside:
            return sorted(inside, key=lambda i: -heights[i])[:4]
    return list(np.argsort(heights)[::-1][:2])


class Local:
    """RF-DETR + ViTPose and WASB, loaded once."""

    def __init__(self):
        import torch

        from dsa.astra.ball import Wasb
        from dsa.data.paths import MODELS
        from dsa.pose.backends import load_models

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.pose = load_models("rfdetr", 1152, MODELS / "vitpose-plus-huge", device)
        self.wasb = Wasb(device)

    def people(self, bgr: np.ndarray) -> list[dict]:
        from dsa.pose.backends import run_vitpose

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        xyxy, conf = self.pose.detector.detect(rgb)
        xyxy = xyxy[conf >= 0.4]
        xy, sc = run_vitpose(*self.pose.vitpose, self.device, rgb, xyxy)
        return [{"box": b, "xy": k, "score": s} for b, k, s in zip(xyxy, xy, sc)]

    def ball(self, frames: list[np.ndarray]) -> tuple[bool, float, float]:
        vis, x, y, _ = self.wasb.locate(frames)
        return vis, x, y


def label_frame(row: dict, local: Local, work: Path) -> dict:
    fps = row["fps"]
    f = row["frame"]
    offsets = [-1, 0, 1] + [round(dt * fps) for dt in (-0.75, -0.25, 0.25, 0.75)]
    imgs = read_frames(row["media"], [f + o for o in offsets])
    bgr = imgs[1]
    h, w = bgr.shape[:2]
    stem = row["sample"].replace("/", "__")
    frame_path = work / f"{stem}.jpg"
    cv2.imwrite(str(frame_path), bgr)
    sheet = np.vstack([np.hstack([cv2.resize(i, (640, round(640 * h / w))) for i in imgs[3:5]]),
                       np.hstack([cv2.resize(i, (640, round(640 * h / w))) for i in imgs[5:7]])])
    sheet_path = work / f"{stem}_sheet.jpg"
    cv2.imwrite(str(sheet_path), sheet)
    people = local.people(bgr)
    boxed = bgr.copy()
    for i, person in enumerate(people):
        x1, y1, x2, y2 = (int(v) for v in person["box"])
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(boxed, str(i), (x1, max(y1 - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    boxes_path = work / f"{stem}_boxes.jpg"
    cv2.imwrite(str(boxes_path), boxed)
    return {"row": row, "bgr": bgr, "ball_frames": imgs[:3], "frame_path": frame_path, "sheet_path": sheet_path,
            "boxes_path": boxes_path, "people": people, "ball": local.ball(imgs[:3])}


def parse_scene(ans: dict, n_people: int) -> dict:
    court = None
    if ans["court_visible"]:
        court = np.array([[p["x"], p["y"], 2.0 if p["in_view"] else 0.0] for p in ans["points"]])
    named = [i for i in ans["players"] if 0 <= i < n_people]
    return {"view": ans["view"], "in_play": ans["in_play"], "court": court, "players": named}


def astra_scene(item: dict, diagram_path: Path, cache: Path, variant: str = "v0", **kw) -> dict:
    """One scene call. v0 wants frame_path, sheet_path, boxes_path; v1 wants sheet_path and boxes_path
    made with sheet_image(..., "v1") and draw_boxes(..., "v1"). `kw` goes to ask (model, effort)."""
    h, w = item["bgr"].shape[:2]
    if variant == "v0":
        imgs = [item["frame_path"], diagram_path, item["sheet_path"], item["boxes_path"]]
    else:
        imgs = [diagram_path, item["sheet_path"], item["boxes_path"]]
    ans, _ = ask(scene_prompt(w, h, variant), [Path(p).resolve() for p in imgs], SCENE_SCHEMA, cache, **kw)
    return parse_scene(ans, len(item["people"]))


def astra_scenes(items: list[dict], diagram_path: Path, cache: Path, **kw) -> list[dict]:
    """v2: the scene call for BATCH keyframes at once (fewer padded with the last one). Each item wants
    sheet_path (sheet_image(..., "v2", label="moment k")) and boxes_path (draw_boxes(..., "v2"))."""
    batch = list(items) + [items[-1]] * (BATCH - len(items))
    sizes = [(it["bgr"].shape[1], it["bgr"].shape[0]) for it in batch]
    imgs = [diagram_path] + [p for it in batch for p in (it["sheet_path"], it["boxes_path"])]
    ans, _ = ask(slim_scene_prompt(sizes), [Path(p).resolve() for p in imgs], BATCH_SCHEMA, cache, **kw)
    return [parse_scene(a, len(it["people"])) for a, it in zip(ans["moments"], items)]


def head_top_rule(box: np.ndarray, xy: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Head top without Astra: the box top above the face centre, unless the box top is well above the head
    (an arm raised in a serve), then face centre + 0.774 x (face-to-shoulder-midpoint distance) straight up.

    Face centre = mean of the ViTPose nose, eyes and ears with score >= 0.3 (all five if none). Fitted on
    half the clips_v1 keyframe players against Astra's head top (dsa.astra.scene_cost report)."""
    ok = score[:5] >= 0.3
    face = xy[:5][ok].mean(0) if ok.any() else xy[:5].mean(0)
    est = face[1] - 0.774 * np.linalg.norm(face - (xy[5] + xy[6]) / 2)
    return np.array([face[0], est if est - box[1] > 0.05 * (box[3] - box[1]) else box[1]])


def astra_feet(item: dict, idx: list[int], work: Path, cache: Path) -> dict[int, np.ndarray]:
    if not idx:
        return {}
    crops, maps = [], []
    for i in idx:
        path = work / f"{item['row']['sample'].replace('/', '__')}_p{i}.png"
        scale, origin, size = upscaled_crop(item["bgr"], np.asarray(item["people"][i]["box"]), path)
        crops.append((path.resolve(), size))
        maps.append((scale, origin))
    ans, _ = ask(feet_prompt([s for _, s in crops]), [p for p, _ in crops], FEET_SCHEMA, cache)
    out = {}
    for i, (scale, origin), pts in zip(idx, maps, ans["people"]):
        xy = {k: np.array([pts[k]["x"], pts[k]["y"]]) / scale + origin for k in FOOT_KEYS}
        body = item["people"][i]["xy"]
        # Side from the nearest ViTPose ankle: swap Astra's feet if its left ankle is nearer the right one.
        if np.linalg.norm(xy["l_ankle"] - body[16]) + np.linalg.norm(xy["r_ankle"] - body[15]) < \
                np.linalg.norm(xy["l_ankle"] - body[15]) + np.linalg.norm(xy["r_ankle"] - body[16]):
            for a, b in (("l_big_toe", "r_big_toe"), ("l_small_toe", "r_small_toe"), ("l_heel", "r_heel")):
                xy[a], xy[b] = xy[b], xy[a]
        out[i] = xy
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("frames", help="frames.parquet to pre-fill")
    p.add_argument("--out", required=True, help="labels source name, e.g. gold_v1_prefill")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--no-rackets", action="store_true", help="skip the Modal racket step")
    a = p.parse_args()

    frames = pd.read_parquet(a.frames)
    if a.limit:
        frames = frames.head(a.limit)
    work = SCRATCH / "prefill" / a.out
    work.mkdir(parents=True, exist_ok=True)
    cache = Path("output/astra_cache/prefill")
    diagram(work / "diagram.png")

    local = Local()
    items = []
    for k, row in enumerate(frames.to_dict("records")):
        items.append(label_frame(row, local, work))
        print(f"local {k + 1}/{len(frames)}", flush=True)

    with ThreadPoolExecutor(a.workers) as pool:
        scenes = list(pool.map(lambda it: astra_scene(it, work / "diagram.png", cache), items))
        chosen = [sc["players"] or players(it["people"], sc["court"]) for it, sc in zip(items, scenes)]
        feet = list(pool.map(lambda z: astra_feet(z[0], z[1], work, cache), zip(items, chosen)))

    rackets = [[] for _ in items]
    if not a.no_rackets:
        from dsa.cloud.racket_pose import app, predict

        with app.run():
            preds = predict.remote([cv2.imencode(".jpg", it["bgr"])[1].tobytes() for it in items],
                                   [[] for _ in items])
        rackets = [p["detections"] for p in preds]

    people_rows, racket_rows, ball_rows, court_rows, scene_rows = [], [], [], [], []
    for it, sc, ch, ft, rk in zip(items, scenes, chosen, feet, rackets):
        s = it["row"]["sample"]
        kps = {}
        for i, person in enumerate(it["people"]):
            kp = schema.empty_kp()
            vis = np.where(person["score"] >= 0.3, 2.0, 1.0)
            kp[:17] = np.column_stack([person["xy"], vis])
            kp[schema.KP["neck"]] = [*((person["xy"][5] + person["xy"][6]) / 2), min(vis[5], vis[6])]
            if i in ft:
                kp[schema.KP["head_top"]] = [*ft[i]["head_top"], 2.0]
                if FEET:
                    for name, key in (("left_big_toe", "l_big_toe"), ("left_small_toe", "l_small_toe"),
                                      ("left_heel", "l_heel"), ("right_big_toe", "r_big_toe"),
                                      ("right_small_toe", "r_small_toe"), ("right_heel", "r_heel")):
                        kp[schema.KP[name]] = [*ft[i][key], 2.0]
            kps[i] = kp
        # Each racket to the player whose wrist is nearest its handle.
        for r, det in enumerate(rk):
            rkp = np.column_stack([np.array(det["kp"]), np.where(np.array(det["kp_score"]) >= 0.3, 2.0, 1.0)])
            owner = np.nan
            best = np.inf
            for i in ch:
                person = it["people"][i]
                hgt = person["box"][3] - person["box"][1]
                d = min(np.linalg.norm(rkp[2, :2] - person["xy"][w]) for w in (9, 10))
                if d < 0.25 * hgt and d < best:
                    owner, best = i, d
            racket_rows.append({"sample": s, "racket": r, "box": schema.flat(det["box"]), "kp": schema.flat(rkp),
                                "person": owner, "labeler": RACKETS})
            if not np.isnan(owner):
                kps[int(owner)][schema.KP["racket_tip"]:] = rkp
        for i, person in enumerate(it["people"]):
            people_rows.append({"sample": s, "person": i, "track": None, "box": schema.flat(person["box"]),
                                "kp": schema.flat(kps[i]), "stroke": None, "player": i in ch,
                                "labeler": PEOPLE + (f"; {'feet, ' if FEET else ''}head top: {ASTRA}" if i in ft else "")
                                + ("; racket: " + RACKETS if not np.isnan(kps[i][25, 0]) else "")})
        vis, x, y = it["ball"]
        ball_rows.append({"sample": s, "x": x if vis else np.nan, "y": y if vis else np.nan, "visible": vis,
                          "labeler": WASB})
        if sc["court"] is not None:
            court_rows.append({"sample": s, "kp": schema.flat(sc["court"]), "labeler": ASTRA})
        scene_rows.append({"sample": s, "view": sc["view"], "in_play": sc["in_play"], "labeler": ASTRA})

    cols = {"people": schema.TABLES["people"] + ["player"]}
    tables = {"frames": frames, "people": pd.DataFrame(people_rows, columns=cols["people"]),
              "rackets": pd.DataFrame(racket_rows, columns=schema.TABLES["rackets"]),
              "ball": pd.DataFrame(ball_rows, columns=schema.TABLES["ball"]),
              "court": pd.DataFrame(court_rows, columns=schema.TABLES["court"]),
              "scene": pd.DataFrame(scene_rows, columns=schema.TABLES["scene"])}
    out = schema.write(a.out, tables)
    print(json.dumps({k: len(v) for k, v in tables.items()}), "->", out)


if __name__ == "__main__":
    main()
