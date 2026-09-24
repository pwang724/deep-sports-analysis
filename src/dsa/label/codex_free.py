"""Scene labels without Codex, scored against Astra's keyframe answers and public labels.

On every keyframe of a clips run (dsa.label.clips; images, people and the
WASB track from its scratch caches, Astra's answers replayed from the
prefill cache, never asked):

  court    TennisCourtDetector (dsa.astra.court.Tcd) and Astra, each raw and
           snapped to the painted lines (dsa.label.court_refine).
  view     a court was found and holds up (view_rule).
  players  people in the playing area of that court, one per half
           (players_rule).
  in play  the WASB track around the keyframe (in_play_rule).

Players are also scored on TennisSegmentation (both players labelled),
with RF-DETR candidates and the TCD (+ snap) court.

    python -m dsa.label.codex_free clips_v1            # -> output/codex_free/clips_v1/
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dsa.astra.codex import NotCached
from dsa.astra.court import REF
from dsa.data.paths import SCRATCH
from dsa.label.court_refine import line_mask, refine
from dsa.label.prefill import METRE, astra_scene

CACHE = Path("output/astra_cache/prefill")
OUT = Path("output/codex_free")
REF_A = np.array(REF, float)
LEFT, RIGHT, FAR, NEAR, NET, MID = 286, 1379, 561, 2935, 1748, 832


def _pkl(path: Path):
    return pickle.loads(path.read_bytes()) if path.exists() else None


def astra_answer(stem: Path, bgr: np.ndarray, people: list[dict], diagram: Path) -> dict | None:
    """Astra's keyframe scene answer from the cache (v0, else v1), or None if it was never asked."""
    item = {"bgr": bgr, "people": people, "frame_path": stem.with_suffix(".jpg"),
            "sheet_path": Path(f"{stem}_sheet.jpg"), "boxes_path": Path(f"{stem}_boxes.jpg")}
    try:
        return astra_scene(item, diagram, CACHE, "v0", cache_only=True)
    except (NotCached, FileNotFoundError):
        pass
    try:
        return astra_scene({**item, "sheet_path": Path(f"{stem}_sheet_v1.jpg"),
                            "boxes_path": Path(f"{stem}_boxes_v1.jpg")}, diagram, CACHE, "v1", cache_only=True)
    except (NotCached, FileNotFoundError):
        return None


def tcd_points(tcd, bgr: np.ndarray) -> np.ndarray:
    """TCD's 14 points on a frame of any size (it assumes 1280 x 720 input)."""
    h, w = bgr.shape[:2]
    if (w, h) == (1280, 720):
        return tcd.locate(bgr)
    return tcd.locate(cv2.resize(bgr, (1280, 720), interpolation=cv2.INTER_AREA)) * np.array([w / 1280, h / 720])


def keyframes(source: str, tcd) -> list[dict]:
    """Every keyframe with an image, people and an Astra answer (cache only), plus TCD's court."""
    work = SCRATCH / "clips" / source
    clips = json.loads((work / "clips.json").read_text())
    out = []
    for c in clips:
        scan = _pkl(work / "scan" / f"{c['clip']}.pkl")
        kpeople = _pkl(work / "keyframes" / f"{c['clip']}.pkl")
        if scan is None or kpeople is None:
            continue
        modal = _pkl(work / "modal" / f"{c['clip']}.pkl")
        for k, people in zip(scan["keys"], kpeople):
            stem = work / f"{c['clip']}__{k}"
            bgr = cv2.imread(str(stem.with_suffix(".jpg")))
            if bgr is None:
                continue
            ans = astra_answer(stem, bgr, people, work / "diagram.png")
            if ans is None:
                continue
            row = {"clip": c["clip"], "k": k, "fps": c["fps"], "width": bgr.shape[1], "height": bgr.shape[0],
                   "camera": c.get("camera"), "collection": c.get("collection"), "people": people,
                   "shot": next(i for i, (a, b) in enumerate(scan["shots"]) if a <= k < b),
                   "shots": scan["shots"], "ball": {f: v for f, v in scan["ball"].items()
                                                    if abs(f - k) <= 3 * c["fps"]},
                   "image": str(stem.with_suffix(".jpg")), "astra": ans, "tcd": tcd_points(tcd, bgr)}
            if modal is not None:
                row["track_people"] = {f: p for f, p in zip(modal["todo"], modal["people"])
                                       if abs(f - k) <= 1.5 * c["fps"]}
            out.append(row)
        print(f"{c['clip']}: {len(out)} keyframes so far", flush=True)
    return out


def snap(kfs: list[dict]) -> None:
    """TCD's and Astra's court snapped to the lines (None where the snap fails), in place."""
    for kf in kfs:
        bgr = cv2.imread(kf["image"])
        mask = line_mask(bgr)
        kf["tcd_ref"] = refine(bgr, kf["tcd"], mask=mask) if np.isfinite(kf["tcd"]).all(1).sum() >= 4 else None
        c = kf["astra"]["court"]
        kf["astra_ref"] = refine(bgr, c, mask=mask) if c is not None else None


# ---------------------------------------------------------------- rules

def court_ok(pts: np.ndarray | None, size: tuple[int, int]) -> bool:
    """A plausible whole-court view: far baseline in frame, near baseline below it and wider, court area a
    reasonable share of the frame, left / right not flipped."""
    if pts is None or not np.isfinite(pts).all():
        return False
    w, h = size
    fl, fr, nl, nr = pts[0], pts[1], pts[2], pts[3]
    far_in = all(0 <= p[0] < w and 0 <= p[1] < h for p in (fl, fr))
    order = fl[0] < fr[0] and nl[0] < nr[0] and max(fl[1], fr[1]) < min(nl[1], nr[1])
    wider = np.linalg.norm(nr - nl) >= 0.9 * np.linalg.norm(fr - fl)
    quad = np.float32([fl, fr, nr, nl])
    area = cv2.contourArea(quad) / (w * h)
    return bool(far_in and order and wider and 0.05 <= area <= 1.5)


def view_rule(kf: dict) -> bool:
    """View without Codex: TCD's court snaps to the painted lines and is a plausible whole court."""
    r = kf["tcd_ref"]
    return r is not None and court_ok(r["points"], (kf["width"], kf["height"]))


def homography(pts: np.ndarray) -> np.ndarray | None:
    ok = np.isfinite(pts).all(1)
    if ok.sum() < 4:
        return None
    m, _ = cv2.findHomography(np.float32(pts[ok, :2]), np.float32(REF_A[ok]), 0)
    return m


def to_court(people: list[dict], pts: np.ndarray) -> np.ndarray | None:
    """Foot point (box bottom centre) of each person on the reference court."""
    m = homography(pts)
    if m is None or not people:
        return None
    feet = np.float32([[(p["box"][0] + p["box"][2]) / 2, p["box"][3]] for p in people])
    return cv2.perspectiveTransform(feet[:, None], m).reshape(-1, 2)


def motion(kf: dict) -> np.ndarray:
    """Per keyframe person: median over the frames around it of the nearest box's centre shift, in box
    heights per second (NaN when no tracked frames); static people (umpire, line judges) stay near 0."""
    people, around = kf["people"], kf.get("track_people") or {}
    out = np.full(len(people), np.nan)
    if not around:
        return out
    for i, p in enumerate(people):
        b = np.asarray(p["box"], float)
        c, hgt = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]), b[3] - b[1]
        speeds = []
        for side in (1, -1):                     # chain the nearest box outward from the keyframe, both ways
            f0, c0 = kf["k"], c
            for f in sorted((f for f in around if (f - kf["k"]) * side > 0), key=lambda f: abs(f - kf["k"])):
                boxes = np.array([q["box"] for q in around[f]], float) if around[f] else np.zeros((0, 4))
                if not len(boxes):
                    continue
                cs = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2]
                d = np.linalg.norm(cs - c0, axis=1)
                j = int(np.argmin(d))
                if d[j] > 0.5 * hgt + 1.5 * hgt * abs(f - f0) / kf["fps"]:
                    break
                speeds.append(d[j] / hgt / (abs(f - f0) / kf["fps"]))
                f0, c0 = f, cs[j]
        if speeds:
            out[i] = float(np.median(speeds))
    return out


def players_rule(people: list[dict], pts: np.ndarray | None, speed: np.ndarray | None = None,
                 pick: str = "centre") -> list[int]:
    """Players without Codex: people whose feet are in the playing area (court + 8 m behind each baseline,
    3.5 m beside, minus the umpire band within 2 m of the net outside the doubles sidelines), at most one
    per half. `pick` chooses within a half: centre (nearest the centre line), tall, or move (fastest;
    people slower than 0.05 heights / s count as static and are dropped when anyone else is in the half)."""
    if pts is None:
        return []
    ref = to_court(people, pts)
    if ref is None:
        return []
    heights = np.array([p["box"][3] - p["box"][1] for p in people])
    chosen = []
    for half in (ref[:, 1] < NET, ref[:, 1] >= NET):
        x, y = ref[:, 0], ref[:, 1]
        area = (x >= LEFT - 3.5 * METRE) & (x <= RIGHT + 3.5 * METRE) & (y >= FAR - 8 * METRE) & (y <= NEAR + 8 * METRE)
        by_net = (np.abs(y - NET) < 2 * METRE) & ~((x >= LEFT) & (x <= RIGHT))
        cand = np.flatnonzero(half & area & ~by_net)
        if not len(cand):
            continue
        if pick == "move" and speed is not None and np.isfinite(speed[cand]).any():
            moving = cand[np.nan_to_num(speed[cand], nan=0) >= 0.05]
            cand = moving if len(moving) else cand
        if pick == "tall":
            j = cand[np.argmax(heights[cand])]
        else:
            # nearest the court's centre line, beyond the court counted in metres out from its edge
            out_x = np.maximum(0, np.abs(x[cand] - MID) - (RIGHT - LEFT) / 2)
            out_y = np.maximum(0, np.maximum(FAR - y[cand], y[cand] - NEAR))
            j = cand[np.argmin(np.abs(x[cand] - MID) + 2 * out_x + 2 * out_y)]
        chosen.append(int(j))
    return sorted(chosen)


def in_play_rule(kf: dict, window_s: float = 1.0, min_frames: int = 15, min_speed: float = 0.1) -> bool:
    """In play without Codex: within +-window_s of the keyframe (same shot), WASB sees the ball on at least
    min_frames frames moving at >= min_speed frame widths per second (median over consecutive visible
    frames); a ball bounced before a serve or lying still moves far slower. Defaults: the best of a small
    grid against Astra's keyframe answers on half the clips_v1 clips (82% on the other half)."""
    k, fps, w = kf["k"], kf["fps"], kf["width"]
    s0, s1 = kf["shots"][kf["shot"]]
    fs = [f for f in sorted(kf["ball"]) if abs(f - k) <= window_s * fps and s0 <= f < s1 and kf["ball"][f][0]]
    if len(fs) < min_frames:
        return False
    xy = np.array([kf["ball"][f][1:] for f in fs])
    dt = np.diff(fs) / fps
    sp = np.linalg.norm(np.diff(xy, axis=0), axis=1) / w / dt
    sp = sp[np.diff(fs) <= 3]
    return bool(len(sp) >= min_frames - 1 and np.median(sp) >= min_speed)


# ---------------------------------------------------------------- scoring

def _court(kf: dict, name: str) -> np.ndarray | None:
    if name == "astra":
        c = kf["astra"]["court"]
        return None if c is None else c[:, :2]
    if name == "tcd":
        return kf["tcd"] if np.isfinite(kf["tcd"]).all() else None
    return None if kf[name] is None else kf[name]["points"]


def codex_free_court(kf: dict) -> np.ndarray | None:
    """TCD snapped to the lines, else TCD as it is."""
    return _court(kf, "tcd_ref") if kf["tcd_ref"] is not None else _court(kf, "tcd")


def score_court(kfs: list[dict]) -> dict:
    """Per-point agreement (px at 1280 wide) between courts on Astra's view keyframes, points Astra marks in view."""
    pairs = [("astra", "tcd"), ("astra", "astra_ref"), ("astra_ref", "tcd_ref"), ("astra", "tcd_ref"), ("tcd", "tcd_ref")]
    view = [kf for kf in kfs if kf["astra"]["view"] and kf["astra"]["court"] is not None]
    out = {"keyframes": len(view), **{f"{n}_found": float(np.mean([_court(kf, n) is not None for kf in view]))
                                      for n in ("tcd", "tcd_ref", "astra_ref")}}
    for a, b in pairs:
        d, frames = [], 0
        for kf in view:
            pa, pb = _court(kf, a), _court(kf, b)
            if pa is None or pb is None:
                continue
            frames += 1
            d.append(np.linalg.norm(pa - pb, axis=1)[kf["astra"]["court"][:, 2] > 0] * 1280 / kf["width"])
        d = np.concatenate(d) if d else np.zeros(0)
        out[f"{a} vs {b}"] = {"frames": frames, "points": len(d), "median": float(np.median(d)) if len(d) else None,
                              "within3": float((d <= 3).mean()) if len(d) else None,
                              "within7": float((d <= 7).mean()) if len(d) else None,
                              "within15": float((d <= 15).mean()) if len(d) else None}
    return out


def score_clips(kfs: list[dict]) -> dict:
    view = pd.DataFrame({"astra": [kf["astra"]["view"] for kf in kfs], "rule": [view_rule(kf) for kf in kfs]})
    res = {"court": score_court(kfs),
           "view": {"n": len(view), "agree": float((view.astra == view.rule).mean()),
                    "confusion (astra, rule)": {f"{a},{r}": int(n) for (a, r), n in
                                                view.value_counts(["astra", "rule"]).items()}}}
    rows = []
    for kf in kfs:
        a = kf["astra"]
        if not a["view"] or len(a["players"]) > 2:
            continue
        sp = motion(kf)
        astra_court = _court(kf, "astra_ref") if kf["astra_ref"] is not None else _court(kf, "astra")
        r = {"astra": sorted(a["players"]), "tcd_ref": kf["tcd_ref"] is not None}
        for src, court in (("codex-free court", codex_free_court(kf)), ("Astra court", astra_court)):
            for pick in ("centre", "tall", "move"):
                r[f"{src}, {pick}"] = players_rule(kf["people"], court, sp, pick) == r["astra"]
        rows.append(r)
    pl = pd.DataFrame(rows)
    res["players vs Astra (exact set)"] = {"n": len(pl), "n with TCD snap": int(pl.tcd_ref.sum()), **{
        c: {"all": float(pl[c].mean()), "TCD snap frames": float(pl[c][pl.tcd_ref].mean())}
        for c in pl if "," in c}}
    view_kf = [kf for kf in kfs if kf["astra"]["view"]]
    y = np.array([kf["astra"]["in_play"] for kf in view_kf])
    p = np.array([in_play_rule(kf) for kf in view_kf])
    held = np.array([kf["clip"] not in set(sorted({k["clip"] for k in view_kf})[::2]) for kf in view_kf])
    res["in play vs Astra"] = {"n": len(y), "astra in play": float(y.mean()), "agree": float((y == p).mean()),
                               "agree, held-out half": float((y == p)[held].mean()),
                               "astra yes, rule no": int((y & ~p).sum()), "astra no, rule yes": int((~y & p).sum())}
    return res


def score_segmentation(tcd) -> dict:
    """Players on TennisSegmentation (both players labelled): RF-DETR candidates, TCD (+ snap) court."""
    import torch

    from dsa.data.paths import MODELS, SOURCES
    from dsa.data.tennis_segmentation import load
    from dsa.pose.backends import load_models

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    det = load_models("rfdetr", 1152, MODELS / "vitpose-plus-huge", dev).detector
    hits = {"TCD": [], "TCD + snap": [], "TCD + snap, else TCD": []}
    found = {"TCD complete": 0, "snap ok": 0, "view rule": 0}
    for s in load(SOURCES / "tennis_segmentation"):
        if len(s.boxes) < 2:
            continue
        bgr = cv2.cvtColor(s.image, cv2.COLOR_RGB2BGR)
        xyxy, conf = det.detect(s.image)
        people = [{"box": b} for b in xyxy[conf >= 0.4]]
        raw = tcd_points(tcd, bgr)
        r = refine(bgr, raw) if np.isfinite(raw).all(1).sum() >= 4 else None
        raw = raw if np.isfinite(raw).all() else None
        ref = None if r is None else r["points"]
        found["TCD complete"] += raw is not None
        found["snap ok"] += ref is not None
        found["view rule"] += ref is not None and court_ok(ref, bgr.shape[1::-1])
        gt = set()
        for g in s.boxes.values():
            ious = [_iou(p["box"], g) for p in people]
            if ious and max(ious) > 0.5:
                gt.add(int(np.argmax(ious)))
        for k, c in (("TCD", raw), ("TCD + snap", ref), ("TCD + snap, else TCD", ref if ref is not None else raw)):
            hits[k].append(set(players_rule(people, c)) == gt)
    return {"n": len(hits["TCD"]), **found, **{f"players exact, {k}": float(np.mean(v)) for k, v in hits.items()}}


def _iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main() -> None:
    import torch

    from dsa.astra.court import Tcd

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="clips labels source, e.g. clips_v1")
    p.add_argument("--rescan", action="store_true", help="redo TCD on the keyframes (cached otherwise)")
    a = p.parse_args()
    out = OUT / a.source
    out.mkdir(parents=True, exist_ok=True)
    tcd = Tcd("mps" if torch.backends.mps.is_available() else "cpu")
    cached = out / "keyframes.pkl"
    if cached.exists() and not a.rescan:
        kfs = pickle.loads(cached.read_bytes())
    else:
        kfs = keyframes(a.source, tcd)
        cached.write_bytes(pickle.dumps(kfs))
    snap(kfs)
    res = {"keyframes": len(kfs), "clips": len({kf["clip"] for kf in kfs}), **score_clips(kfs),
           "tennis_segmentation": score_segmentation(tcd)}
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
