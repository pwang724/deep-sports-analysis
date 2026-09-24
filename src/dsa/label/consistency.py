"""Mask clip labels that disagree with their neighbours in time, the court, or another labeler.

Reads a clip labels source (dsa.label.clips) and writes <source>_clean: the
same tables with every label that fails a check set to NaN, plus a `flags`
table (dsa.data.schema) saying what was masked and why. The raw source is
kept, so the video review can show what was dropped.

  ball     isolated     visible on one frame with none within BALL_GAP frames
           off track    far from the ball's path extrapolated from both sides
           static       still for BALL_STATIC_S: a ball lying on court or held
  pose     flip         a left / right joint pair matches its neighbours swapped
           spike        a joint jumps away and back between neighbouring frames
           head top     Astra's head top far from ViTPose's face (Astra vs specialist)
  court    unstable     a point moves between the shot's keyframes
           homography   a point off the homography fitted to the others
           camera moved the camera panned away from where the keyframes saw the court
  players  choice       a track Astra named on one keyframe and not on another
           outside      a named player standing outside the playing area
           track lost   fewer player tracks than Astra named (a track broke);
                        the frame's other people lose their player flag
  view     shot         keyframes of one shot disagree
           no court     a view with no court found
  in_play  no ball      in play but WASB saw no ball within IN_PLAY_BALL_S

    python -m dsa.label.consistency clips_v1            # -> data/labels/clips_v1_clean
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

from dsa.astra.court import REF
from dsa.data import schema
from dsa.data.paths import LABELS

BALL_GAP = 2                # frames; a real ball is seen for many frames in a row, a false one for one
BALL_RESID = 0.02           # x width (26 px at 1280): a smooth ball path predicts the next frame within this
BALL_STATIC_S = 1.0         # s; a ball in play never stays within BALL_STATIC_PX this long
BALL_STATIC_PX = 0.004      # x width (5 px at 1280)
JOINT_SPIKE = 0.15          # x box height; away and back by this within JOINT_GAP_S is noise, not motion
JOINT_GAP_S = 0.25          # s; neighbours further apart than this are not compared
FLIP_RATIO = 0.5            # swapped pair at most this x the unswapped distance to both neighbours
FLIP_MIN = 0.05             # x box height; ignore pairs that are nearly on top of each other
HEAD_TOP_MAX = 0.2          # x box height from the face (nose, eyes, ears) centre: a head is ~0.13 tall
COURT_PX = 0.006            # x width (8 px at 1280): Astra's court points agree to ~2 px median, worst ~17
COURT_REPROJ = 0.006        # x width; lines are planar, so a homography fits good points to a few px
PLAY_AREA_M = (3.5, 8.0)    # m beside / behind the court counted as the playing area (as dsa.label.prefill)
IN_PLAY_BALL_S = 1.0        # s; WASB finds the ball within a second in any rally

PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
HT = schema.KP["head_top"]


def kparr(values, n: int = len(schema.KEYPOINTS)) -> np.ndarray:
    """A stored flat list to a writable (n, 3) array (schema.keypoints may view a read-only Arrow buffer)."""
    return np.array(values, float).reshape(n, 3)


class Flags:
    def __init__(self):
        self.rows = []
        self.denom = defaultdict(int)

    def add(self, sample, head, reason, labeler, person=np.nan, point=np.nan):
        self.rows.append({"sample": sample, "head": head, "reason": reason, "labeler": labeler,
                          "person": float(person), "point": float(point)})

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=schema.TABLES["flags"])


def check_ball(ball: pd.DataFrame, fr: pd.DataFrame, flags: Flags) -> None:
    b = ball.join(fr[["clip", "shot", "frame", "width", "fps"]], on="sample")
    flags.denom["ball"] = int(b["visible"].fillna(False).sum())
    masked = set()
    for _, g in b.groupby(["clip", "shot"]):
        g = g.sort_values("frame")
        vis = dict(zip(g["frame"], g["visible"].fillna(False)))
        xy = dict(zip(g["frame"], zip(g["x"], g["y"])))
        w, fps = g["width"].iloc[0], g["fps"].iloc[0]
        pos = lambda f: np.array(xy[f]) if vis.get(f) else None
        for idx, f in zip(g.index, g["frame"]):
            if not vis[f]:
                continue
            p = pos(f)
            if not any(vis.get(f + d) for d in range(-BALL_GAP, BALL_GAP + 1) if d):
                reason = "isolated"
            else:
                preds = []
                for s in (-1, 1):
                    p1, p2, p3 = pos(f + s), pos(f + 2 * s), pos(f + 3 * s)
                    if p1 is not None and p2 is not None:
                        preds.append(3 * p1 - 3 * p2 + p3 if p3 is not None else 2 * p1 - p2)
                reason = "off track" if preds and min(np.linalg.norm(p - q) for q in preds) > BALL_RESID * w else None
                n = round(BALL_STATIC_S * fps / 2)
                near = [pos(f + d) for d in range(-n, n + 1)]
                if reason is None and all(q is not None for q in near):
                    near = np.array(near)
                    if np.abs(near - np.median(near, 0)).max() < BALL_STATIC_PX * w:
                        reason = "static"
            if reason:
                masked.add(idx)
                flags.add(b.at[idx, "sample"], "ball", reason, b.at[idx, "labeler"])
    ball.loc[list(masked), ["x", "y"]] = np.nan
    ball.loc[list(masked), "visible"] = pd.NA


def check_pose(people: pd.DataFrame, fr: pd.DataFrame, flags: Flags) -> None:
    p = people.join(fr[["frame", "fps"]], on="sample")
    kps = {i: kparr(v) for i, v in people["kp"].items()}
    flags.denom["pose"] = int(sum(np.isfinite(k[:17, 0]).sum() for k in kps.values()))
    flags.denom["head top"] = int(sum(np.isfinite(k[HT, 0]) for k in kps.values()))
    for _, g in p.groupby("track"):
        g = g.sort_values("frame")
        idx, frames, fps = list(g.index), g["frame"].to_numpy(), g["fps"].iloc[0]
        for j in range(1, len(idx) - 1):
            if (frames[j + 1] - frames[j - 1]) / fps > 2 * JOINT_GAP_S:
                continue
            k, a, c = kps[idx[j]], kps[idx[j - 1]], kps[idx[j + 1]]
            box = people.at[idx[j], "box"]
            h = box[3] - box[1]
            s = people.at[idx[j], "sample"]
            lab = people.at[idx[j], "labeler"].split(";")[0]
            d = lambda u, v: np.linalg.norm(u[:2] - v[:2])
            for l, r in PAIRS:
                same = [d(k[l], n[l]) + d(k[r], n[r]) for n in (a, c)]
                swap = [d(k[l], n[r]) + d(k[r], n[l]) for n in (a, c)]
                if all(np.isfinite(same)) and all(sw < FLIP_RATIO * sa and sa > FLIP_MIN * h
                                                  for sw, sa in zip(swap, same)):
                    for q in (l, r):
                        k[q] = np.nan
                        flags.add(s, "pose", "flip", lab, people.at[idx[j], "person"], q)
            for q in range(17):
                if np.isfinite(k[q, 0]) and d(k[q], a[q]) > JOINT_SPIKE * h and d(k[q], c[q]) > JOINT_SPIKE * h \
                        and d(a[q], c[q]) < JOINT_SPIKE * h / 2:
                    k[q] = np.nan
                    flags.add(s, "pose", "spike", lab, people.at[idx[j], "person"], q)
    for i, k in kps.items():
        if not (np.isfinite(k[5, 0]) and np.isfinite(k[6, 0])):
            k[schema.KP["neck"]] = np.nan
        if np.isfinite(k[HT, 0]):
            box = people.at[i, "box"]
            h = box[3] - box[1]
            face = k[:5][np.isfinite(k[:5, 0])]
            if len(face) and (np.linalg.norm(k[HT, :2] - face[:, :2].mean(0)) > HEAD_TOP_MAX * h
                              or (np.isfinite(k[0, 1]) and k[HT, 1] > k[0, 1])):
                k[HT] = np.nan
                flags.add(people.at[i, "sample"], "pose", "head top", people.at[i, "labeler"].split("head top: ")[-1],
                          people.at[i, "person"], HT)
    people["kp"] = [schema.flat(kps[i]) for i in people.index]


def homography(kp: np.ndarray, to_ref: bool) -> np.ndarray | None:
    ok = np.isfinite(kp[:, 0]) & (kp[:, 2] > 0)
    if ok.sum() < 4:
        return None
    src, dst = np.float32(REF)[ok], np.float32(kp[ok, :2])
    return cv2.findHomography(dst, src, 0)[0] if to_ref else cv2.findHomography(src, dst, 0)[0]


def check_court(court: pd.DataFrame, fr: pd.DataFrame, flags: Flags) -> None:
    c = court.join(fr[["clip", "shot", "frame", "width", "keyframe", "cam_dx", "cam_dy"]], on="sample")
    kps = {i: kparr(v, 14) for i, v in court["kp"].items()}
    flags.denom["court"] = int(sum((np.isfinite(k[:, 0]) & (k[:, 2] > 0)).sum() for k in kps.values()))
    for _, g in c.groupby(["clip", "shot"]):
        w, lab = g["width"].iloc[0], g["labeler"].iloc[0]
        med = kps[g.index[0]].copy()           # every row of a shot carries the shot's court
        kf = g[g["keyframe"] & g["kf_kp"].notna()]
        bad = {}
        for q in range(14):
            if med[q, 2] <= 0 or not np.isfinite(med[q, 0]):
                continue
            dev = [np.linalg.norm(kparr(v, 14)[q, :2] - med[q, :2]) for v in kf["kf_kp"]
                   if kparr(v, 14)[q, 2] > 0]
            if dev and max(dev) > COURT_PX * w:
                bad[q] = "unstable"
        good = med.copy()
        good[list(bad), 2] = 0
        m = homography(good, False)
        if m is None:
            bad.update({q: "homography" for q in range(14) if med[q, 2] > 0 and q not in bad})
        else:
            proj = cv2.perspectiveTransform(np.float32(REF)[:, None], m).reshape(14, 2)
            for q in range(14):
                if q not in bad and med[q, 2] > 0 and np.linalg.norm(proj[q] - med[q, :2]) > COURT_REPROJ * w:
                    bad[q] = "homography"
        ref = np.array([kf["cam_dx"].mean(), kf["cam_dy"].mean()]) if len(kf) else None
        for i, r in g.iterrows():
            k = kps[i]
            moved = ref is not None and np.linalg.norm(np.array([r.cam_dx, r.cam_dy]) - ref) > COURT_PX * w
            for q in range(14):
                if k[q, 2] > 0 and np.isfinite(k[q, 0]) and (moved or q in bad):
                    k[q] = np.nan
                    flags.add(r["sample"], "court", "camera moved" if moved and q not in bad else bad[q], lab, point=q)
    court["kp"] = [schema.flat(kps[i]) for i in court.index]


def in_area(foot: np.ndarray, court: np.ndarray) -> bool:
    m = homography(court, True)
    if m is None:
        return True
    x, y = cv2.perspectiveTransform(np.float32(foot)[None, None], m).reshape(2)
    metre = 1117 / 10.97
    left, right, far, near = 286, 1379, 561, 2935
    side, back = PLAY_AREA_M
    return left - side * metre <= x <= right + side * metre and far - back * metre <= y <= near + back * metre


def check_players(people: pd.DataFrame, court: pd.DataFrame, fr: pd.DataFrame, flags: Flags) -> None:
    p = people.join(fr[["clip", "shot", "frame", "keyframe"]], on="sample")
    flags.denom["players"] = int(p["player"].fillna(False).sum())
    courts = {s: kparr(v, 14) for s, v in zip(court["sample"], court["kp"])}
    mask = set()
    for _, g in p.groupby(["clip", "shot"]):
        kf = g[g["keyframe"] & g["named"].notna()]
        for t, tg in kf.groupby("track"):
            if tg["named"].nunique() > 1:
                mask.update((i, "choice") for i in g.index[g["track"] == t])
        for i, r in kf[kf["named"].fillna(False).astype(bool)].iterrows():
            if r["sample"] in courts and not in_area(np.array([(r.box[0] + r.box[2]) / 2, r.box[3]]), courts[r["sample"]]):
                mask.update((j, "outside") for j in g.index[g["track"] == r["track"]])
        expected = kf.groupby("sample")["named"].sum().max() if len(kf) else 0
        for s, fg in g.groupby("sample"):
            if fg["player"].fillna(False).sum() < expected and not fg["keyframe"].iloc[0]:
                mask.update((j, "track lost") for j in fg.index[~fg["player"].fillna(False).astype(bool)])
                if fg["player"].fillna(False).all():
                    flags.add(s, "players", "track lost", people.at[fg.index[0], "labeler"].split(";")[0])
    seen = set()
    for i, reason in sorted(mask, key=lambda z: z[0]):
        if i in seen:
            continue
        seen.add(i)
        flags.add(people.at[i, "sample"], "players", reason, "astra", people.at[i, "person"])
        people.at[i, "player"] = pd.NA


def check_scene(scene: pd.DataFrame, ball: pd.DataFrame, court: pd.DataFrame, fr: pd.DataFrame, flags: Flags) -> None:
    s = scene.join(fr[["clip", "shot", "frame", "keyframe", "fps", "scope"]], on="sample")
    flags.denom["view"] = int(s["view"].notna().sum())
    flags.denom["in_play"] = int(s["in_play"].notna().sum())
    has_court = set(court["sample"])
    for _, g in s.groupby(["clip", "shot"]):
        views = g.loc[g["keyframe"], "view"].dropna()
        reason = "shot" if views.nunique() > 1 else \
            "no court" if (g["scope"] == "view").any() and not g["sample"].isin(has_court).any() else None
        if reason:
            for i in g.index[g["view"].notna()]:
                flags.add(scene.at[i, "sample"], "view", reason, scene.at[i, "labeler"])
                scene.at[i, "view"] = pd.NA
    b = ball.join(fr[["clip", "frame"]], on="sample")
    seen = {c: np.sort(g.loc[g["visible"].fillna(False).astype(bool), "frame"].to_numpy()) for c, g in b.groupby("clip")}
    labelled = {c: set(g["frame"]) for c, g in b.groupby("clip")}
    for i, r in s[s["in_play"].fillna(False).astype(bool)].iterrows():
        if r["frame"] not in labelled.get(r["clip"], ()):
            continue                       # no ball labels here (not a view): nothing to compare
        f, n = seen.get(r["clip"], np.array([])), IN_PLAY_BALL_S * r["fps"]
        if not ((f >= r["frame"] - n) & (f <= r["frame"] + n)).any():
            flags.add(r["sample"], "in_play", "no ball", r["labeler"])
            scene.at[i, "in_play"] = pd.NA


CHECKS = [("ball", "isolated"), ("ball", "off track"), ("ball", "static"), ("pose", "flip"), ("pose", "spike"),
          ("head top", "head top"), ("court", "unstable"), ("court", "homography"), ("court", "camera moved"),
          ("players", "choice"), ("players", "outside"), ("players", "track lost"), ("view", "shot"),
          ("view", "no court"), ("in_play", "no ball")]


def report(flags: Flags) -> pd.DataFrame:
    t = flags.table()
    t["key"] = np.where(t["reason"] == "head top", "head top", t["head"])
    n = t.groupby(["key", "reason"]).size()
    out = pd.DataFrame([{"key": k, "reason": r, "masked": int(n.get((k, r), 0))} for k, r in CHECKS])
    out["of"] = out["key"].map(flags.denom)
    out["pct"] = (100 * out["masked"] / out["of"].clip(lower=1)).round(2)
    return out.rename(columns={"key": "head"})


def run(source: str, out: str) -> pd.DataFrame:
    fr = schema.read(source, "frames").set_index("sample", drop=False)
    fr.index.name = None
    t = {name: schema.read(source, name) for name in ("people", "rackets", "ball", "court", "scene")
         if (LABELS / source / f"{name}.parquet").exists()}
    raw_court = t["court"].copy()
    t["ball"]["visible"] = t["ball"]["visible"].astype("boolean")
    t["people"]["player"] = t["people"]["player"].astype("boolean")
    t["people"] = t["people"].drop(columns=["keyframe"], errors="ignore")
    if "named" not in t["people"]:
        t["people"]["named"] = pd.Series(pd.NA, index=t["people"].index, dtype="boolean")
    flags = Flags()
    check_ball(t["ball"], fr, flags)
    check_pose(t["people"], fr, flags)
    check_court(t["court"], fr, flags)
    check_players(t["people"], t["court"], fr, flags)       # the filtered court: a bad court misplaces players
    check_scene(t["scene"], t["ball"], raw_court, fr, flags)
    t["flags"] = flags.table()
    schema.write(out, {"frames": fr.reset_index(drop=True), **t})
    return report(flags)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="clip labels source from dsa.label.clips")
    p.add_argument("--out", help="default <source>_clean")
    a = p.parse_args()
    rep = run(a.source, a.out or f"{a.source}_clean")
    print(rep.to_string(index=False))
    print(f"-> {LABELS / (a.out or a.source + '_clean')}")


if __name__ == "__main__":
    main()
