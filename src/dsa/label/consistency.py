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
  court    unstable     a point's keyframe answers disagree (fewer than half, or
                        than two, within COURT_PX of the median of the others;
                        camera pan undone) and none is near the shot's planar
                        fit either (see consensus); agreeing answers give the
                        point (outliers dropped), else the fit
           homography   a point off the homography fitted to the others
           camera moved the camera panned more than CAM_FOLLOW from where the keyframes
                        saw the court, or the camera track jumped (CAM_JUMP in one
                        frame) between this frame and a keyframe; otherwise the
                        court follows the pan
  players  choice       a track Astra named on some keyframes and not on others, by a
                        close vote (dsa.label.clips takes the majority)
           outside      a named player standing outside the playing area
           track lost   a player is missing on a frame and someone else in the
                        playing area stands where the player was within
                        REACQ_S: that person might be the player, so their
                        player flag goes (dsa.label.clips.link_players re-acquires
                        the unambiguous cases; bystanders elsewhere keep theirs)
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
COURT_REPROJ = 0.012        # x width; lines are planar, so a homography fits good points to a few px
COURT_FIT_MIN = 6           # distinct court points among the inliers for the shot's planar fit to be trusted
COURT_FAR = 2.0             # x COURT_PX: with a trusted fit, a point is unstable only if no answer comes this close
CAM_FOLLOW = 0.05           # x width: the court follows the camera pan (a shift) up to this far from the keyframes
CAM_JUMP = 0.025            # x width in one frame (32 px at 1280): pans are smooth; a step this big is a missed cut or a
                            # whip pan, so frames across it from a keyframe are not trusted
CHOICE_MARGIN = 1           # keyframe votes: a track named on k of n is a close call when |k - (n - k)| <= this
FOOT_TOL = 0.15             # x box height the foot may be lower than the box bottom (see box_in_area)
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


def consensus(answers: list[np.ndarray], shift: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """The shot's court from its keyframe answers (k, 14, 3), each moved by -`shift` (k, 2) to undo camera pan.

    Per point: answers within `tol` of the median of the point's other answers agree; when at least half
    (and two, if there are two or more) agree, the point is their median (Astra's own answer, right even
    under a wide-angle lens a homography cannot fit). Astra scatters 10-20 px on far lines behind the net;
    for such a point the court's planar fit (one RANSAC homography over every answer, trusted with
    COURT_FIT_MIN distinct inlier points; ~1 px from the agreeing points) places it, when one of its
    answers comes within COURT_FAR x `tol` of the fit. Otherwise unstable.
    Returns (14, 2) points (NaN: no answer) and (14,) stable."""
    xy = np.full((14, 2), np.nan)
    ok = np.zeros(14, bool)
    obs = [(q, a[q, :2] - d) for a, d in zip(answers, shift) for q in range(14) if a[q, 2] > 0 and np.isfinite(a[q, 0])]
    if not obs:
        return xy, ok
    qs, pts = np.array([q for q, _ in obs]), np.array([v for _, v in obs], float)
    fit = None
    if len(np.unique(qs)) >= 4:
        m, inl = cv2.findHomography(np.float32(REF)[qs], np.float32(pts), cv2.RANSAC, tol)
        if m is not None and len(np.unique(qs[inl.ravel().astype(bool)])) >= COURT_FIT_MIN:
            fit = cv2.perspectiveTransform(np.float32(REF)[:, None], m).reshape(14, 2)
    for q in np.unique(qs):
        v = pts[qs == q]
        near = np.array([np.linalg.norm(v[j] - np.median(np.delete(v, j, 0), 0)) <= tol for j in range(len(v))]) \
            if len(v) > 1 else np.ones(1, bool)
        xy[q] = np.median(v[near], 0) if near.any() else np.median(v, 0)
        ok[q] = near.sum() >= max(np.ceil(len(v) / 2), min(len(v), 2))
        if not ok[q] and fit is not None and (np.linalg.norm(v - fit[q], axis=1) <= COURT_FAR * tol).any():
            xy[q], ok[q] = fit[q], True
    return xy, ok


def check_court(court: pd.DataFrame, fr: pd.DataFrame, flags: Flags) -> None:
    c = court.join(fr[["clip", "shot", "frame", "width", "keyframe", "cam_dx", "cam_dy", "camera"]], on="sample")
    kps = {i: kparr(v, 14) for i, v in court["kp"].items()}
    flags.denom["court"] = int(sum((np.isfinite(k[:, 0]) & (k[:, 2] > 0)).sum() for k in kps.values()))
    for cam, g in c.groupby("camera", dropna=False):
        flags.denom[("court", cam)] = int(sum((np.isfinite(kps[i][:, 0]) & (kps[i][:, 2] > 0)).sum() for i in g.index))
    for _, g in c.groupby(["clip", "shot"]):
        w, lab = g["width"].iloc[0], g["labeler"].iloc[0]
        med = kps[g.index[0]].copy()           # every row of a shot carries the shot's court
        kf = g[g["keyframe"] & g["kf_kp"].notna()]
        answers = [kparr(v, 14) for v in kf["kf_kp"]]
        shift = np.nan_to_num(kf[["cam_dx", "cam_dy"]].to_numpy(float))
        shift = shift - shift.mean(0) if len(shift) else shift
        xy, supported = consensus(answers, shift, COURT_PX * w)
        bad = {}
        for q in range(14):
            if med[q, 2] <= 0 or not np.isfinite(med[q, 0]) or not answers:
                continue
            if np.isfinite(xy[q, 0]):
                med[q, :2] = xy[q]
            if not supported[q]:
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
        order = g.sort_values("frame")
        steps = np.linalg.norm(np.diff(order[["cam_dx", "cam_dy"]].to_numpy(float), axis=0), axis=1)
        jumps = order["frame"].to_numpy()[1:][np.nan_to_num(steps) > CAM_JUMP * w]     # first frame after a jump
        kfr = kf["frame"].to_numpy()
        for i, r in g.iterrows():
            k = kps[i]
            d = np.array([r.cam_dx, r.cam_dy]) - ref if ref is not None else np.zeros(2)
            d = np.nan_to_num(d)
            k[:, :2] = np.where(k[:, 2:] > 0, med[:, :2] + d, k[:, :2])     # the shot's court, following the pan
            across = any(((jumps > min(r.frame, x)) & (jumps <= max(r.frame, x))).any() for x in kfr)
            moved = ref is not None and (np.linalg.norm(d) > CAM_FOLLOW * w or across)
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


def box_in_area(box, court: np.ndarray) -> bool:
    """A person (xyxy box) stands in the playing area: the foot (box bottom centre), or the foot
    FOOT_TOL box heights lower, is in it. Seen from low behind the baseline, a pixel near the far baseline is
    ~0.5 m, so a box bottom a few px high puts a far player metres back."""
    b = np.asarray(box, float)
    x, h = (b[0] + b[2]) / 2, b[3] - b[1]
    return in_area(np.array([x, b[3]]), court) or in_area(np.array([x, b[3] + FOOT_TOL * h]), court)


def check_players(people: pd.DataFrame, court: pd.DataFrame, fr: pd.DataFrame, flags: Flags) -> None:
    from dsa.label.clips import REACQ_S, reacquire_gate

    p = people.join(fr[["clip", "shot", "frame", "keyframe", "fps"]], on="sample")
    flags.denom["players"] = int(p["player"].fillna(False).sum())
    courts = {s: kparr(v, 14) for s, v in zip(court["sample"], court["kp"])}
    mask = set()
    for _, g in p.groupby(["clip", "shot"]):
        kf = g[g["keyframe"] & g["named"].notna()]
        for t, tg in kf.groupby("track"):
            yes = int(tg["named"].astype(bool).sum())
            if 0 < yes < len(tg) and abs(2 * yes - len(tg)) <= CHOICE_MARGIN:
                mask.update((i, "choice") for i in g.index[g["track"] == t])
        for i, r in kf[kf["named"].fillna(False).astype(bool)].iterrows():
            if r["sample"] in courts and not box_in_area(r.box, courts[r["sample"]]):
                mask.update((j, "outside") for j in g.index[g["track"] == r["track"]])
        expected = kf.groupby("sample")["named"].sum().max() if len(kf) else 0
        is_player = g["player"].fillna(False).astype(bool)
        ptracks = set(g.loc[is_player, "track"])
        if not ptracks:
            continue
        fps = g["fps"].iloc[0]
        seen = {t: (tg["frame"].to_numpy(), np.stack(tg["box"].map(np.asarray).to_list()))
                for t, tg in g[g["track"].isin(ptracks)].groupby("track")}
        for s, fg in g.groupby("sample"):
            present = set(fg["track"]) & ptracks
            if fg["keyframe"].iloc[0] or len(present) >= expected:
                continue
            f = fg["frame"].iloc[0]
            others = fg[~is_player[fg.index]]
            for t in ptracks - present:
                frames, boxes = seen[t]
                j = np.abs(frames - f).argmin()
                gap = abs(frames[j] - f)
                if gap > REACQ_S * fps:
                    continue                  # the player is long gone: nobody here can be mistaken for them
                b0 = boxes[j]
                for i, r in others.iterrows():
                    b = np.asarray(r["box"], float)
                    d = np.linalg.norm((b[:2] + b[2:]) / 2 - (b0[:2] + b0[2:]) / 2) / max(b[3] - b[1], b0[3] - b0[1])
                    if d <= reacquire_gate(gap, fps) and (s not in courts or box_in_area(b, courts[s])):
                        mask.add((i, "track lost"))
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


def court_by_camera(flags: Flags, fr: pd.DataFrame) -> pd.DataFrame:
    """Court points masked per check and camera type, as a % of the court points of that camera."""
    t = flags.table()
    t = t[t["head"] == "court"].join(fr["camera"], on="sample")
    n = t.groupby(["camera", "reason"]).size().unstack(fill_value=0)
    cams = sorted(k[1] for k in flags.denom if isinstance(k, tuple) and k[0] == "court")
    out = pd.DataFrame({"points": [flags.denom[("court", c)] for c in cams]}, index=cams)
    for r in ("unstable", "homography", "camera moved"):
        out[r] = [round(100 * n.get(r, {}).get(c, 0) / max(out.at[c, "points"], 1), 2) for c in cams]
    return out


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
    rep = report(flags)
    rep.attrs["court_by_camera"] = court_by_camera(flags, fr)
    return rep


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="clip labels source from dsa.label.clips")
    p.add_argument("--out", help="default <source>_clean")
    a = p.parse_args()
    rep = run(a.source, a.out or f"{a.source}_clean")
    print(rep.to_string(index=False))
    print("\ncourt points masked, % per camera\n" + rep.attrs["court_by_camera"].to_string())
    print(f"-> {LABELS / (a.out or a.source + '_clean')}")


if __name__ == "__main__":
    main()
