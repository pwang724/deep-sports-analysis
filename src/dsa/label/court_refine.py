"""Snap rough court points (Astra or TennisCourtDetector) onto the painted lines.

Given 14 rough points (dsa.astra.court POINTS order) and the frame:

  1. homography H from the reference court (dsa.astra.court REF) to the rough
     points (RANSAC, all points given).
  2. line pixels: white top-hat of the per-pixel minimum over B, G, R (white
     is bright in every channel; blue, green and clay courts are not), so a
     painted line is a narrow ridge whatever the surface.
  3. every model line (baselines, doubles and singles sidelines, service
     lines, centre service line) is sampled; at each sample the top-hat is
     read along the projected line's normal within +-R px and the ridge
     centre taken (a narrow peak only: a line crossing the normal gives a
     plateau and is ignored). Samples near a court point (line crossings)
     are skipped.
  4. H (8 parameters) and, with `distortion`, one radial term k1 are fitted
     to the ridge centres by robust least squares (point-to-projected-line
     distance, soft L1 at 1.5 px), with a weak pull (PRIOR_PX) towards the
     rough points so a direction the lines leave open stays put.
  5. repeated with a shrinking search radius (R_STEPS).

Quality: `support`, the share of in-frame line samples whose ridge lies
within INLIER_PX of the fitted lines, and `rms` of those residuals. The
result is None (keep the rough points) when the court is implausible
(plausible: behind the camera, flipped, far baseline below or longer than
the near one, not convex; checked on the start and on the fit), fewer than
MIN_SUPPORT (or MIN_INLIERS) of the samples support the fit, fewer than
MIN_LINES roughly horizontal or vertical court lines are found, more than
MAX_FALSE_LINE of the places where a court has no line (EMPTY) lie on one
(a fit shifted by one line), or the
points move more than MAX_SHIFT (median) from the rough answer.

Scored on 100 TennisCourtDetector val frames (the sample dsa.astra.court
scores, predictions cached in output/astra_eval/court/<model>/preds.npy;
a failed snap keeps the rough points): Astra 96.7% of points within 7 px,
median 2.2 px -> snapped 99.7%, 1.6 px (2% fail); TCD 95.6%, 1.9 px ->
97.9%, 1.6 px (4% fail). The radial term (distortion=True) scores worse on
these broadcast frames and is off by default.

    python -m dsa.label.court_refine --n 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from dsa.astra.court import REF, ROOT, sample, score

REF_A = np.array(REF, float)
# Model lines as pairs of point indexes; horizontal (across the court) first.
H_LINES = [(0, 1), (2, 3), (8, 9), (10, 11)]
V_LINES = [(0, 2), (1, 3), (4, 5), (6, 7), (12, 13)]
LINES = H_LINES + V_LINES
# Where a real court has no line: the near service line's continuation across the doubles alleys and the
# centre service line's continuation behind both service lines (short of the centre marks). A fit shifted
# by one line (a service line on the baseline, a sideline on the singles line) puts one of these on a
# painted line. The far alleys are left out: on a low camera the net lies across them; so does the net's
# white centre strap across the far centre continuation, which is only checked when the far service line
# is at least FAR_GAP_PX below the far baseline.
EMPTY = [((286, 2386), (423, 2386)), ((1242, 2386), (1379, 2386)), ((832, 1110), (832, 700)),
         ((832, 2386), (832, 2796))]
FAR_EMPTY = 2
FAR_GAP_PX = 60           # px at 1280
SAMPLES = 60              # samples per model line
R_STEPS = (24, 12, 6)       # search radius per iteration, px at 1280 wide
MIN_PEAK = 12             # top-hat grey levels a line ridge must reach
PEAK_FRAC = 0.4          # a ridge counts if at least this share of the brightest one in the window
KNOT_PX = 8               # skip samples this close (px at 1280) to a court point: crossings are ambiguous
INLIER_PX = 2.5           # px at 1280
MIN_SUPPORT = 0.5         # share of in-frame line samples that must be inliers
MIN_INLIERS = 100         # and at least this many (of 9 x SAMPLES): a court mostly out of frame proves little
MAX_FALSE_LINE = 0.3      # share of an EMPTY segment's samples allowed on a painted line
MIN_LINES = 3             # horizontal and vertical court lines each that must be found
MIN_LINE_SUPPORT = 0.3    # a line counts as found when this share of its samples are inliers
PRIOR_PX = 25.0           # sigma of the pull towards the rough points
MAX_SHIFT = 30.0          # px at 1280, median over the points
K1_SIGMA = 0.05           # prior on the radial term: broadcast lenses are near 0


def line_mask(bgr: np.ndarray) -> np.ndarray:
    """Top-hat of min(B, G, R): painted lines as bright ridges, float32."""
    lo = bgr.min(axis=2)
    k = max(9, int(round(bgr.shape[1] / 1280 * 17)) | 1)
    th = cv2.morphologyEx(lo, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return th.astype(np.float32)


def _project(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ h.T
    return p[:, :2] / p[:, 2:3]


class Model:
    """Reference court -> undistorted image by H; undistorted <-> distorted by one radial term about the
    image centre: u = c + (d - c)(1 + k1 |d - c|^2 / W^2)."""

    def __init__(self, h: np.ndarray, k1: float, size: tuple[int, int]):
        self.h, self.k1 = h / h[2, 2], k1
        self.c = np.array([size[0] / 2, size[1] / 2])
        self.w = size[0]

    def undistort(self, d: np.ndarray) -> np.ndarray:
        r2 = (((d - self.c) / self.w) ** 2).sum(-1, keepdims=True)
        return self.c + (d - self.c) * (1 + self.k1 * r2)

    def distort(self, u: np.ndarray) -> np.ndarray:
        d = u.copy()
        for _ in range(20):
            r2 = (((d - self.c) / self.w) ** 2).sum(-1, keepdims=True)
            d = self.c + (u - self.c) / (1 + self.k1 * r2)
        return d

    def points(self, ref: np.ndarray) -> np.ndarray:
        return self.distort(_project(self.h, ref))


def _samples(model: Model, size: tuple[int, int], margin_px: float, segments=None, n: int = SAMPLES,
             span: tuple[float, float] = (0.02, 0.98)):
    """(ref point, line index, image point, image normal) of every in-frame sample away from court points,
    on the model lines or on `segments` (pairs of reference points)."""
    kps = model.points(REF_A)
    out = []
    t = np.linspace(*span, n)
    segs = [(REF_A[a], REF_A[b]) for a, b in LINES] if segments is None else \
        [(np.array(a, float), np.array(b, float)) for a, b in segments]
    for li, (ra, rb) in enumerate(segs):
        ref = ra + t[:, None] * (rb - ra)
        img = model.points(ref)
        step = model.points(ref + 1e-3 * (rb - ra)) - img
        tang = step / (np.linalg.norm(step, axis=1, keepdims=True) + 1e-12)
        nrm = np.c_[-tang[:, 1], tang[:, 0]]
        inside = (img[:, 0] >= 2) & (img[:, 0] < size[0] - 2) & (img[:, 1] >= 2) & (img[:, 1] < size[1] - 2)
        far = np.min(np.linalg.norm(img[:, None] - kps[None], axis=2), axis=1) > margin_px
        for i in np.flatnonzero(inside & far & np.isfinite(img).all(1)):
            out.append((ref[i], li, img[i], nrm[i]))
    return out


def _ridges(mask: np.ndarray, samples, radius: float) -> list[tuple[int, np.ndarray]]:
    """Ridge centre along the normal of each sample, or none: (sample index, image point)."""
    if not samples:
        return []
    s = np.arange(-radius, radius + 0.5, 1.0, dtype=np.float32)
    img = np.array([q[2] for q in samples], np.float32)
    nrm = np.array([q[3] for q in samples], np.float32)
    xy = img[:, None] + s[None, :, None] * nrm[:, None]
    prof = cv2.remap(mask, xy[..., 0], xy[..., 1], cv2.INTER_LINEAR, borderValue=0)
    found = []
    for i, p in enumerate(prof):
        # the ridge nearest the prediction, not the brightest: a brighter line nearby (the net tape
        # above a far baseline on a low camera) must not pull the sample over
        top = p.max()
        if top < MIN_PEAK:
            continue
        loc = np.flatnonzero((p[1:-1] >= p[:-2]) & (p[1:-1] > p[2:]) & (p[1:-1] >= max(MIN_PEAK, PEAK_FRAC * top))) + 1
        if not len(loc):
            continue
        j = int(loc[np.argmin(np.abs(s[loc]))])
        peak = p[j]
        half = p >= max(peak / 2, np.median(p) + 0.5 * (peak - np.median(p)))
        lo, hi = j, j
        while lo > 0 and half[lo - 1]:
            lo -= 1
        while hi < len(p) - 1 and half[hi + 1]:
            hi += 1
        if hi - lo + 1 > max(0.6 * len(p), 12) or lo == 0 or hi == len(p) - 1:
            continue                           # a plateau (a crossing line, a bright area), not a ridge
        wts = p[lo:hi + 1] - peak / 2
        off = float((s[lo:hi + 1] * wts).sum() / max(wts.sum(), 1e-6))
        found.append((i, img[i] + off * nrm[i]))
    return found


def _residuals(model: Model, samples, ridges) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance of each ridge centre (undistorted) to its model line under H; line index per ridge."""
    if not ridges:
        return np.zeros(0), np.zeros(0, int)
    li = np.array([samples[i][1] for i, _ in ridges])
    pts = model.undistort(np.array([p for _, p in ridges]))
    ends = np.array([[REF_A[a], REF_A[b]] for a, b in LINES])
    ea, eb = _project(model.h, ends[:, 0]), _project(model.h, ends[:, 1])
    d = eb - ea
    n = np.c_[-d[:, 1], d[:, 0]] / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    return ((pts - ea[li]) * n[li]).sum(1), li


def plausible(h: np.ndarray) -> bool:
    """A court seen from behind the near baseline: every point in front of the camera, left left of right,
    far baseline above the near one and not longer than it, the outline convex."""
    w = np.c_[REF_A, np.ones(14)] @ h.T
    if not ((w[:, 2] > 0).all() or (w[:, 2] < 0).all()):
        return False
    p = w[:, :2] / w[:, 2:3]
    fl, fr, nl, nr = p[0], p[1], p[2], p[3]
    quad = np.float32([fl, fr, nr, nl])
    return bool(fl[0] < fr[0] and nl[0] < nr[0] and max(fl[1], fr[1]) < min(nl[1], nr[1])
                and np.linalg.norm(nr - nl) >= 0.95 * np.linalg.norm(fr - fl) and cv2.isContourConvex(quad))


def refine(bgr: np.ndarray, rough: np.ndarray, distortion: bool = False, mask: np.ndarray | None = None,
           debug: bool = False) -> dict | None:  # debug: return the fit whatever the checks say
    """Refined (14, 2) points, homography, k1 and quality, or None when the lines do not support a fit.

    `rough` is (14, 2) or (14, 3) (x, y, in-view flag; all points are used, hidden ones as estimates);
    NaN rows are ignored."""
    hgt, wid = bgr.shape[:2]
    size, px = (wid, hgt), wid / 1280
    rough = np.asarray(rough, float)[:, :2]
    ok = np.isfinite(rough).all(1)
    if ok.sum() < 4:
        return None
    h0, _ = cv2.findHomography(np.float32(REF_A[ok]), np.float32(rough[ok]), cv2.RANSAC, 15.0 * px)
    if h0 is None or not plausible(h0):
        return None
    mask = line_mask(bgr) if mask is None else mask
    model = Model(h0, 0.0, size)
    start = model.points(REF_A)
    # Conditioning: H acts on ref / 1000 and returns px / W.
    tr = np.diag([1e-3, 1e-3, 1.0])
    ti = np.diag([1 / wid, 1 / wid, 1.0])

    def unpack(x):
        g = np.append(x[:8], 1.0).reshape(3, 3)
        return Model(np.linalg.inv(ti) @ g @ tr, x[8] if distortion else 0.0, size)

    g0 = ti @ model.h @ np.linalg.inv(tr)
    x = np.append((g0 / g0[2, 2]).ravel()[:8], 0.0)
    samples, ridges = [], []
    for radius in R_STEPS:
        samples = _samples(model, size, KNOT_PX * px)
        ridges = _ridges(mask, samples, radius * px)
        if len(ridges) < 12:
            return None

        def fun(x):
            m = unpack(x)
            r, _ = _residuals(m, samples, ridges)
            prior = ((m.points(REF_A[ok]) - rough[ok]) / (PRIOR_PX * px)).ravel()
            return np.r_[r / px, prior * 1.5, [m.k1 / K1_SIGMA] if distortion else []]

        use = x if distortion else x[:8]
        sol = least_squares(lambda v: fun(v if distortion else np.append(v, 0.0)), use, loss="soft_l1",
                            f_scale=1.5, x_scale="jac", max_nfev=200)
        x = sol.x if distortion else np.append(sol.x, 0.0)
        model = unpack(x)
    samples = _samples(model, size, KNOT_PX * px)
    ridges = _ridges(mask, samples, R_STEPS[-1] * px)
    r, li = _residuals(model, samples, ridges)
    inl = np.abs(r) <= INLIER_PX * px
    per_line = np.array([inl[li == k].sum() / max(sum(1 for q in samples if q[1] == k), 1)
                         for k in range(len(LINES))])
    found = per_line >= MIN_LINE_SUPPORT
    n_h, n_v = int(found[:len(H_LINES)].sum()), int(found[len(H_LINES):].sum())
    pts = model.points(REF_A)
    support = float(inl.sum() / max(len(samples), 1))
    shift = float(np.median(np.linalg.norm(pts[ok] - rough[ok], axis=1)))
    false_line = 0.0                           # worst EMPTY segment: share of its samples on a painted line
    far_gap = np.linalg.norm(np.subtract(*model.points(np.array([[832.0, 561.0], [832.0, 1110.0]]))))
    for k, seg in enumerate(EMPTY):
        if k == FAR_EMPTY and far_gap < FAR_GAP_PX * px:
            continue                           # a low camera: the net's centre strap stands over it
        e = _samples(model, size, KNOT_PX * px, [seg], 20, (0.2, 0.8))
        if len(e) >= 5:
            hit = [i for i, q in _ridges(mask, e, 2 * INLIER_PX * px)
                   if abs(np.dot(q - e[i][2], e[i][3])) <= INLIER_PX * px]
            false_line = max(false_line, len(hit) / len(e))
    res = {"points": pts, "h": model.h, "k1": float(model.k1), "support": support,
           "rms": float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else np.inf,
           "lines_h": n_h, "lines_v": n_v, "shift": shift, "false_line": false_line, "shift_start": float(np.nanmax(
               np.linalg.norm(start[ok] - rough[ok], axis=1)))}
    if debug:
        return res
    res["ok"] = bool(plausible(model.h) and inl.sum() >= MIN_INLIERS and support >= MIN_SUPPORT
                     and false_line <= MAX_FALSE_LINE and n_h >= MIN_LINES and n_v >= MIN_LINES and shift <= MAX_SHIFT * px)
    return res if res["ok"] else None


def main() -> None:
    import pandas as pd

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--preds", default="output/astra_eval/court/gpt-6-astra-medium/preds.npy")
    p.add_argument("--out", default="output/court_refine")
    a = p.parse_args()
    preds = np.load(a.preds, allow_pickle=True).item()
    items = sample(a.n)
    rows = []
    for it in items:
        bgr = cv2.imread(str(ROOT / "images" / f"{it['id']}.png"))
        mask = line_mask(bgr)
        for lab in ("tcd", "astra"):
            xy = np.asarray(preds[f"{lab}/{it['id']}"], float)
            rows.append({**score(lab, xy, it), "failed": False})
            for dist in (False, True):
                r = refine(bgr, xy, distortion=dist, mask=mask)
                name = f"{lab}+refine" + ("+k1" if dist else "")
                if r is None:                   # failed: keep the rough answer
                    rows.append({**score(name, xy, it), "failed": True})
                else:
                    rows.append({**score(name, r["points"], it), "failed": False, "support": r["support"],
                                 "rms": r["rms"]})
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "tcd_val.parquet", index=False)
    summary = {}
    for name, g in df.groupby("labeler", sort=False):
        errs = pd.concat([g[f"err{k}"] for k in range(14) if f"err{k}" in g]).dropna()
        summary[name] = {"frames": len(g), "within7": round(float((errs <= 7).mean()), 4),
                         "within15": round(float((errs <= 15).mean()), 4),
                         "median_err": round(float(errs.median()), 2), "worst": round(float(errs.max()), 1),
                         "failed": round(float(g.failed.mean()), 3)}
    (out / "tcd_val_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
