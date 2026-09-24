"""Suggest the test set from clip-labelled frames, for precise review in dsa.label.review.

  random   ~150 analysis-view singles frames (--view) and ~15 non-view frames
           (--nonview), spread evenly over collection x camera x surface x
           level strata and random within each (at most --per-clip frames a
           clip, --gap-s apart), so scores are unbiased within a stratum;
           `weight` (stratum share of the pool / share of the sample)
           reweights them to the whole pool. Frames Astra called doubles are
           left out.
  hard     up to --hard frames the consistency filters masked something on
           (<source>_clean/flags) or the user flagged in the video review
           (data/gold/<flags-version>/video_flags, within FLAG_WINDOW_S),
           round-robin over reasons, from the test videos first. Scored
           separately: they are chosen for being wrong.

Every video that gives a frame is held out (videos.txt). Writes
data/gold/<version>/frames.parquet and the labels source gold_<version>_prefill
built from the raw clip labels (no labeler is re-run), so the review tool
works directly:

    python -m dsa.label.testset --source clips_v1 --version v2
    python -m dsa.label.review serve --version v2 --prefill gold_v2_prefill
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dsa.data import schema
from dsa.data.paths import GOLD, LABELS

STRATA = ["collection", "camera", "surface", "level"]
FLAG_WINDOW_S = 0.5            # a video flag marks the moment a key was pressed, give or take reaction time


def allocate(sizes: pd.Series, n: int) -> pd.Series:
    """Even split of n over strata, capped by each stratum's size, leftovers passed on."""
    take = pd.Series(0, index=sizes.index)
    while n > 0 and (take < sizes).any():
        open_ = sizes.index[take < sizes]
        share = max(n // len(open_), 1)
        for s in open_:
            add = min(share, sizes[s] - take[s], n)
            take[s] += add
            n -= add
            if n == 0:
                break
    return take


def spaced(pool: pd.DataFrame, k: int, gap_s: float, per_clip: int, used: pd.DataFrame, rng) -> list:
    """Up to k random rows of pool, no two in a clip within gap_s, at most per_clip a clip (counting `used`)."""
    picked = []
    taken = {c: list(g["frame"] / g["fps"]) for c, g in used.groupby("clip")}
    for i in rng.permutation(len(pool)):
        if len(picked) == k:
            break
        r = pool.iloc[i]
        t = taken.setdefault(r["clip"], [])
        if len(t) < per_clip and all(abs(r["frame"] / r["fps"] - u) >= gap_s for u in t):
            t.append(r["frame"] / r["fps"])
            picked.append(pool.index[i])
    return picked


def stratified(pool: pd.DataFrame, n: int, gap_s: float, per_clip: int, rng) -> pd.DataFrame:
    if not len(pool) or n <= 0:
        return pool.iloc[:0]
    strata = pool[STRATA].fillna("unknown").astype(str).agg(" / ".join, axis=1)
    # Stratum sizes in clips x per-clip cap, so the allocation is what spacing can actually deliver.
    cap = pool.assign(s=strata).groupby("s")["clip"].nunique() * per_clip
    sizes = pd.concat([strata.value_counts(), cap], axis=1).min(axis=1)
    take = allocate(sizes, n)
    picked, share = [], strata.value_counts(normalize=True)
    for s, k in take.items():
        rows = spaced(pool[strata == s], int(k), gap_s, per_clip, pool.iloc[:0], rng)
        picked += rows
    out = pool.loc[picked].copy()
    st = strata.loc[picked]
    out["stratum"] = st
    out["weight"] = (st.map(share) / st.map(st.value_counts(normalize=True))).round(3)
    return out


def hard_frames(source: str, clean: str, flags_version: str, frames: pd.DataFrame) -> pd.DataFrame:
    """Frames with a consistency flag or a user video flag, with the reason."""
    rows = []
    if (LABELS / clean / "flags.parquet").exists():
        f = schema.read(clean, "flags")
        rows += [{"sample": s, "reason": f"filter: {h} {r}"} for s, h, r in zip(f["sample"], f["head"], f["reason"])]
    by_clip = {c: g for c, g in frames.groupby("clip")}
    for p in sorted((GOLD / flags_version / "video_flags").glob("*.json")):
        if p.name.startswith("._"):
            continue
        b = json.loads(p.read_text())
        g = by_clip.get(b["clip"])
        if g is None:
            continue
        for fl in b.get("flags", []):
            f0 = g["frame"].min() + fl["frame"]
            near = g[(g["frame"] - f0).abs() <= FLAG_WINDOW_S * g["fps"]]
            rows += [{"sample": s, "reason": f"video: {fl['tag']}"} for s in near["sample"]]
    return pd.DataFrame(rows, columns=["sample", "reason"]).drop_duplicates()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="raw clip labels source (dsa.label.clips)")
    p.add_argument("--clean", help="filtered source with the flags table; default <source>_clean")
    p.add_argument("--version", required=True)
    p.add_argument("--flags-version", help="video review version holding video_flags/; default --version")
    p.add_argument("--view", type=int, default=150)
    p.add_argument("--nonview", type=int, default=15)
    p.add_argument("--hard", type=int, default=30)
    p.add_argument("--per-clip", type=int, default=4)
    p.add_argument("--gap-s", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    clean = a.clean or f"{a.source}_clean"

    fr = schema.read(a.source, "frames")
    scene = schema.read(a.source, "scene")
    fr = fr.merge(scene[["sample", "singles"]], on="sample", how="left")
    heads = fr["heads"].fillna("").str.split(",")
    view_pool = fr[(fr["scope"] == "view") & heads.map(lambda h: "people" in h) & fr["singles"].fillna(False).astype(bool)]
    nonview_pool = fr[(fr["scope"] == "nonview") & heads.map(lambda h: "scene" in h)]
    view = stratified(view_pool, a.view, a.gap_s, a.per_clip, rng).assign(subset="random")
    nonview = stratified(nonview_pool, a.nonview, a.gap_s, a.per_clip, rng).assign(subset="nonview")
    chosen = pd.concat([view, nonview])

    hard = hard_frames(a.source, clean, a.flags_version or a.version, fr)
    hard = hard[hard["sample"].isin(view_pool["sample"]) & ~hard["sample"].isin(chosen["sample"])]
    hard = hard.merge(fr, on="sample")
    hard = hard.sample(frac=1, random_state=a.seed)
    hard["held"] = hard["video"].isin(set(chosen["video"]))
    queues = {r: g.sort_values("held", ascending=False, kind="stable").to_dict("records")
              for r, g in hard.groupby("reason")}
    taken = {c: list(g["frame"] / g["fps"]) for c, g in chosen.groupby("clip")}
    picked, names = [], set()
    while len(picked) < a.hard and any(queues.values()):
        for reason in sorted(queues):              # round-robin over reasons
            q = queues[reason]
            while q and len(picked) < a.hard:
                r = q.pop(0)
                t, when = taken.setdefault(r["clip"], []), r["frame"] / r["fps"]
                if r["sample"] not in names and len(t) < a.per_clip + 2 and all(abs(when - u) >= a.gap_s for u in t):
                    t.append(when)
                    picked.append(r)
                    names.add(r["sample"])
                    break
    hard = pd.DataFrame(picked).drop(columns=["held"], errors="ignore").assign(subset="hard")
    chosen = pd.concat([chosen, hard], ignore_index=True)

    src = f"gold_{a.version}"
    ids = {s: schema.sample_id(src, s.split("/", 1)[1].rsplit("/", 1)[0], int(s.rsplit("/", 1)[1])) for s in chosen["sample"]}
    out = chosen.rename(columns={"sample": "clip_sample"})
    out.insert(0, "sample", out["clip_sample"].map(ids))
    out["source"], out["split"] = src, "test"
    keep = schema.TABLES["frames"] + ["clip_sample", "subset", "reason", "stratum", "weight", "clip", "collection", "video",
                                      "camera", "surface", "level", "play", "scope"]
    out = out[[c for c in keep if c in out]].astype({"frame": int, "width": int, "height": int})
    schema.validate("frames", out)
    d = GOLD / a.version
    d.mkdir(parents=True, exist_ok=True)
    out.to_parquet(d / "frames.parquet", index=False)
    (d / "videos.txt").write_text("\n".join(sorted(out["video"].dropna().unique())) + "\n")

    # The pre-fill source for the review tool: the raw clip labels on these frames, re-keyed.
    tables = {"frames": out}
    for t in ("people", "rackets", "ball", "court", "scene"):
        df = schema.read(a.source, t)
        df = df[df["sample"].isin(ids)].copy()
        df["sample"] = df["sample"].map(ids)
        tables[t] = df
    tables["people"]["player"] = tables["people"]["player"].fillna(False).astype(bool)
    tables["court"] = tables["court"].drop(columns=["kf_kp"], errors="ignore")
    for c in ("view", "in_play"):             # the review tool reads booleans; an unknown in play starts as False
        tables["scene"][c] = tables["scene"][c].fillna(False).astype(bool)
    schema.write(f"{src}_prefill", tables)

    print(out.groupby("subset").agg(frames=("sample", "size"), clips=("clip", "nunique"), videos=("video", "nunique")).to_string())
    print("\nrandom by stratum:\n" + view["stratum"].value_counts().to_string())
    if len(hard):
        print("\nhard by reason:\n" + hard["reason"].value_counts().to_string())
    print(f"\n{out['video'].nunique()} videos held out -> {d / 'videos.txt'}")
    print(f"-> {d / 'frames.parquet'}, labels source {src}_prefill")
    print(f"   python -m dsa.label.review serve --version {a.version} --prefill {src}_prefill")


if __name__ == "__main__":
    main()
