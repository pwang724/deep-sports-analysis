"""Full broadcast videos behind the E2E-Spot and F3Set event labels, from YouTube.

    python -m dsa.data.broadcast probe --source f3set       # yt-dlp -J per match -> info/<name>.json, plan table
    python -m dsa.data.broadcast download --source f3set    # full videos, one at a time, skips existing files
    python -m dsa.data.broadcast check --source f3set       # decoded fps / frame count vs the labels
    python -m dsa.data.broadcast sheet --source f3set       # contact sheets of random hits, to check alignment

Videos land at the events' `media` path, data/videos/<source>/<match>.mp4
(not data/videos/broadcast, which the gold and clip samplers glob). YouTube
ids come from the sources' videos.csv (E2E-Spot: data/tennis, F3Set:
data/f3set-tennis). 21 YouTube videos are in both sources under different
names; a video already on disk under either name (or in data/videos/broadcast)
at the labelled rate is copied instead of downloaded again. Whole videos
rather than labelled sections: the labels cover most of each match, the
whole set fits on disk, and a whole file keeps frame numbers exact (a cut
section starts at a keyframe, not at the frame asked for).

Format: <= 720p (E2E-Spot's 1080p only changes pixels; events carry no
coordinates), h264 preferred for fast seeking, m4a audio kept, and the frame
rate the labels were made at: 25 / 30 / 24, never the 50 / 60 fps rendition
of a match labelled at 25 / 30 (all 138 matches have one, including the
Jabeur QF whose earlier copy in broadcast/ is 59.94 fps). Should YouTube
only serve the high rate, the file is 50 / 60 fps and a label frame f is
video frame round(f / fps * video fps) (`video_frame`); `check` reports these.

YouTube rate-limits: one download at a time, 5-15 s sleeps, `--cookies-from-browser
chrome` (read from the browser each run, never written to disk), and the run stops
after two sign-in / 429 errors in a row.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dsa.data.paths import DATA, LABELS, SCRATCH, SOURCES, VIDEOS
from dsa.data.youtube import GENTLE, YTDLP

OLD = VIDEOS / "broadcast"   # the first three matches, downloaded by hand before this module
CSV = {"e2e_spot": SOURCES / "e2e_spot/data/tennis/videos.csv",
       "f3set": SOURCES / "f3set/data/f3set-tennis/videos.csv"}
BLOCKED = re.compile(r"not a bot|Sign in|HTTP Error 429|Too Many Requests", re.I)


def matches(source: str) -> pd.DataFrame:
    """One row per labelled match: name, media, fps (as labelled), yt_id ('' if the source gives none)."""
    spans = pd.read_parquet(LABELS / source / "spans.parquet")
    ids = {r["name" if "name" in r else "video_name"]: r["yt_id"] for r in csv.DictReader(open(CSV[source]))}
    df = spans.groupby("media", as_index=False).agg(fps=("fps", "median"), last=("end", "max"))
    df["name"] = df.media.map(lambda m: Path(m).stem)
    df["yt_id"] = df.name.map(lambda n: ids.get(n, ""))
    return df


def path(media: str) -> Path:
    return DATA / media


def info_path(source: str, name: str) -> Path:
    return VIDEOS / source / "info" / f"{name}.json"


def on_disk(yt_id: str, fps: float) -> Path | None:
    """The same YouTube video already downloaded under any name, at `fps` (either source, or broadcast/)."""
    for source in CSV:
        for r in csv.DictReader(open(CSV[source])):
            if r["yt_id"].split("&")[0] != yt_id.split("&")[0]:
                continue
            name = r.get("name") or r["video_name"]
            for p in (VIDEOS / source / f"{name}.mp4", OLD / f"{name}.mp4"):
                if p.exists() and abs(stream(p)["video_fps"] - fps) < 0.01:
                    return p
    return None


def fmt(fps: float, height: int = 720) -> str:
    """yt-dlp format: <= height at the labelled rate (h264 first), then any rate as a last resort."""
    r = f"[fps>={round(fps) - 1}][fps<={round(fps) + 1}]"
    v = [f"bv*[height<={height}]{r}[vcodec^=avc1]", f"bv*[height<={height}]{r}", f"bv*[height<={height}]"]
    return "/".join(f"{x}+ba[ext=m4a]/{x}+ba" for x in v)


def ytdlp(cookies: str | None) -> list[str]:
    return YTDLP + (["--cookies-from-browser", cookies] if cookies else [])


def probe(source: str, cookies: str | None, sleep: float) -> None:
    """Metadata for every labelled match (cached), then the plan: formats, hours, GB."""
    (VIDEOS / source / "info").mkdir(parents=True, exist_ok=True)
    rows, blocked = [], 0
    for m in matches(source).itertuples():
        info = info_path(source, m.name)
        if m.yt_id and not info.exists():
            out = subprocess.run(ytdlp(cookies) + ["-J", f"https://www.youtube.com/watch?v={m.yt_id}"],
                                 capture_output=True, text=True)
            if out.returncode:
                err = (out.stderr.strip().splitlines() or ["?"])[-1][:160]
                print(f"{m.name}: {err}")
                info.with_suffix(".err").write_text(err)
                blocked = blocked + 1 if BLOCKED.search(err) else 0
                if blocked >= 2:
                    print("stopping: YouTube is asking to sign in / rate limiting")
                    break
            else:
                d = json.loads(out.stdout)
                keep = ["id", "title", "channel", "duration", "width", "height", "fps", "upload_date"]
                fmts = [{k: f.get(k) for k in ("format_id", "ext", "vcodec", "acodec", "height", "fps",
                                               "filesize", "filesize_approx", "tbr")} for f in d.get("formats", [])]
                info.write_text(json.dumps({k: d.get(k) for k in keep} | {"formats": fmts}))
                info.with_suffix(".err").unlink(missing_ok=True)
                blocked = 0
            time.sleep(sleep)
        rows.append(plan_row(m, info))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    ok = df[df.duration_h.notna()]
    print(f"\n{len(ok)}/{len(df)} available: {ok.duration_h.sum():.1f} h, ~{ok.gb.sum():.0f} GB; "
          f"{(ok.pick_fps.round() != ok.fps.round()).sum()} without a format at the labelled rate")


def plan_row(m, info: Path) -> dict:
    row = dict(name=m.name, yt_id=m.yt_id, fps=round(m.fps, 3), label_end_h=round(m.last / m.fps / 3600, 2),
               duration_h=None, pick_fps=None, pick=None, gb=None, have=path(m.media).exists())
    if not info.exists():
        row["pick"] = info.with_suffix(".err").read_text()[:60] if info.with_suffix(".err").exists() else "no yt id"
        return row
    d = json.loads(info.read_text())
    video = [f for f in d["formats"] if f.get("vcodec") not in (None, "none") and (f.get("height") or 0) <= 720
             and f.get("fps")]
    exact = [f for f in video if abs(f["fps"] - round(m.fps)) <= 1] or video
    exact.sort(key=lambda f: (f["height"] or 0, str(f["vcodec"]).startswith("avc1")), reverse=True)
    f = exact[0] if exact else {}
    size = f.get("filesize") or f.get("filesize_approx") or (f.get("tbr") or 0) * 125 * d["duration"]
    return row | dict(duration_h=round(d["duration"] / 3600, 2), pick_fps=f.get("fps"),
                      pick=f"{f.get('format_id')} {f.get('height')}p {str(f.get('vcodec'))[:4]}",
                      gb=round((size + 16000 * d["duration"]) / 1e9, 2))


def download(source: str, cookies: str | None, sleep: float, only: list[str] | None) -> None:
    todo = matches(source)
    todo = todo[todo.name.isin(only)] if only else todo
    blocked = 0
    for n, m in enumerate(todo.itertuples(), 1):
        out = path(m.media)
        if out.exists() or not m.yt_id:
            print(f"[{n}/{len(todo)}] {m.name}: {'have it' if out.exists() else 'no YouTube id'}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        if same := on_disk(m.yt_id, m.fps):
            shutil.copyfile(same, out.with_suffix(".dl.mp4"))
            out.with_suffix(".dl.mp4").rename(out)
            print(f"[{n}/{len(todo)}] {m.name}: copied from {same}")
            continue
        t = time.time()
        r = subprocess.run(ytdlp(cookies) + GENTLE[:2] + ["-f", fmt(m.fps), "--merge-output-format", "mp4",
                                                         "--no-part", "-o", str(out.with_suffix(".dl.%(ext)s")),
                                                         f"https://www.youtube.com/watch?v={m.yt_id}"],
                           capture_output=True, text=True)
        done = out.with_suffix(".dl.mp4")
        if r.returncode == 0 and done.exists():
            done.rename(out)
            blocked = 0
            print(f"[{n}/{len(todo)}] {m.name}: {out.stat().st_size / 1e9:.2f} GB in {time.time() - t:.0f} s", flush=True)
        else:
            err = (r.stderr.strip().splitlines() or ["?"])[-1][:200]
            print(f"[{n}/{len(todo)}] {m.name}: FAILED {err}", flush=True)
            blocked = blocked + 1 if BLOCKED.search(err) else 0
            if blocked >= 2:
                print("stopping: YouTube is asking to sign in / rate limiting; wait and rerun")
                return
        time.sleep(random.uniform(sleep, 3 * sleep))


def stream(p: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_frames,codec_name:format=duration", "-of", "json",
                          str(p)], capture_output=True, text=True)
    d = json.loads(out.stdout)
    s, f = d["streams"][0], d["format"]
    num, den = map(int, s["r_frame_rate"].split("/"))
    return dict(codec=s["codec_name"], width=s["width"], height=s["height"], video_fps=num / den,
                frames=int(s.get("nb_frames") or 0), duration_s=float(f["duration"]))


def video_frame(frame: int, fps: float, video_fps: float) -> int:
    """A label frame (at the labelled fps) in a video at another rate; the identity when the rates match."""
    return frame if abs(fps - video_fps) < 0.01 else round(frame / fps * video_fps)


def check(source: str) -> pd.DataFrame:
    """Per downloaded match: decoded rate vs labelled rate, and whether every label frame is inside the video."""
    rows = []
    for m in matches(source).itertuples():
        if path(m.media).exists():
            s = stream(path(m.media))
            rows.append(dict(name=m.name[:50], fps=round(m.fps, 3), **s, rate_ok=abs(s["video_fps"] - m.fps) < 0.01,
                             last_label=m.last, fits=video_frame(m.last, m.fps, s["video_fps"]) <= s["frames"] + 1))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))
    if len(df):
        print(f"\n{len(df)} videos, {df.duration_s.sum() / 3600:.1f} h; rate differs: "
              f"{list(df.name[~df.rate_ok])}; labels past the end: {list(df.name[~df.fits])}")
    return df


def read(p: Path, frames: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(p))
    out = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, bgr = cap.read()
        out.append(bgr if ok else np.zeros((720, 1280, 3), np.uint8))
    cap.release()
    return out


def sheet(source: str, videos: int, events: int, seed: int, out_dir: Path) -> None:
    """Per video, `events` random hits: frames f-2, f, f+2 of the hitter's half (far half enlarged), captioned."""
    ev = pd.read_parquet(LABELS / source / "events.parquet")
    ev = ev[ev.type.isin(["serve", "shot"]) & ev.media.map(lambda m: path(m).exists())]
    rng = random.Random(seed)
    media = sorted(ev.media.unique())
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in rng.sample(media, min(videos, len(media))):
        vfps = stream(path(m))["video_fps"]
        rows = []
        for e in ev[ev.media == m].sample(events, random_state=seed).sort_values("frame").itertuples():
            step = max(1, round(vfps / e.fps))
            f = video_frame(e.frame, e.fps, vfps)
            tiles = []
            for k, im in zip((-2, 0, 2), read(path(m), [f - 2 * step, f, f + 2 * step])):
                h, w = im.shape[:2]
                # the far player at native size, the near half shrunk to the same width
                im = im[: int(0.5 * h), int(0.15 * w): int(0.85 * w)] if e.side == "far" else \
                    cv2.resize(im[int(0.35 * h):], (768, round(768 * 0.65 * h / w)))
                im = cv2.resize(im, (768, round(768 * im.shape[0] / im.shape[1])))
                txt = f"{e.side} {e.type} {e.hand if isinstance(e.hand, str) else ''} f={e.frame}{k:+d} (video {f + k * step})"
                cv2.rectangle(im, (0, 0), (768, 20), (0, 0, 0), -1)
                cv2.putText(im, txt, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (0, 255, 255) if k == 0 else (255, 255, 255), 1)
                tiles.append(im)
            rows.append(np.hstack(tiles))
        w = max(r.shape[1] for r in rows)
        grid = np.vstack([np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0))) for r in rows])
        p = out_dir / f"{source}_{Path(m).stem[:60]}.jpg"
        cv2.imwrite(str(p), grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(p)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("probe", "download", "check", "sheet"):
        s = sub.add_parser(name)
        s.add_argument("--source", required=True, choices=sorted(CSV))
        if name in ("probe", "download"):
            s.add_argument("--cookies-from-browser", default="chrome", help="'' for none")
            s.add_argument("--sleep", type=float, default=5.0)
        if name == "download":
            s.add_argument("--only", nargs="*", help="match names")
        if name == "sheet":
            s.add_argument("--videos", type=int, default=5)
            s.add_argument("--events", type=int, default=5)
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--out", type=Path, default=SCRATCH / "broadcast_alignment")
    a = p.parse_args()
    if a.cmd == "probe":
        probe(a.source, a.cookies_from_browser or None, a.sleep)
    elif a.cmd == "download":
        download(a.source, a.cookies_from_browser or None, a.sleep, a.only)
    elif a.cmd == "check":
        check(a.source)
    else:
        sheet(a.source, a.videos, a.events, a.seed, a.out)


if __name__ == "__main__":
    main()
