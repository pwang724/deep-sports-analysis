"""YouTube tennis footage for labelling: search, select, download short segments.

    python -m dsa.data.youtube search [--per-query 20]      # metadata only, appends to the manifest
    python -m dsa.data.youtube select [--n 150]              # diverse pick, <= 4 videos per channel
    python -m dsa.data.youtube download [--workers 3]        # 3 x 60 s per selected video, <= 720p mp4
    python -m dsa.data.youtube reject ID [ID ...] --why "..."  # after looking; select refills
    python -m dsa.data.youtube looked FILE                  # tags set by eye: "id camera surface light note"
    python -m dsa.data.youtube sheet [--n 24]               # contact sheet -> data/scratch/youtube_contact_sheet.jpg
    python -m dsa.data.youtube summary

Everything is recorded in data/videos/youtube/manifest.csv (one row per video id).
Segments land in data/videos/youtube/<video_id>/seg<k>_<start s>.mp4. The
camera / surface / play / level columns are keyword guesses from the title and
description until someone looks at the frames (`looked`, checked=eye).

Camera: baseline_high (elevated, > ~3 m, behind the baseline), baseline_low (behind
the baseline at <= ~2.5 m, including phones clipped to the back fence), broadcast,
fence (side-fence mount), side, corner, pov (worn by a player), drone.

Scope (for now): singles, behind-the-baseline cameras. `select` rejects doubles and
other angles ("scope: doubles" / "scope: angle"); their files stay on disk but are
not counted. No cookies by default; `download --cookies-from-browser chrome` uses the
browser's YouTube session when YouTube asks to sign in (the user's choice, cookies are
never written to disk). yt-dlp uses node for YouTube's JS challenge.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from dsa.data.paths import SCRATCH, VIDEOS

ROOT = VIDEOS / "youtube"
MANIFEST = ROOT / "manifest.csv"
YTDLP = [str(Path(".venv/bin/yt-dlp")), "--js-runtimes", "node", "--no-warnings"]
FORMAT = "bv*[height<=720][vcodec^=avc1]+ba[ext=m4a]/b[height<=720][ext=mp4]/bv*[height<=720]+ba/b[height<=720]"
GENTLE = ["--sleep-requests", "1", "--sleep-interval", "5", "--max-sleep-interval", "15"]
MAX_GB = 40.0

COLUMNS = ["video_id", "url", "title", "channel", "channel_id", "duration_s", "upload_date", "width", "height",
           "fps", "license", "view_count", "query", "category", "camera", "surface", "play", "level",
           "light", "checked", "selected", "segments", "status", "notes", "description"]

# (category, query). Category is our tag for why the video was looked for.
QUERIES = [
    ("club", "club tennis match full baseline view"),
    ("club", "amateur tennis match full"),
    ("club", "recreational tennis match full set"),
    ("club", "tennis match point of view behind baseline amateur"),
    ("usta", "4.0 USTA match full"),
    ("usta", "3.5 USTA tennis match"),
    ("usta", "4.5 NTRP tennis match full"),
    ("utr", "UTR tennis match full"),
    ("utr", "UTR 8 tennis match"),
    ("utr", "UTR pro tennis tour match full"),
    ("junior", "junior tennis match full court"),
    ("junior", "ITF junior tennis match full"),
    ("junior", "U14 tennis match full"),
    ("kids", "kids tennis match 10 and under"),
    ("college", "college tennis match full"),
    ("college", "NCAA D1 tennis singles match"),
    ("college", "D3 college tennis match"),
    ("clay", "clay court tennis match amateur"),
    ("clay", "har-tru tennis match club"),
    ("grass", "grass court club tennis"),
    ("grass", "grass court tennis match amateur"),
    ("indoor", "indoor tennis match full"),
    ("indoor", "indoor carpet tennis match"),
    ("night", "night tennis match recreational"),
    ("night", "tennis match under lights club"),
    ("doubles", "doubles tennis match full rec"),
    ("doubles", "mixed doubles tennis match club"),
    ("doubles", "4.0 doubles tennis match"),
    ("pro_practice", "tennis practice session pro court level"),
    ("pro_practice", "ATP practice session full"),
    ("pro_practice", "WTA practice court level"),
    ("pro_practice", "Challenger tennis practice session"),
    ("drill", "tennis drills baseline camera"),
    ("drill", "tennis hitting session baseline rally"),
    ("drill", "tennis serve practice session"),
    ("lesson", "tennis lesson full court view"),
    ("lesson", "private tennis lesson full"),
    ("gopro", "gopro tennis match"),
    ("gopro", "tennis match filmed on phone"),
    ("fence", "tennis match fence camera"),
    ("side", "tennis match side view camera"),
    ("drone", "drone tennis match"),
    ("senior", "senior tennis match"),
    ("senior", "over 60 tennis match"),
    ("women", "women's club tennis match"),
    ("women", "women's 3.5 tennis match"),
    ("women", "college women's tennis match full"),
    ("wheelchair", "wheelchair tennis match full"),
    ("futures", "ITF M15 tennis match full"),
    ("futures", "ITF W15 tennis match full"),
    ("broadcast", "ATP tennis full match"),
    ("broadcast", "WTA full match"),
    ("es", "partido de tenis amateur completo"),
    ("es", "partido tenis tierra batida"),
    ("es", "tenis dobles partido amateur"),
    ("fr", "match de tennis amateur complet"),
    ("fr", "match tennis terre battue club"),
    ("de", "Tennis Match Amateur ganzes Spiel"),
    ("de", "Tennis Medenspiel"),
    ("de", "Tennis Halle Match LK"),
    ("it", "partita di tennis amatoriale completa"),
    ("it", "tennis terra rossa partita"),
    ("ja", "テニス 試合 シングルス 一般"),
    ("ja", "テニス ダブルス 試合"),
    ("ja", "ソフトテニス 試合"),
    ("pt", "jogo de tênis amador completo"),
    ("pt", "tênis saibro jogo"),
    ("nl", "tennis wedstrijd competitie"),
    ("ko", "테니스 경기 단식"),
    ("zh", "网球 比赛 业余"),
]

# Title words that mean edited footage, not continuous play.
REJECT = re.compile(r"highlight|best points|top \d+|compilation|slow ?mo|reaction|react|funny|fails|ps5|ps4|xbox|"
                    r"tennis world tour|ao tennis|top ?spin|gameplay|podcast|interview|press conference|vlog|"
                    r"how to|tutorial|explained|analysis|breakdown|#shorts|mejores puntos|resumen|meilleurs points|"
                    r"hot shots|trick shot|montage|pickleball|\bpb\b|dink|pklric|padel|pádel|table tennis|badminton|squash|soft ?tennis|ソフトテニス|"
                    r"pickle|beach tennis|camera bag|review|unboxing|rules|drone show|points|shots|"
                    r"tribute|documentary|prodigy|week \d|lesson \d+ di|試合に勝ちたい|"
                    r"archive|\b19[5-9]\d\b|tips|tactics|adjustments|what is it like|\bwhy\b|movie|episode|"
                    r"point slam|online|debate|strateg|wall practice|recording|blooper|laugh|mistakes|adapta|failure|minutes straight|king of", re.I)

# Tournament / tour channels: their uploads are TV broadcasts.
OFFICIAL = re.compile(r"^(Wimbledon|Australian Open|US Open|Roland-Garros|Tennis TV|WTA|ATP|United States Tennis|"
                      r"Tennis Channel|Eurosport|NCAA Championships|Billie Jean King Cup|Davis Cup|World Tennis\b)", re.I)
# How many picks each query category deserves, relative to 1. Self-recorded amateur play is the target.
WEIGHT = {"club": 3, "usta": 3, "utr": 2, "junior": 2, "college": 2, "clay": 2, "grass": 2, "indoor": 2,
          "night": 2, "doubles": 0.3, "drill": 1.5, "lesson": 1.5, "gopro": 1, "fence": 1, "side": 0.3,
          "senior": 1.5, "women": 2, "kids": 1, "drone": 0.1, "wheelchair": 0.5, "futures": 1, "broadcast": 1,
          "pro_practice": 1.5}
MAX_BROADCAST = 12
SCOPE_CAMERAS = {"baseline_high", "baseline_low", "broadcast", "unknown"}
BASELINE_HINT = re.compile(r"baseline|behind|court[- ]level|full court|swingvision|playsight|tripod|full match", re.I)

GUESSES = {
    "camera": [("drone", r"drone|dji|aerial|vogel"), ("fence", r"fence|zaun"),
               ("side", r"side ?view|sideline|side[- ]on|side angle|vue de côté|lateral"),
               ("corner", r"corner"),
               ("baseline_low", r"court level|ground level|low angle|player'?s view|pov|eye level"),
               ("baseline_high", r"high angle|elevated|birds?[- ]eye|behind the baseline|baseline view|"
                                 r"full court view|playsight|swingvision|pov"),
               ("broadcast", r"\batp\b|\bwta\b|grand slam|open 20|tennis tv")],
    "surface": [("clay", r"clay|har-?tru|arcilla|tierra batida|terre battue|sand|terra rossa|saibro|クレー|"
                         r"red court|green clay|roland"),
                ("grass", r"grass|gazon|rasen|erba|césped|wimbledon|queens|halle open|芝"),
                ("indoor", r"indoor|carpet|halle|hall\b|couvert|coperto|インドア|cubierta|bubble"),
                ("hard", r"hard ?court|acrylic|pista dura|dur\b|us open|australian open|cemento|ハード")],
    "play": [("doubles", r"doubles|dobles|double\b|doppel|doppio|ダブルス|duplas|복식|双打"),
             ("drill", r"drill|exercise|ejercicio|exercice|übung|esercizi|練習メニュー"),
             ("lesson", r"lesson|coaching|clase|cours|lezione|レッスン|aula|trainerstunde"),
             ("practice", r"practice|training|hitting|hit session|entrenamiento|entraînement|allenamento|"
                          r"練習|treino|rally"),
             ("match", r"match|partido|spiel|partita|試合|jogo|wedstrijd|경기|比赛|vs\.?|set\b")],
    "level": [("junior", r"junior|kids|\bu1\d\b|u-1\d|under 1\d|juvenil|jeunes|jugend|jr\.?\b|high school|"
                         r"10 and under|ジュニア|中学|高校|infantil|youth|boys|girls"),
              ("college", r"college|ncaa|\bd[123]\b|university|varsity|大学|collegiate"),
              ("pro", r"\batp\b|\bwta\b|challenger|\bitf\b|m15|m25|w15|w25|djokovic|alcaraz|sinner|nadal|"
                      r"federer|medvedev|zverev|rune|fritz|swiatek|sabalenka|gauff|rybakina|pegula|"
                      r"practice pass|grand slam|pro tennis|tour\b|futures"),
              ("rec", r"club|amateur|rec\b|recreational|usta|ntrp|utr|\b[2345]\.[05]\b|\blk\s?\d|league|"
                      r"senior|over \d0|medenspiel|amador|amatoriale|一般|社会人|competitie|业余|동호인")],
}


def guess(kind: str, text: str) -> str:
    for label, pattern in GUESSES[kind]:
        if re.search(pattern, text, re.I):
            return label
    return "unknown"


def tag(r: pd.Series) -> dict:
    """Keyword guesses for camera / surface / play / level from title, description and channel."""
    text = f"{r.title} {r.description} {r.channel}"
    official = (bool(OFFICIAL.match(r.channel)) or r.category == "broadcast") and not re.search(
        r"practice|court[- ]level", r.title, re.I)
    return dict(camera="broadcast" if official else guess("camera", text), surface=guess("surface", text),
                play=guess("play", text),
                level=level if (level := guess("level", text)) in ("junior", "college") or not (
                    official or r.category in ("pro_practice", "futures")) else "pro")


def out_of_scope(r: pd.Series) -> str:
    if r.play == "doubles":
        return "scope: doubles"
    return "" if r.camera in SCOPE_CAMERAS else "scope: angle"


def load() -> pd.DataFrame:
    if MANIFEST.exists():
        return pd.read_csv(MANIFEST, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=COLUMNS)


def save(df: pd.DataFrame) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    df = df.reindex(columns=COLUMNS, fill_value="")
    tmp = MANIFEST.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(MANIFEST)


def search(per_query: int, sleep: float) -> None:
    df = load()
    seen = set(df.video_id)
    rows = []
    for category, query in QUERIES:
        out = subprocess.run(YTDLP + ["--flat-playlist", "--dump-json", f"ytsearch{per_query}:{query}"],
                             capture_output=True, text=True)
        new = 0
        for line in out.stdout.splitlines():
            e = json.loads(line)
            if e.get("id") in seen or len(e.get("id", "")) != 11:
                continue
            seen.add(e["id"])
            rows.append(dict(
                video_id=e["id"], url=f"https://www.youtube.com/watch?v={e['id']}", title=e.get("title", ""),
                channel=e.get("channel") or "", channel_id=e.get("channel_id") or "",
                duration_s=str(int(e["duration"])) if e.get("duration") else "",
                view_count=str(e.get("view_count") or ""), query=query, category=category,
                selected="False", status="candidate",
                description=(e.get("description") or "").replace("\n", " ")[:300]))
            rows[-1] |= tag(pd.Series(rows[-1]))
            new += 1
        print(f"{new:3} new  {category:13} {query}" + (f"  [{out.stderr.strip()[-120:]}]" if out.returncode else ""))
        time.sleep(sleep)
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    save(df)
    print(f"{len(rows)} new candidates, {len(df)} in the manifest")


def eligible(r: pd.Series) -> bool:
    dur = int(r.duration_s) if r.duration_s else 0
    return (300 <= dur <= 6 * 3600 and r.status in ("candidate", "selected", "failed") and not out_of_scope(r)
            and not REJECT.search(r.title) and not REJECT.search(r.channel))


def select(n: int, per_channel: int) -> None:
    """Greedy: each pick maximises how rare its category / camera / surface / play / level still are.

    Rows already downloaded (done / partial) stay selected unless out of scope; everything else is
    re-tagged (unless tagged by eye) and re-picked.
    """
    df = load()
    for i, r in df[df.status.isin(["selected", "done", "partial"])].iterrows():
        if why := out_of_scope(r):
            df.loc[i, ["selected", "status", "notes"]] = ["False", "rejected", f"{why}; {r.notes}".strip("; ")]
    open_ = df.status.isin(["candidate", "selected"]) | (df.status.eq("failed") & ~df.notes.str.contains("not available"))
    guessed = open_ & (df.checked != "eye")
    df.loc[guessed, ["camera", "surface", "play", "level"]] = pd.DataFrame(
        [tag(r) for _, r in df[guessed].iterrows()], index=df[guessed].index)
    df.loc[open_, ["selected", "status"]] = ["False", "candidate"]
    df.loc[~open_, "selected"] = df.loc[~open_, "status"].isin(["done", "partial"]).map(str)
    chosen = df[df.selected == "True"]
    counts = {k: Counter(chosen[k]) for k in ("category", "camera", "surface", "play", "level")}
    per_chan = Counter(chosen.channel_id)
    pool = df[open_ & df.apply(eligible, axis=1)]
    while len(chosen) < n:
        pool = pool[pool.channel_id.map(lambda c: per_chan[c] < per_channel)]
        if counts["camera"]["broadcast"] >= MAX_BROADCAST:
            pool = pool[pool.camera != "broadcast"]
        if pool.empty:
            break
        score = sum(pool[k].map(lambda v, k=k: (0.3 if v == "unknown" else 1.0) / (1 + counts[k][v]))
                    for k in ("camera", "surface", "play", "level"))
        score += 3.0 * pool.category.map(lambda c: WEIGHT.get(c, 1) / (1 + counts["category"][c]))
        score += 0.5 * ((pool.camera == "unknown") & (pool.title + " " + pool.description).str.contains(BASELINE_HINT))
        i = score.idxmax()
        r = df.loc[i]
        df.loc[i, ["selected", "status"]] = ["True", "selected"]
        for k in counts:
            counts[k][r[k]] += 1
        per_chan[r.channel_id] += 1
        chosen = df[df.selected == "True"]
        pool = pool.drop(i)
    save(df)
    print(f"{(df.selected == 'True').sum()} selected")


def starts(duration: float, k: int, length: int) -> list[int]:
    """k non-overlapping windows spread over the middle 80% of the video."""
    lo, span = 0.1 * duration, 0.8 * duration
    k = max(0, min(k, int(span // length)))
    return [int(lo + (i + 0.5) * span / k - length / 2) for i in range(k)]


def dir_gb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*.mp4") if not p.name.startswith("._")) / 1e9


def fetch(vid: str, k: int, length: int, sleep: float) -> dict:
    """Full metadata, then each missing segment. Returns the manifest fields to update."""
    out_dir = ROOT / vid
    out_dir.mkdir(parents=True, exist_ok=True)
    info = subprocess.run(YTDLP + GENTLE[:2] + ["-J", f"https://www.youtube.com/watch?v={vid}"],
                          capture_output=True, text=True)
    if info.returncode:
        return dict(status="failed", notes=info.stderr.strip().splitlines()[-1][:200] if info.stderr else "no info")
    d = json.loads(info.stdout)
    meta = dict(upload_date=d.get("upload_date") or "", width=str(d.get("width") or ""),
                height=str(d.get("height") or ""), fps=str(d.get("fps") or ""),
                license=d.get("license") or "youtube", duration_s=str(int(d.get("duration") or 0)))
    if d.get("is_live") or not d.get("duration"):
        return meta | dict(status="failed", notes="live or no duration")
    segs, errors = [], []
    with tempfile.NamedTemporaryFile("w", suffix=".info.json", delete=False) as f:
        json.dump(d, f)
    for i, s in enumerate(starts(float(d["duration"]), k, length)):
        path = out_dir / f"seg{i}_{s:05d}.mp4"
        if not path.exists():
            for attempt in range(2):   # ffmpeg sometimes gets a 403 mid-stream; a retry usually works
                r = subprocess.run(YTDLP + GENTLE + ["--load-info-json", f.name, "-f", FORMAT, "--merge-output-format", "mp4",
                                            "--download-sections", f"*{s}-{s + length}", "-o", str(path)],
                                   capture_output=True, text=True)
                if path.exists():
                    break
                time.sleep(10 * (attempt + 1))
            if not path.exists():
                errors.append(f"seg{i}: " + (r.stderr.strip().splitlines() or ["?"])[-1][:120])
                continue
            time.sleep(sleep)
        segs.append(path.name)
    Path(f.name).unlink()
    status = "done" if segs and not errors else ("partial" if segs else "failed")
    return meta | dict(segments=";".join(segs), status=status, notes="; ".join(errors))


def download(k: int, length: int, workers: int, sleep: float, max_gb: float) -> None:
    df = load()
    todo = df[(df.selected == "True") & df.status.isin(["selected", "partial", "failed"])].video_id.tolist()
    print(f"{len(todo)} videos to fetch, {workers} at a time")
    with ThreadPoolExecutor(workers) as pool:
        futures = {}
        for vid in todo:
            futures[pool.submit(fetch, vid, k, length, sleep)] = vid
        for n, fut in enumerate(as_completed(futures), 1):
            vid = futures[fut]
            try:
                fields = fut.result()
            except Exception as e:  # keep going; the row says what broke
                fields = dict(status="failed", notes=repr(e)[:200])
            i = df.index[df.video_id == vid][0]
            for c, v in fields.items():
                df.loc[i, c] = v
            save(df)
            gb = dir_gb(ROOT)
            print(f"[{n}/{len(todo)}] {vid} {fields['status']:7} {fields.get('segments', '')} "
                  f"{fields.get('notes', '')[:80]}  ({gb:.2f} GB)", flush=True)
            if re.search(r"not a bot|429|Sign in", fields.get("notes", "")):
                print("stopping: YouTube is asking to sign in / rate limiting; wait and rerun, or pass --cookies-from-browser")
                for f in futures:
                    f.cancel()
                break
            if gb > max_gb:
                print(f"stopping: {gb:.1f} GB > {max_gb} GB")
                for f in futures:
                    f.cancel()
                break


def reject(ids: list[str], why: str) -> None:
    """Mark videos as unusable (after looking at titles or frames); `select` then refills."""
    df = load()
    hit = df.video_id.isin(ids)
    df.loc[hit, ["selected", "status", "notes"]] = ["False", "rejected", why]
    save(df)
    print(f"{hit.sum()} rejected" + (f"; unknown ids {set(ids) - set(df.video_id)}" if hit.sum() < len(ids) else ""))


def middle_frame(path: Path, width: int = 480) -> Image.Image:
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", "30", "-i", str(path), "-frames:v", "1",
                          "-vf", f"scale={width}:-2", "-f", "image2pipe", "-c:v", "png", "-"], capture_output=True)
    return Image.open(io.BytesIO(out.stdout)).convert("RGB")


def sheet(n: int, out: Path, seed: int, ids: list[str] | None = None, cols: int = 6) -> None:
    """Grid of middle frames, one random segment per video, captioned with the video id."""
    df = load()
    rows = df[df.video_id.isin(ids)] if ids else df[(df.selected == "True") & (df.segments != "")]
    rows = rows if ids else rows.sample(min(n, len(rows)), random_state=seed)
    rng = random.Random(seed)
    tiles = []
    for r in rows.itertuples():
        segs = [s for s in r.segments.split(";") if s]
        if not segs:
            continue
        im = middle_frame(ROOT / r.video_id / rng.choice(segs))
        im = im.resize((480, 270))
        ImageDraw.Draw(im).rectangle((0, 0, 480, 18), fill="black")
        ImageDraw.Draw(im).text((4, 3), f"{r.video_id} {r.category} {r.camera}", fill="white")
        tiles.append(im)
    grid = Image.new("RGB", (480 * cols, 270 * -(-len(tiles) // cols)))
    for i, im in enumerate(tiles):
        grid.paste(im, (480 * (i % cols), 270 * (i // cols)))
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out, quality=85)
    print(f"{len(tiles)} frames -> {out}")


def looked(path: Path) -> None:
    """Apply tags read off the frames. Lines: `id camera surface light free note`, or `id REJECT why`."""
    df = load()
    for line in Path(path).read_text().splitlines():
        vid, *rest = line.split()
        i = df.index[df.video_id == vid]
        if not rest or i.empty:
            continue
        if rest[0] == "REJECT":
            df.loc[i, ["selected", "status", "notes"]] = ["False", "rejected", " ".join(rest[1:])]
            continue
        camera, surface, light, note = rest[0], rest[1], rest[2], " ".join(rest[3:])
        df.loc[i, ["camera", "surface", "light", "notes", "checked"]] = [camera, surface, light, note, "eye"]
        if re.search(r"doubles", note):
            df.loc[i, "play"] = "doubles"
        r = df.loc[i[0]]
        if r.status == "rejected" and (why := out_of_scope(r)):
            df.loc[i, "notes"] = f"{why}; {note}".strip("; ")
    save(df)
    print(f"{(df.checked == 'eye').sum()} rows checked by eye")


def summary() -> None:
    df = load()
    sel = df[df.selected == "True"]
    print(f"candidates {len(df)}  selected {len(sel)}  status {dict(Counter(df.status))}  "
          f"checked by eye {(sel.checked == 'eye').sum()}")
    for k in ("category", "camera", "surface", "light", "play", "level"):
        print(f"\n{k}: " + ", ".join(f"{v or '?'} {c}" for v, c in Counter(sel[k]).most_common()))
    print("\n" + pd.crosstab(sel.camera, sel.play, margins=True).to_string())
    files = [ROOT / v / s for v, segs in zip(sel.video_id, sel.segments) for s in segs.split(";") if s]
    gb = sum(f.stat().st_size for f in files if f.exists()) / 1e9
    print(f"\n{len(files)} segments from {(sel.segments != '').sum()} videos, ~{len(files)} min, {gb:.2f} GB "
          f"({dir_gb(ROOT):.2f} GB on disk incl. rejected), {sel.channel_id.nunique()} channels")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("--per-query", type=int, default=20)
    s.add_argument("--sleep", type=float, default=2.0)
    s = sub.add_parser("select")
    s.add_argument("--n", type=int, default=150)
    s.add_argument("--per-channel", type=int, default=4)
    s = sub.add_parser("download")
    s.add_argument("--segments", type=int, default=3)
    s.add_argument("--length", type=int, default=60)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--sleep", type=float, default=3.0)
    s.add_argument("--max-gb", type=float, default=MAX_GB)
    s.add_argument("--cookies-from-browser", help="e.g. chrome; only when YouTube asks to sign in")
    s = sub.add_parser("reject")
    s.add_argument("ids", nargs="+")
    s.add_argument("--why", required=True)
    s = sub.add_parser("looked")
    s.add_argument("file", type=Path)
    s = sub.add_parser("sheet")
    s.add_argument("--n", type=int, default=24)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--ids", nargs="*")
    s.add_argument("--out", type=Path, default=SCRATCH / "youtube_contact_sheet.jpg")
    sub.add_parser("summary")
    a = p.parse_args()
    if a.cmd == "search":
        search(a.per_query, a.sleep)
    elif a.cmd == "select":
        select(a.n, a.per_channel)
    elif a.cmd == "download":
        if a.cookies_from_browser:
            YTDLP.extend(["--cookies-from-browser", a.cookies_from_browser])
        download(a.segments, a.length, a.workers, a.sleep, a.max_gb)
    elif a.cmd == "sheet":
        sheet(a.n, a.out, a.seed, a.ids)
    elif a.cmd == "looked":
        looked(a.file)
    elif a.cmd == "reject":
        reject(a.ids, a.why)
    else:
        summary()


if __name__ == "__main__":
    main()
