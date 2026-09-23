"""Burn a scoreboard into each separately exported eleven or match segment."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .annotations import edit_timeline, score_timeline, validate_log


def font(size: int):
    for name in ("/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(name).exists():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def scoreboard(players: list[str], segment: dict, score: dict) -> Image.Image:
    match = segment["kind"] == "match"
    width, height = (430, 174)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), 12, fill=(12, 21, 30, 235))
    title = "MATCH PLAY" if match else segment["id"].replace("_", " ").upper()
    draw.text((16, 10), title, fill="#b8e660", font=font(18))
    if match:
        for x, name in ((247, "SETS"), (307, "GAMES"), (371, "PTS")):
            draw.text((x, 14), name, fill="#a8b4c0", font=font(11))
    for i, player in enumerate(players):
        y = 40 + i * 42
        draw.text((16, y), player[:18], fill="white", font=font(24))
        if match:
            draw.text((257, y), str(score["sets"][i]), fill="#a8b4c0", font=font(24))
            draw.text((323, y), str(score["games"][i]), fill="white", font=font(24))
        draw.text((370, y), str(score["points"][i]), fill="#b8e660", font=font(26))
    status = score["status"]
    if segment["status"] != "verified":
        status = "DRAFT · boundaries / score need review"
    if not status:
        status = "Standard advantage" if match else "First to 11 · win by two"
    draw.text((16, 143), status, fill="#a8b4c0", font=font(13))
    # Keep the far court visible: the scoreboard stays left of its singles line.
    return image.resize((300, 122), Image.Resampling.LANCZOS)


def render_segment(log: dict, source: dict, segment: dict, out: Path, draft: bool = False) -> Path:
    validate_log(log, source)
    if segment["status"] != "verified" and not draft:
        raise ValueError(f"{segment['id']} is unverified; review it or explicitly use --draft")
    prefix = "draft_" if segment["status"] != "verified" else ""
    name = prefix + segment["id"]
    assets = out / "overlays" / name
    assets.mkdir(parents=True, exist_ok=True)
    fps = source["fps"]
    spans = edit_timeline(segment, fps)
    if not spans:
        raise ValueError(f"{segment['id']} has no retained point clips; annotate points before exporting")
    listing = ["ffconcat version 1.0"]
    for i, span in enumerate(spans):
        first, last = span["source_start_frame"], span["source_end_frame"]
        duration = (last - first) / fps
        png = assets / f"score_{i:05d}.png"
        clip = assets / f"clip_{i:05d}.mkv"
        scoreboard(log["players"], segment, span["score"]).save(png)
        command = ["ffmpeg", "-v", "error", "-y", "-ss", str(first / fps),
                   "-i", source["video"], "-loop", "1", "-framerate", str(fps), "-i", str(png),
                   "-filter_complex", "[0:v][1:v]overlay=20:20:shortest=1[v]",
                   "-map", "[v]", "-map", "0:a:0?", "-t", str(duration), "-r", str(fps),
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                   "-c:a", "pcm_s16le", "-ar", "48000", "-f", "matroska", str(clip)]
        # PCM intermediate audio avoids accumulating AAC encoder delay at every cut.
        subprocess.run(command, check=True)
        listing.append(f"file '{clip.name}'")
        print(f"{name}: retained span {i + 1}/{len(spans)}", flush=True)
    concat = assets / "clips.ffconcat"
    concat.write_text("\n".join(listing) + "\n")
    destination = out / f"{name}.mp4"
    temporary = out / f"{name}.partial.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
                    str(temporary)], check=True)
    temporary.replace(destination)
    destination.with_suffix(".json").write_text(json.dumps({
        "segment": segment, "players": log["players"], "source_video": source["original_video"],
        "analysis_video": source["video"], "fps": fps, "edit_timeline": spans,
        "score_timeline": score_timeline(segment)}, indent=2))
    return destination
