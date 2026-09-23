"""Unscored activity edits, independent of point winners and scoring modes.

The edit decision list uses half-open source frame ranges on the normalized
master. It deliberately makes no claim that a clip corresponds to one point.
"""
from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

from ..manifest import validate_manifest
from .media import probe


def validate_cuts(edit: dict, source: dict) -> None:
    if edit.get("version") != 1 or edit.get("source_sha256") != source["original_sha256"]:
        raise ValueError("Cut list does not match this source/version")
    if edit.get("fps") != source["fps"]:
        raise ValueError("Cut list must use the analysis master's frame rate")
    previous = 0
    if not edit.get("clips"):
        raise ValueError("Cut list has no retained activity")
    for clip in edit["clips"]:
        a, b = clip["start_frame"], clip["end_frame"]
        if type(a) is not int or type(b) is not int or not previous <= a < b <= source["frames"]:
            raise ValueError("Cuts must be ordered, non-overlapping source frame ranges")
        previous = b


def cut_timeline(edit: dict, source: dict) -> list[dict]:
    validate_cuts(edit, source)
    cursor, result = 0, []
    for i, c in enumerate(edit["clips"]):
        count = c["end_frame"] - c["start_frame"]
        result.append({"clip_index": i, "source_start_frame": c["start_frame"],
                       "source_end_frame": c["end_frame"], "output_start_frame": cursor,
                       "output_end_frame": cursor + count})
        cursor += count
    return result


def cut_manifest(edit: dict, source: dict) -> dict:
    validate_cuts(edit, source)
    shots, cursor = [], 0
    for i, c in enumerate(edit["clips"]):
        a, b = c["start_frame"], c["end_frame"]
        if a > cursor:
            shots.append({"start_frame": cursor, "end_frame": a, "label": "discard"})
        shots.append({"start_frame": a, "end_frame": b, "label": "keep", "activity_clip": i})
        cursor = b
    if cursor < source["frames"]:
        shots.append({"start_frame": cursor, "end_frame": source["frames"], "label": "discard"})
    manifest = {k: source[k] for k in ("video", "fps", "fps_fraction", "frames", "width", "height")}
    manifest.update(version=1, pipeline="desktop_tennis_activity", shots=shots,
                    original_sha256=source["original_sha256"],
                    description="Unscored activity cuts; boundaries are not point or game labels.")
    validate_manifest(manifest)
    return manifest


def load_cuts(out: Path) -> tuple[dict, dict]:
    source = json.loads((out / "source.json").read_text())
    path = out / "cuts.json"
    if not path.exists():
        for profile in Path(__file__).with_name("profiles").glob("*_cuts.json"):
            candidate = json.loads(profile.read_text())
            if candidate.get("source_sha256") == source["original_sha256"]:
                validate_cuts(candidate, source)
                path.write_text(json.dumps(candidate, indent=2) + "\n")
                break
        else:
            raise ValueError("No source-matched cut list. Create cuts.json with retained frame ranges first.")
    edit = json.loads(path.read_text())
    validate_cuts(edit, source)
    return edit, source


def export_play(out: Path, destination: Path | None = None, *, encoder: str | None = None) -> Path:
    """Render every retained interval with original audio and no overlays.

    Each intermediate is frame-exact H.264 + PCM. Encoding AAC once after
    concatenation prevents an encoder-delay error from accumulating per cut.
    """
    edit, source = load_cuts(out)
    destination = destination or out / "tennis_play.mp4"
    if destination.resolve() in {Path(source[k]).resolve() for k in ("video", "original_video")}:
        raise ValueError("Export destination must not overwrite a source video")
    destination.parent.mkdir(parents=True, exist_ok=True)
    spans = cut_timeline(edit, source)
    assets = out / "activity_render"
    assets.mkdir(exist_ok=True)
    fps = source["fps"]
    encoder = encoder or ("h264_videotoolbox" if platform.system() == "Darwin" else "libx264")
    encoding = (["-c:v", encoder, "-b:v", "5000k"] if encoder == "h264_videotoolbox"
                else ["-c:v", encoder, "-preset", "veryfast", "-crf", "20"])
    listing = ["ffconcat version 1.0"]
    for i, s in enumerate(spans):
        clip = assets / f"clip_{i:04d}.mkv"
        a, b = s["source_start_frame"], s["source_end_frame"]
        duration = (b - a) / fps
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a / fps), "-i", source["video"],
                        "-map", "0:v:0", "-map", "0:a:0?", "-t", str(duration), "-r", str(fps),
                        *encoding, "-pix_fmt", "yuv420p", "-c:a", "pcm_s16le", "-ar", "48000",
                        str(clip)], check=True)
        listing.append(f"file '{clip.name}'\nduration {duration:.9f}")
        if i % 10 == 0 or i + 1 == len(spans):
            print(f"{out.name}: rendered {i + 1}/{len(spans)} activity clips", flush=True)
    concat = assets / "clips.ffconcat"
    concat.write_text("\n".join(listing) + "\n")
    temp = destination.with_name(destination.stem + ".partial.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(temp)],
                   check=True)
    result = probe(temp)
    video = next(s for s in result["streams"] if s["codec_type"] == "video")
    expected = spans[-1]["output_end_frame"]
    if int(video["nb_frames"]) != expected:
        raise ValueError(f"Export frame count {video['nb_frames']} != cut list {expected}")
    duration = expected / fps
    audio = next((s for s in result["streams"] if s["codec_type"] == "audio"), None)
    if audio and abs(float(audio["duration"]) - duration) > .1:
        raise ValueError("Export audio duration drift exceeds 100 ms")
    temp.replace(destination)
    destination.with_suffix(".json").write_text(json.dumps({
        "source_sha256": source["original_sha256"], "source_video": source["original_video"],
        "analysis_video": source["video"], "fps": fps, "frames": expected,
        "duration_seconds": duration, "source_duration_seconds": source["frames"] / fps,
        "clip_count": len(spans), "scoring": "none", "edit_timeline": spans,
        "method": edit.get("method"), "review": edit.get("review")}, indent=2) + "\n")
    (out / "play_shots.json").write_text(json.dumps(cut_manifest(edit, source), indent=2) + "\n")
    print(f"Ready: {destination} ({duration:.1f}s, {len(spans)} clips)", flush=True)
    return destination
