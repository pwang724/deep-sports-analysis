"""Video I/O and review artifacts shared by view-selection pipelines."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from .manifest import validate_manifest

def source_info(video: Path) -> dict:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,width,height,duration", "-of", "json", str(video),
    ])
    info = json.loads(raw)["streams"][0]
    num, den = map(int, info["avg_frame_rate"].split("/"))
    return {"fps": num / den, "fps_fraction": info["avg_frame_rate"],
            "frames": int(info["nb_frames"]), "width": info["width"], "height": info["height"]}


def make_proxy(video: Path, out: Path, info: dict) -> Path:
    """Small H.264 copy at source frame rate, avoiding AV1 decoder differences."""
    proxy = out / "proxy.mp4"
    metadata = out / "proxy-source.json"
    stat = video.stat()
    key = {"path": str(video.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if not proxy.exists() or not metadata.exists() or json.loads(metadata.read_text()) != key:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", "scale=480:270",
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23", "-an", str(proxy)], check=True)
        metadata.write_text(json.dumps(key))
    actual = source_info(proxy)
    if actual["frames"] != info["frames"] or abs(actual["fps"] - info["fps"]) > 1e-6:
        raise ValueError("Proxy does not preserve source frame indices; constant frame rate video required")
    return proxy


def extract_frames(video: Path, indices: list[int]) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(video))
    images = []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = cap.read()
            if not ok:
                raise ValueError(f"Could not read frame {index} from {video}")
            images.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    finally:
        cap.release()
    return images


def contact_sheet(items: list[tuple[Image.Image, str]], path: Path, columns: int = 4) -> None:
    width, height = 320, 204
    sheet = Image.new("RGB", (columns * width, ((len(items) + columns - 1) // columns) * height), "#111827")
    draw = ImageDraw.Draw(sheet)
    for i, (image, label) in enumerate(items):
        x, y = (i % columns) * width, (i // columns) * height
        thumb = image.copy()
        thumb.thumbnail((width, 180))
        sheet.paste(thumb, (x, y))
        draw.text((x + 6, y + 183), label, fill="white")
    sheet.save(path)


def selection_expression(shots: list[dict]) -> str:
    """Coalesce for playback only; balance sums to avoid ffmpeg parser depth limits."""
    ranges = []
    for shot in shots:
        a, b = shot["start_frame"], shot["end_frame"]
        if ranges and ranges[-1][1] == a:
            ranges[-1] = (ranges[-1][0], b)
        else:
            ranges.append((a, b))
    terms = [f"between(n,{a},{b - 1})" for a, b in ranges]
    while len(terms) > 1:
        terms = [f"({terms[i]}+{terms[i + 1]})" if i + 1 < len(terms) else terms[i]
                 for i in range(0, len(terms), 2)]
    return terms[0] if terms else "0"


def export_reviews(manifest: dict, out: Path, width: int = 960) -> None:
    """Silent review videos; tracking must use source + manifest, not these joins."""
    validate_manifest(manifest)
    out.mkdir(parents=True, exist_ok=True)
    for name, labels in (("kept", {"keep"}), ("unkept", {"discard", "review"})):
        shots = [s for s in manifest["shots"] if s["label"] in labels]
        if not shots:
            (out / f"{name}.mp4").unlink(missing_ok=True)
            continue
        expression = selection_expression(shots)
        rate = manifest.get("fps_fraction", str(manifest["fps"]))
        filters = f"select='{expression}',setpts=N/(({rate})*TB),scale={width}:-2"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", manifest["video"], "-vf", filters,
                        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                        "-movflags", "+faststart", str(out / f"{name}.mp4")], check=True)
