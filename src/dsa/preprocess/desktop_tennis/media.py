"""Normalize phone rotation, HDR and variable frame timing before analysis."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

from ..media import source_info


def probe(path: Path) -> dict:
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]))


def normalize(video: Path, out: Path, width: int = 1280) -> dict:
    """CFR analysis master. Original elapsed times map to master times, not frame IDs."""
    out.mkdir(parents=True, exist_ok=True)
    stat = video.stat()
    signature = {"path": str(video.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                 "width": width, "fps": 30, "normalization_version": 1}
    metadata = out / "source.json"
    master = out / "analysis.mp4"
    if metadata.exists() and master.exists():
        saved = json.loads(metadata.read_text())
        if saved["signature"] == signature:
            return saved
        raise ValueError(f"Source/settings changed: choose a new output directory instead of reusing {out}")
    original = probe(video)
    stream = next(s for s in original["streams"] if s["codec_type"] == "video")
    rotation = next((s["rotation"] for s in stream.get("side_data_list", []) if "rotation" in s), 0)
    if abs(rotation) not in (0, 180):
        raise ValueError("This landscape recipe supports only 0/180 degree phone rotation")
    height = round(width * stream["height"] / stream["width"] / 2) * 2
    hdr = stream.get("color_transfer") in {"arib-std-b67", "smpte2084"}
    command = ["ffmpeg", "-v", "error", "-y", "-noautorotate"]
    if platform.system() == "Darwin" and hdr:
        command += ["-hwaccel", "videotoolbox", "-hwaccel_output_format", "videotoolbox_vld"]
        filters = [f"scale_vt=w={width}:h={height}:color_matrix=bt709:color_primaries=bt709:color_transfer=bt709",
                   "hwdownload", "format=p010le", "format=yuv420p"]
    elif hdr:
        filters = ["zscale=t=linear:npl=100", "format=gbrpf32le", "zscale=p=bt709",
                   "tonemap=tonemap=mobius", "zscale=t=bt709:m=bt709:r=tv",
                   f"scale={width}:{height}", "format=yuv420p"]
    else:
        filters = [f"scale={width}:{height}", "format=yuv420p"]
    if abs(rotation) == 180:
        filters += ["hflip", "vflip"]
    filters += ["setpts=PTS-STARTPTS", "fps=30"]
    temporary = out / "analysis.partial.mp4"
    command += ["-i", str(video), "-map", "0:v:0", "-map", "0:a:0?", "-vf", ",".join(filters),
                "-af", "asetpts=PTS-STARTPTS,aresample=async=1:first_pts=0",
                "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
                "-metadata:s:v:0", "rotate=0"]
    command += (["-c:v", "h264_videotoolbox", "-b:v", "4500k"] if platform.system() == "Darwin"
                else ["-c:v", "libx264", "-crf", "20", "-preset", "fast"])
    command += ["-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(temporary)]
    print(f"Normalizing {video.name} to upright SDR 30 fps", flush=True)
    subprocess.run(command, check=True)
    info = source_info(temporary)
    duration = float(stream["duration"])
    if info["fps"] != 30 or abs(info["frames"] / 30 - duration) > 0.1:
        raise ValueError("Normalization changed the timeline unexpectedly")
    with video.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    temporary.replace(master)
    result = {"signature": signature, "original_video": str(video.resolve()),
              "original_sha256": digest, "original_probe": original,
              "video": str(master.resolve()), **info,
              "timeline": "30 fps analysis frames; time zero matches original video time zero. "
                          "Original phone frame IDs are not preserved."}
    metadata.write_text(json.dumps(result, indent=2))
    print(f"Ready: {master}", flush=True)
    return result
