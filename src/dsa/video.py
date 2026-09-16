"""Thin ffmpeg wrappers: probe a codec, transcode to H.264, join mp4 segments."""
from __future__ import annotations

import subprocess
from pathlib import Path

# Codecs the OpenCV Linux wheel decodes with its bundled ffmpeg. AV1 is not one of them.
CV2_DECODABLE = {"h264", "hevc", "mpeg4", "vp9"}


def codec_of(path: str | Path) -> str:
    """Name of the first video stream's codec, e.g. 'h264' or 'av1'."""
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return out.strip()


def transcode_h264(src: str | Path, dst: str | Path, crf: int = 18, preset: str = "veryfast") -> None:
    """Re-encode `src` to H.264 without audio, keeping resolution and frame rate."""
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-c:v", "libx264", "-preset", preset,
                    "-crf", str(crf), "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dst)], check=True)


def concat_videos(videos: list[Path], out: Path) -> None:
    """Join mp4 segments without re-encoding using ffmpeg's concat demuxer."""
    listing = out.with_suffix(".txt")
    listing.write_text("".join(f"file '{v.resolve()}'\n" for v in videos))
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", str(out)], check=True)
    finally:
        listing.unlink()
