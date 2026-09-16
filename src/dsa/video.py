"""Thin ffmpeg wrappers: probe a codec, transcode to H.264, write and join H.264 mp4s."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

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


class H264Writer:
    """Write BGR frames to an H.264 mp4 through ffmpeg.

    Replaces cv2.VideoWriter, whose mp4v output is ~4x larger at the same
    quality. CRF 23 is visually lossless enough for annotated review video.
    """

    def __init__(self, path: str | Path, fps: float, size: tuple[int, int], crf: int = 23, preset: str = "fast"):
        w, h = size
        self.proc = subprocess.Popen(
            ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.6f}",
             "-i", "-", "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE,
        )

    def write(self, bgr: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(bgr).tobytes())

    def release(self) -> None:
        self.proc.stdin.close()
        if self.proc.wait() != 0:
            raise RuntimeError("ffmpeg failed while writing video")
