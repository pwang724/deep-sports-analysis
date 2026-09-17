"""The only contract between a view filter and downstream tracking.

Intervals cover the original video exactly once, in source frames [start, end).
Keep neighboring camera shots separate even when their labels are identical.
Extra pipeline metadata is allowed; it does not change this contract.
"""
import math
from pathlib import Path


def validate_manifest(manifest: dict, video: str | Path | None = None) -> None:
    if not isinstance(manifest.get("video"), str) or not manifest["video"]:
        raise ValueError("Manifest needs a source video path")
    if video is not None and Path(manifest["video"]).name != Path(video).name:
        raise ValueError("Shot manifest belongs to a different source video")
    fps, frames = manifest.get("fps"), manifest.get("frames")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Manifest fps must be finite and positive")
    if type(frames) is not int or frames <= 0:
        raise ValueError("Manifest frames must be a positive integer")
    shots = manifest.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("Manifest needs a complete shot partition")
    previous_end = 0
    for shot in shots:
        start, end = shot.get("start_frame"), shot.get("end_frame")
        if (type(start) is not int or type(end) is not int
                or start != previous_end or not start < end <= frames):
            raise ValueError("Shot intervals must be contiguous, non-overlapping source frame ranges")
        if shot.get("label") not in {"keep", "discard", "review"}:
            raise ValueError("Unknown shot label")
        previous_end = end
    if previous_end != frames:
        raise ValueError("Shot intervals must cover the complete source video")
