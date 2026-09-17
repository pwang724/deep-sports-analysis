"""Split a video into segments for parallel tracking and merge the results back.

Pure functions, no models: this is the part of the cloud fan-out that can be
unit tested locally.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dsa.preprocess.manifest import validate_manifest

# Track IDs from segment i are offset by i * TRACK_ID_STRIDE so they never collide.
TRACK_ID_STRIDE = 100_000


@dataclass(frozen=True)
class Segment:
    index: int
    start: float
    duration: float
    start_frame: int | None = None
    end_frame: int | None = None

    @property
    def name(self) -> str:
        return f"seg{self.index:04d}"

    @property
    def track_id_offset(self) -> int:
        return self.index * TRACK_ID_STRIDE


def plan_segments(start: float, duration: float, segment_len: float) -> list[Segment]:
    """Cover [start, start + duration) with consecutive segments of at most `segment_len` seconds."""
    if duration <= 0 or segment_len <= 0:
        raise ValueError("duration and segment_len must be positive")
    segments = []
    t = start
    i = 0
    while t < start + duration - 1e-9:
        length = min(segment_len, start + duration - t)
        segments.append(Segment(i, round(t, 3), round(length, 3)))
        t += length
        i += 1
    return segments


def plan_kept_shots(manifest: dict, video: str | Path) -> list[Segment]:
    """One fresh tracker per kept camera shot; preserve exact source frame bounds."""
    validate_manifest(manifest, video)
    fps = manifest["fps"]
    segments = []
    for shot in manifest["shots"]:
        start, end = shot["start_frame"], shot["end_frame"]
        if shot["label"] == "keep":
            segments.append(Segment(len(segments), start / fps, (end - start) / fps, start, end))
    if not segments:
        raise ValueError("Manifest has no kept shots; review the references and shot decisions")
    return segments


def merge_tables(segment_dirs: list[Path], out_dir: Path) -> dict:
    """Concatenate per-segment joints tables and summaries into `out_dir`. Returns the merged summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_parquet(d / "joints.parquet") for d in segment_dirs]
    df = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else pd.DataFrame()
    if len(df):
        df = df.sort_values(["frame", "track_id"], ignore_index=True)
    df.to_parquet(out_dir / "joints.parquet", index=False)

    parts = [json.loads((d / "summary.json").read_text()) for d in segment_dirs]
    total = sum(p["frames_processed"] for p in parts)
    summary = {
        "video": parts[0]["video"],
        "start": min(p["start"] for p in parts),
        "duration": sum(p["duration"] for p in parts),
        "fps": parts[0]["fps"],
        "config": parts[0]["config"],
        "segments": len(parts),
        "frames_processed": total,
        "rows": int(len(df)),
        "tracks": int(df.track_id.nunique()) if len(df) else 0,
        "sec_per_frame": {
            k: sum(p["sec_per_frame"][k] * p["frames_processed"] for p in parts) / max(total, 1)
            for k in parts[0]["sec_per_frame"]
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
