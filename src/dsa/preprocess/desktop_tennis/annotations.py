"""Editable source-time point log and its adapter to the shared shot contract."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from ..manifest import validate_manifest
from .scoring import Score


def validate_log(log: dict, source: dict) -> None:
    if log.get("version") != 1 or log.get("source_sha256") != source["original_sha256"]:
        raise ValueError("Point log does not match this source/version")
    players = log.get("players", [])
    if len(players) != 2 or any(not isinstance(p, str) or not p.strip() for p in players):
        raise ValueError("Exactly two player names are required")
    duration = source["frames"] / source["fps"]
    previous = 0
    identifiers = set()
    for segment in log.get("segments", []):
        name = segment.get("id", "")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) or name in identifiers:
            raise ValueError("Segment IDs must be unique lowercase filename-safe names")
        identifiers.add(name)
        start, end = segment["start"], segment["end"]
        if not all(type(t) in (int, float) and math.isfinite(t) for t in (start, end)):
            raise ValueError("Segment times must be finite numbers")
        if not 0 <= previous <= start < end <= duration:
            raise ValueError("Segments must be ordered, non-overlapping, and inside the video")
        if round(start * source["fps"]) >= round(end * source["fps"]):
            raise ValueError("Segments must contain at least one analysis frame")
        if segment.get("status") not in {"draft", "verified"}:
            raise ValueError("Segment status must be draft or verified")
        state = initial_score(segment)
        last = start
        for point in segment.get("points", []):
            t = point["time"]
            if type(t) not in (int, float) or not math.isfinite(t) or not last < t <= end:
                raise ValueError("Point times must increase strictly inside their segment")
            clips = point.get("clips", [])
            if not clips:
                raise ValueError("Each point needs retained clips, including missed first serves")
            clip_end = last
            for clip in clips:
                a, b = clip["start"], clip["end"]
                if not all(type(v) in (int, float) and math.isfinite(v) for v in (a, b)):
                    raise ValueError("Clip times must be finite numbers")
                if not clip_end <= a < b <= end or round(a * source["fps"]) >= round(b * source["fps"]):
                    raise ValueError("Point clips must be ordered, non-overlapping, nonempty source ranges")
                if clip.get("role") not in {"rally", "first_serve_fault", "double_fault"}:
                    raise ValueError("Clip role must be rally, first_serve_fault, or double_fault")
                clip_end = b
            if not clips[-1]["start"] < t <= clips[-1]["end"]:
                raise ValueError("Point result must fall in its final retained clip")
            if point.get("status") not in {"draft", "verified"}:
                raise ValueError("Point status must be draft or verified")
            winner = point.get("winner")
            if winner is not None and (type(winner) is not int or winner not in (0, 1)):
                raise ValueError("Point winner must be 0, 1, or null")
            if segment["status"] == "verified" and (point["status"] != "verified" or winner is None):
                raise ValueError("Verified segments require every point winner to be verified")
            state.award(winner if point["status"] == "verified" else None)
            last = clip_end
        if segment["status"] == "verified":
            if not segment.get("coverage_verified") or not state.known:
                raise ValueError("Verify the initial score and complete point coverage first")
            if not segment.get("points"):
                raise ValueError("A verified playing segment needs a point log")
        previous = end


def initial_score(segment: dict) -> Score:
    initial = segment.get("initial_score", {})
    return Score(segment["kind"], points=list(initial.get("points", [0, 0])),
                 games=list(initial.get("games", [0, 0])), sets=list(initial.get("sets", [0, 0])),
                 known=bool(initial.get("verified", False)))


def score_timeline(segment: dict) -> list[dict]:
    state = initial_score(segment)
    timeline = [{"time": segment["start"], **state.display()}]
    for point in segment.get("points", []):
        state.award(point.get("winner") if point.get("status") == "verified" else None)
        timeline.append({"time": point["time"], **state.display()})
    return timeline


def manifest_for(log: dict, source: dict) -> dict:
    """One tracker per retained point clip, with explicit review/discard gaps."""
    validate_log(log, source)
    fps, total = source["fps"], source["frames"]
    shots, cursor = [], 0
    for segment in log.get("segments", []):
        start, end = round(segment["start"] * fps), round(segment["end"] * fps)
        if start > cursor:
            shots.append({"start_frame": cursor, "end_frame": start, "label": "review"})
        cursor = start
        gap_label = "discard" if segment.get("coverage_verified") else "review"
        for i, point in enumerate(segment.get("points", [])):
            for clip in point["clips"]:
                a, b = round(clip["start"] * fps), round(clip["end"] * fps)
                if a > cursor:
                    shots.append({"start_frame": cursor, "end_frame": a, "label": gap_label})
                shots.append({"start_frame": a, "end_frame": b,
                              "label": "keep" if point["status"] == "verified" else "review",
                              "segment_id": segment["id"], "point_index": i, "kind": segment["kind"],
                              "role": clip["role"]})
                cursor = b
        if cursor < end:
            shots.append({"start_frame": cursor, "end_frame": end, "label": gap_label})
        cursor = end
    if cursor < total:
        shots.append({"start_frame": cursor, "end_frame": total, "label": "review"})
    manifest = {key: source[key] for key in ("video", "fps", "fps_fraction", "frames", "width", "height")}
    manifest.update(version=1, pipeline="desktop_tennis", original_video=source["original_video"],
                    original_sha256=source["original_sha256"], timeline=source["timeline"], shots=shots)
    validate_manifest(manifest)
    return manifest


def edit_timeline(segment: dict, fps: float) -> list[dict]:
    """Map each retained source span to output time; cuts never award a point."""
    state = initial_score(segment)
    cursor = 0
    spans = []
    for point_index, point in enumerate(segment.get("points", [])):
        result_frame = round(point["time"] * fps)
        awarded = False
        for clip in point["clips"]:
            a, b = round(clip["start"] * fps), round(clip["end"] * fps)
            cuts = [a, result_frame, b] if a < result_frame < b else [a, b]
            for first, last in zip(cuts, cuts[1:]):
                if first >= result_frame and not awarded:
                    state.award(point.get("winner") if point["status"] == "verified" else None)
                    awarded = True
                count = last - first
                spans.append({"source_start_frame": first, "source_end_frame": last,
                              "output_start_frame": cursor, "output_end_frame": cursor + count,
                              "point_index": point_index, "role": clip["role"], "score": state.display()})
                cursor += count
        if not awarded:
            state.award(point.get("winner") if point["status"] == "verified" else None)
    return spans


def write_outputs(log: dict, source: dict, out: Path) -> None:
    manifest = manifest_for(log, source)
    (out / "shots.json").write_text(json.dumps(manifest, indent=2))
    scores = {s["id"]: score_timeline(s) for s in log["segments"]}
    (out / "scores.json").write_text(json.dumps(scores, indent=2))


def load(out: Path) -> tuple[dict, dict]:
    source = json.loads((out / "source.json").read_text())
    log = json.loads((out / "annotations.json").read_text())
    validate_log(log, source)
    return log, source
