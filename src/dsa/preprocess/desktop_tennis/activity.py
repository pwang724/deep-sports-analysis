"""Audio-assisted play proposals for these stationary phone recordings.

These thresholds are tuned to the September sources, not a universal tennis
classifier. Proposals require visual review, especially isolated impacts,
nearby-court noise, quiet serves, and the start/end padding.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def impact_groups(peaks: np.ndarray, start: float, end: float) -> list[dict]:
    # Columns: source seconds, combined energy, prominence, low, mid, high.
    strong = ((peaks[:, 3] > .045) & (peaks[:, 4] > .018) & (peaks[:, 5] > .009)
              & (peaks[:, 4] < peaks[:, 3] * 1.65))
    weak = ((peaks[:, 3] > .025) & (peaks[:, 4] > .012) & (peaks[:, 5] > .008)
            & (peaks[:, 4] < peaks[:, 3] * 1.8))
    events = np.column_stack([peaks, strong])[weak | strong]
    events = events[(events[:, 0] > start) & (events[:, 0] < end)]
    groups = []
    for event in events:
        if not groups or event[0] - groups[-1][-1][0] > 6:
            groups.append([])
        groups[-1].append(event)
    clips = []
    for group in groups:
        group = np.asarray(group)
        if not group[:, -1].any():
            continue
        anchors = group[group[:, -1] > 0, 0]
        # A quiet return can be followed by more quiet returns. Keep the whole
        # connected group: clipping it around the last loud hit truncated live
        # rallies in the first Dylan edit. Audio endpoints still need visual
        # confirmation; neither silence nor a fixed tail proves a point ended.
        clips.append({"start": max(start, float(group[0, 0] - 1.7)),
                      "end": min(end, float(group[-1, 0] + 2.3)),
                      "impacts": group[:, 0].tolist(), "strong": anchors.tolist(),
                      "point_end_verified": False})
    return clips


def propose_play(out: Path, start: float = 0, end: float | None = None) -> Path:
    from scipy.io import wavfile
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks, stft

    source = json.loads((out / "source.json").read_text())
    duration = source["frames"] / source["fps"]
    end = duration if end is None else end
    if not 0 <= start < end <= duration:
        raise ValueError("Analysis interval must be inside the source")
    with tempfile.TemporaryDirectory(prefix="tennis-audio-") as directory:
        wav = Path(directory) / "audio.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", source["video"], "-vn", "-ac", "1",
                        "-ar", "16000", str(wav)], check=True)
        rate, samples = wavfile.read(wav)
        samples = samples.astype(np.float32) / 32768
    frequency, time, spectrum = stft(samples, rate, nperseg=256, noverlap=128, boundary=None)
    power = np.abs(spectrum) ** 2
    bands = np.stack([np.sqrt(power[(frequency >= lo) & (frequency < hi)].sum(0))
                      for lo, hi in [(250, 900), (900, 2500), (2500, 7500)]], 1)
    energy = bands[:, 1] + bands[:, 2]
    prominence = energy - median_filter(energy, size=101)
    positions, _ = find_peaks(prominence, height=.006, distance=18, prominence=.004)
    peaks = np.column_stack([time[positions], energy[positions], prominence[positions], bands[positions]])
    clips = impact_groups(peaks, start, end)
    path = out / "cuts.candidates.json"
    path.write_text(json.dumps({"version": 1, "source_sha256": source["original_sha256"],
                               "fps": source["fps"], "status": "requires visual review",
                               "candidates": clips}, indent=2) + "\n")
    print(f"{len(clips)} proposals: {path}. Review before creating cuts.json.", flush=True)
    return path
