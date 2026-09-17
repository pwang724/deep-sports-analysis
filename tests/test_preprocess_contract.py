import hashlib
import json

import pytest

from dsa.preprocess.manifest import validate_manifest
from dsa.preprocess.uso2026_highlights.pipeline import load_profile, refine_changes


@pytest.mark.parametrize("shots", [
    [{"start_frame": 1, "end_frame": 10, "label": "keep"}],
    [{"start_frame": 0, "end_frame": 9, "label": "keep"}],
    [{"start_frame": 0, "end_frame": 11, "label": "keep"}],
    [{"start_frame": 0.0, "end_frame": 10, "label": "keep"}],
    [{"start_frame": 0, "end_frame": 10, "label": "unknown"}],
    [{"start_frame": 0, "end_frame": 4, "label": "keep"},
     {"start_frame": 5, "end_frame": 10, "label": "discard"}],
])
def test_pipeline_contract_rejects_incomplete_or_invalid_partition(shots):
    with pytest.raises(ValueError):
        validate_manifest({"video": "clip.mp4", "fps": 30, "frames": 10, "shots": shots})


def test_profile_rejects_changed_source_and_invalid_reference(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"original source")
    profile = tmp_path / "profile.json"
    config = {"source": {"filename": video.name, "sha256": hashlib.sha256(video.read_bytes()).hexdigest()},
              "sample_fps": 4, "margin": 0.04,
              "cuts": {"adaptive_threshold": 3, "min_content_val": 7},
              "references": [{"time": 0, "label": "keep"}, {"time": 1, "label": "discard"}]}
    profile.write_text(json.dumps(config))
    info = {"fps": 30, "frames": 60}
    loaded = load_profile(profile, video, info)
    assert loaded["cuts"] == config["cuts"]
    assert loaded["sample_fps"] == 4
    video.write_bytes(b"other video with the same filename")
    with pytest.raises(ValueError, match="source hash"):
        load_profile(profile, video, info)
    config.pop("source")
    config["references"][1]["time"] = 2
    profile.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="outside"):
        load_profile(profile, video, info)


def test_refinement_keeps_intermediate_review_band():
    def label(frame):
        return "keep" if frame < 4 else "review" if frame < 7 else "discard"
    sampled = {0, 10}
    def read(frame):
        assert frame in sampled
        return label(frame)
    result = refine_changes([0, 10], read, sampled.update)
    assert result == [(3, 4), (6, 7)]
