import json
import shutil
import subprocess
from importlib.resources import files

import numpy as np
import pytest

from dsa.preprocess.desktop_tennis.activity import impact_groups
from dsa.preprocess.desktop_tennis.cuts import cut_manifest, cut_timeline, export_play, validate_cuts


@pytest.fixture
def source():
    return {"video": "analysis.mp4", "original_video": "source.MOV", "original_sha256": "abc",
            "fps": 30, "fps_fraction": "30/1", "frames": 300, "width": 320, "height": 180}


@pytest.fixture
def edit():
    # First serve, second attempt, and another activity window; no winners/rules.
    return {"version": 1, "source_sha256": "abc", "fps": 30, "clips": [
        {"start_frame": 31, "end_frame": 57}, {"start_frame": 123, "end_frame": 166},
        {"start_frame": 207, "end_frame": 269}]}


def test_unscored_edits_keep_both_attempts_without_point_labels(edit, source):
    spans = cut_timeline(edit, source)
    assert spans[-1]["output_end_frame"] == 131
    assert [s["source_start_frame"] for s in spans] == [31, 123, 207]
    manifest = cut_manifest(edit, source)
    assert [s["label"] for s in manifest["shots"]] == [
        "discard", "keep", "discard", "keep", "discard", "keep", "discard"]
    assert sum(s["end_frame"] - s["start_frame"] for s in manifest["shots"]) == 300


@pytest.mark.parametrize("change", [
    lambda e: e.update(source_sha256="other"),
    lambda e: e.update(fps=25),
    lambda e: e["clips"][1].update(start_frame=56),
    lambda e: e["clips"][2].update(end_frame=301),
    lambda e: e.update(clips=[]),
])
def test_invalid_edits_are_rejected(edit, source, change):
    change(edit)
    with pytest.raises(ValueError):
        validate_cuts(edit, source)


def test_opening_audio_examples_keep_serves_but_not_caught_toss():
    # Measured Dylan examples: pre-serve bounces/caught toss, then two struck
    # serves. Isolated struck serves must survive even without a return.
    peaks = np.array([
        [31.904, .065, .063, .031, .060, .005],
        [33.944, .049, .047, .057, .043, .007],
        [39.096, .074, .071, .143, .046, .027],
        [41.152, .100, .097, .190, .074, .027],
        [65.640, .057, .055, .073, .040, .017],
    ])
    clips = impact_groups(peaks, 30, 80)
    assert len(clips) == 2
    assert clips[0]["start"] > 34
    assert clips[0]["strong"] == [39.096, 41.152]
    assert clips[1]["strong"] == [65.640]


def test_unscored_export_is_frame_exact_with_audio(tmp_path, edit, source):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg required")
    video = tmp_path / "analysis.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "10",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(video)], check=True)
    source["video"] = str(video)
    (tmp_path / "source.json").write_text(json.dumps(source))
    (tmp_path / "cuts.json").write_text(json.dumps(edit))
    path = export_play(tmp_path, encoder="libx264")
    result = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]))
    v = next(s for s in result["streams"] if s["codec_type"] == "video")
    a = next(s for s in result["streams"] if s["codec_type"] == "audio")
    assert int(v["nb_frames"]) == 131
    assert abs(float(a["duration"]) - 131 / 30) < .1
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert metadata["scoring"] == "none" and metadata["clip_count"] == 3
    assert not (tmp_path / "annotations.json").exists()
    with pytest.raises(ValueError, match="overwrite"):
        export_play(tmp_path, destination=video)


def test_quiet_rally_continuation_is_not_trimmed_to_last_loud_hit():
    # One loud serve followed by several softer returns. The old anchor trim
    # discarded the last two hits even though all gaps were within the rally.
    peaks = np.array([
        [10., .08, .07, .10, .05, .02],
        [12., .04, .03, .03, .02, .01],
        [14., .04, .03, .03, .02, .01],
        [16., .04, .03, .03, .02, .01],
    ])
    clips = impact_groups(peaks, 0, 30)
    assert len(clips) == 1
    assert clips[0]["impacts"] == [10., 12., 14., 16.]
    assert clips[0]["end"] > 16
    assert clips[0]["point_end_verified"] is False


def test_dylan_revision_preserves_reviewed_rally_endings_and_internal_gaps():
    profile = files("dsa.preprocess.desktop_tennis").joinpath("profiles/dylan_cuts.json")
    edit = json.loads(profile.read_text())
    # Source intervals checked visually after the first export cut these rallies
    # short. The 472-500s rally also had an erroneous internal cut at 477s.
    for start, end in [(238, 246), (327, 338), (473, 500.5),
                       (959, 969), (2145, 2151.5), (3054, 3066)]:
        assert any(c["start_frame"] <= round(start * 30)
                   and c["end_frame"] >= round(end * 30) for c in edit["clips"])
    # Both inspected serve attempts remain; the opening caught tosses do not.
    for t in [132.24, 143.064]:
        assert any(c["start_frame"] <= t * 30 < c["end_frame"] for c in edit["clips"])
    for t in [34, 60]:
        assert not any(c["start_frame"] <= t * 30 < c["end_frame"] for c in edit["clips"])


def test_richard_revision_preserves_quiet_continuations_and_serve_attempts():
    profile = files("dsa.preprocess.desktop_tennis").joinpath("profiles/richard_cuts.json")
    edit = json.loads(profile.read_text())
    # Visually checked source intervals: late volleys, quiet far-end serves,
    # and a long continuation beyond the original 3292-second endpoint.
    for start, end in [(128, 144.5), (840, 851), (1545, 1559),
                       (1650, 1660), (2118, 2135), (2335, 2355),
                       (2494, 2503), (3288, 3309), (3498, 3509)]:
        assert any(c["start_frame"] <= round(start * 30)
                   and c["end_frame"] >= round(end * 30) for c in edit["clips"])
    # Empty-court bench break and two caught tosses remain excluded.
    for t in [1300, 2015.5, 2876]:
        assert not any(c["start_frame"] <= t * 30 < c["end_frame"] for c in edit["clips"])
