import json
import shutil
import subprocess

import pytest

from dsa.preprocess.desktop_tennis.annotations import edit_timeline, manifest_for, validate_log
from dsa.preprocess.desktop_tennis.scoring import Score


def test_eleven_requires_two_point_margin_and_stops_at_finish():
    s = Score("eleven", points=[10, 10])
    s.award(0)
    assert s.winner is None
    s.award(1)
    s.award(1)
    assert s.winner is None
    s.award(1)
    assert s.points == [11, 13] and s.winner == 1
    with pytest.raises(ValueError, match="finished"):
        s.award(0)


def test_deuce_advantage_and_game_reset():
    s = Score("match", points=[3, 3])
    s.award(0)
    assert s.display()["points"] == ["AD", "40"]
    s.award(1)
    assert s.display()["points"] == ["40", "40"]
    s.award(1)
    s.award(1)
    assert s.games == [0, 1] and s.points == [0, 0]


def test_set_and_tiebreak_keep_two_point_rule():
    s = Score("match", points=[3, 0], games=[5, 4])
    s.award(0)
    assert s.sets == [1, 0] and s.completed_sets == [[6, 4]]
    s.games, s.points = [6, 6], [6, 6]
    s.award(1)
    assert s.tiebreak and s.points == [6, 7]
    s.award(1)
    assert s.sets == [1, 1] and s.completed_sets[-1] == [6, 7]
    assert s.games == s.points == [0, 0]


def test_unknown_point_does_not_fabricate_later_scores():
    s = Score("match")
    s.award(None)
    s.award(0)
    assert s.display()["points"] == ["?", "?"]
    assert s.display()["games"] == ["?", "?"]


@pytest.fixture
def source():
    return {"video": "analysis.mp4", "original_video": "source.MOV", "original_sha256": "abc",
            "fps": 30, "fps_fraction": "30/1", "frames": 300, "width": 640, "height": 360,
            "timeline": "30 fps analysis timeline"}


@pytest.fixture
def log():
    return {"version": 1, "source_sha256": "abc", "players": ["Peter", "Dylan"], "segments": [{
        "id": "match", "kind": "match", "start": 0, "end": 10, "status": "verified",
        "coverage_verified": True, "initial_score": {"verified": True}, "points": [
            {"time": 5.5, "winner": 0, "status": "verified", "clips": [
                {"start": 1, "end": 2, "role": "first_serve_fault"},
                {"start": 4, "end": 6, "role": "rally"}]},
            {"time": 8.5, "winner": 1, "status": "verified", "clips": [
                {"start": 7, "end": 9, "role": "rally"}]}]}]}


def test_failed_first_serve_kept_toss_gap_removed_and_scored_once(log, source):
    manifest = manifest_for(log, source)
    assert [(s["start_frame"], s["end_frame"], s["label"]) for s in manifest["shots"]] == [
        (0, 30, "discard"), (30, 60, "keep"), (60, 120, "discard"), (120, 180, "keep"),
        (180, 210, "discard"), (210, 270, "keep"), (270, 300, "discard")]
    spans = edit_timeline(log["segments"][0], 30)
    assert spans[-1]["output_end_frame"] == 150
    assert spans[0]["score"]["points"] == ["0", "0"]
    assert spans[1]["score"]["points"] == ["0", "0"]
    assert spans[2]["score"]["points"] == ["15", "0"]
    assert spans[-1]["score"]["points"] == ["15", "15"]


def test_unreviewed_footage_is_withheld(log, source):
    s = log["segments"][0]
    s["status"] = "draft"
    s["coverage_verified"] = False
    s["points"][0]["status"] = "draft"
    result = manifest_for(log, source)
    assert result["shots"][0]["label"] == "review"
    assert result["shots"][1]["label"] == "review"
    assert result["shots"][5]["label"] == "keep"


@pytest.mark.parametrize("mutation", [
    lambda l: l.update(source_sha256="different"),
    lambda l: l["segments"][0]["points"][1]["clips"][0].update(start=5),
    lambda l: l["segments"][0]["points"][0].update(time=3),
    lambda l: l["segments"][0]["points"][0].update(winner=None),
    lambda l: l["segments"][0].update(coverage_verified=False),
    lambda l: l["segments"][0].update(id="../outside"),
])
def test_rejects_wrong_source_overlaps_gap_result_and_false_verification(log, source, mutation):
    mutation(log)
    with pytest.raises(ValueError):
        validate_log(log, source)


def test_render_retains_audio_exact_point_frames_and_mapping(tmp_path, log, source):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg required")
    from dsa.preprocess.desktop_tennis.render import render_segment
    from dsa.preprocess.media import source_info

    video = tmp_path / "analysis.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "10",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(video)], check=True)
    source["video"] = str(video)
    path = render_segment(log, source, log["segments"][0], tmp_path)
    info = source_info(path)
    assert info["frames"] == 150
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)]))
    audio = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert abs(float(audio["duration"]) - 5) < .1
    mapping = json.loads(path.with_suffix(".json").read_text())["edit_timeline"]
    assert mapping[-1]["output_end_frame"] == 150
    assert mapping[0]["source_start_frame"] == 30
