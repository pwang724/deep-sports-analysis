import numpy as np
import pytest

from dsa.preprocess.uso2026_highlights.pipeline import classify, sample_frames
from dsa.preprocess.media import export_reviews, selection_expression, source_info


def test_sample_frames_stay_within_half_open_shot():
    assert sample_frames(10, 11) == [10]
    assert all(10 <= f < 20 for f in sample_frames(10, 20))
    with pytest.raises(ValueError):
        sample_frames(10, 10)


def test_nearest_reference_is_scale_invariant():
    result = classify(np.array([[10., 0.], [2., 0.]]), np.eye(2), ["keep", "discard"])
    assert result["label"] == "keep"
    assert result["nearest_reference"] == [0, 0]


def test_mixed_shot_and_tied_sample_need_review():
    assert classify(np.eye(2), np.eye(2), ["keep", "discard"])["label"] == "review"
    assert classify(np.array([[1., 1.]]), np.eye(2), ["keep", "discard"])["label"] == "review"


def test_discard_and_optional_ambiguity_margin():
    refs = np.eye(2)
    assert classify(np.array([[0., 1.]]), refs, ["keep", "discard"])["label"] == "discard"
    assert classify(np.array([[1., .99]]), refs, ["keep", "discard"], margin=.02)["label"] == "review"


def test_requires_both_reference_classes():
    with pytest.raises(ValueError, match="each of keep and discard"):
        classify(np.eye(2), np.eye(2), ["keep", "keep"])


def test_exports_partition_source_frames_without_boundary_duplication(tmp_path):
    import shutil
    from dsa.video import H264Writer

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    video = tmp_path / "source.mp4"
    writer = H264Writer(video, 16, (64, 36))
    for i in range(16):
        writer.write(np.full((36, 64, 3), i * 15, dtype=np.uint8))
    writer.release()
    manifest = {"video": str(video), "fps": 16, "frames": 16, "fps_fraction": "16/1", "shots": [
        {"start_frame": 0, "end_frame": 4, "label": "keep"},
        {"start_frame": 4, "end_frame": 7, "label": "discard"},
        {"start_frame": 7, "end_frame": 10, "label": "keep"},
        {"start_frame": 10, "end_frame": 16, "label": "review"},
    ]}
    export_reviews(manifest, tmp_path, width=64)
    assert source_info(tmp_path / "kept.mp4")["frames"] == 7
    assert source_info(tmp_path / "unkept.mp4")["frames"] == 9


def test_tracker_honors_exact_shot_frames_and_source_time(tmp_path, monkeypatch):
    import shutil
    from types import SimpleNamespace
    import pandas as pd
    from dsa.pose import tracking
    from dsa.video import H264Writer

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    video = tmp_path / "source.mp4"
    writer = H264Writer(video, 16, (64, 48))
    for i in range(16):
        writer.write(np.full((48, 64, 3), i * 15, dtype=np.uint8))
    writer.release()
    detector = SimpleNamespace(name="test", detect=lambda rgb: (
        np.array([[10., 10., 40., 40.]]), np.array([.9])))
    models = SimpleNamespace(detector=detector, vitpose=(), device="cpu")
    monkeypatch.setattr(tracking, "run_vitpose", lambda device, rgb, boxes: (
        np.zeros((len(boxes), 17, 2)), np.zeros((len(boxes), 17))))
    tracking.track_segment(video, 0, 1, tmp_path / "shot", models,
                           tracking.TrackConfig(stride=2), 100000, frame_range=(3, 9))
    rows = pd.read_parquet(tmp_path / "shot" / "joints.parquet")
    assert rows.frame.tolist() == [3, 5, 7]
    assert rows.t.tolist() == [3 / 16, 5 / 16, 7 / 16]
    assert all(rows.track_id >= 100000)


def test_many_disjoint_intervals_do_not_exceed_ffmpeg_expression_depth():
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg required")
    expression = selection_expression([{"start_frame": i * 2, "end_frame": i * 2 + 1}
                                       for i in range(150)])
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=16x16:rate=30:duration=10",
                    "-vf", f"select='{expression}'", "-f", "null", "-"], check=True)
