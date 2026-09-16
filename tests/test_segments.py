import json

import pandas as pd
import pytest

from dsa.pose.segments import TRACK_ID_STRIDE, merge_tables, plan_segments


def test_plan_segments_covers_range_exactly():
    segs = plan_segments(100.0, 34.5, 15.0)
    assert [(s.start, s.duration) for s in segs] == [(100.0, 15.0), (115.0, 15.0), (130.0, 4.5)]
    assert [s.index for s in segs] == [0, 1, 2]
    assert segs[2].track_id_offset == 2 * TRACK_ID_STRIDE
    assert segs[0].name == "seg0000"


def test_plan_segments_rejects_bad_input():
    with pytest.raises(ValueError):
        plan_segments(0, 0, 10)


def _write_segment(d, start, frames, rows):
    d.mkdir()
    pd.DataFrame(rows).to_parquet(d / "joints.parquet", index=False)
    (d / "summary.json").write_text(json.dumps({
        "video": "v.mp4", "start": start, "duration": 1.0, "fps": 60.0, "config": {"stride": 2},
        "frames_processed": frames, "rows": len(rows), "tracks": 1,
        "sec_per_frame": {"rfdetr": 0.1 * (start + 1), "vitpose": 0.2},
    }))


def test_merge_tables_concatenates_and_weights_timing(tmp_path):
    a, b = tmp_path / "seg0000", tmp_path / "seg0001"
    _write_segment(a, 0.0, 10, [{"frame": 1, "track_id": 1}])
    _write_segment(b, 1.0, 30, [{"frame": 2, "track_id": 100001}])
    summary = merge_tables([a, b], tmp_path / "merged")

    df = pd.read_parquet(tmp_path / "merged" / "joints.parquet")
    assert df.frame.tolist() == [1, 2]
    assert summary["frames_processed"] == 40 and summary["segments"] == 2 and summary["tracks"] == 2
    assert summary["sec_per_frame"]["rfdetr"] == pytest.approx((0.1 * 10 + 0.2 * 30) / 40)
