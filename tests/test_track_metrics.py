import pandas as pd

from dsa.pose.track_metrics import assign_players, id_changes, tracker_report

W, H = 1920, 1080


def _row(frame, tid, cx, foot_y, h=200, w=60):
    return dict(frame=frame, track_id=tid, x1=cx - w / 2, y1=foot_y - h, x2=cx + w / 2, y2=foot_y, conf=0.9)


def test_assign_players_uses_position_bands():
    rows = pd.DataFrame([
        _row(0, 1, 960, 950),          # near player
        _row(0, 2, 960, 250, h=100),   # far player, feet just below the far baseline
        _row(0, 3, 300, 600, h=90),    # ball kid at the left net post: outside both bands
        _row(0, 4, 960, 700),          # someone mid-court, lower feet than far but not lowest overall
    ])
    a = assign_players(rows, W, H).set_index("player").track_id
    assert a["near"] == 1 and a["far"] == 2


def test_id_changes_counts_switches_and_ignores_gaps():
    assert id_changes(pd.Series([1, 1, None, 1, 2, 2, 1])) == 2
    assert id_changes(pd.Series([None, None])) == 0


def test_tracker_report_coverage_and_ids():
    rows = pd.DataFrame([_row(0, 1, 960, 950), _row(1, 1, 960, 950), _row(2, 7, 960, 950), _row(0, 2, 960, 250, h=100)])
    r = tracker_report(rows, frames=[0, 1, 2, 3], W=W, H=H)
    assert r["near_found"] == 0.75 and r["near_ids"] == 2 and r["near_id_changes"] == 1
    assert r["far_found"] == 0.25 and r["far_ids"] == 1
    assert r["boxes_per_frame"] == 1.0 and r["tracks"] == 3


def test_assign_players_follows_the_person_across_frames():
    """Near band holds two people; the choice sticks with the one picked first even when the other has lower feet."""
    rows = pd.DataFrame([
        _row(0, 1, 960, 950), _row(0, 2, 700, 900),   # frame 0: player 1 lowest
        _row(1, 1, 960, 940), _row(1, 2, 700, 960),   # frame 1: person 2 now lower, but player 1 overlaps last choice
    ])
    a = assign_players(rows, W, H)
    assert a[a.player == "near"].track_id.tolist() == [1, 1]
