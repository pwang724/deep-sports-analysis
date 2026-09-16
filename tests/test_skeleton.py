import numpy as np

from dsa.pose.skeleton import COCO17, SKELETON, draw_pose, joints_to_columns


def test_coco17_and_skeleton_are_consistent():
    assert len(COCO17) == 17
    assert all(0 <= a < 17 and 0 <= b < 17 for a, b in SKELETON)


def test_joints_to_columns_names_and_values():
    xy = np.arange(34, dtype=float).reshape(17, 2)
    sc = np.linspace(0, 1, 17)
    cols = joints_to_columns(xy, sc)
    assert len(cols) == 51
    assert cols["nose_x"] == 0.0 and cols["nose_y"] == 1.0
    assert cols["r_ankle_c"] == 1.0


def test_draw_pose_modifies_frame_in_place():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    xy = np.full((17, 2), 50.0)
    draw_pose(frame, xy, np.ones(17), (255, 255, 255), track_id=1, box=(40, 40, 60, 60))
    assert frame.sum() > 0
