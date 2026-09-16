from types import SimpleNamespace

import numpy as np

from dsa.pose.eval_pose import SIGMAS, crop_box, oks, run_rfdetr_crop


def test_crop_box_pads_and_clips():
    assert crop_box((100, 200), np.array([10, 10, 30, 50.0]), pad=0.5).tolist() == [0, 0, 40, 70]


def test_run_rfdetr_crop_maps_joints_back_and_picks_overlapping_person():
    calls = []

    def predict(crop, threshold, include_source_image):
        calls.append(crop.shape)
        # two people in the crop: a far-off one and the one overlapping the target box
        return SimpleNamespace(
            xy=np.stack([np.full((17, 2), 90.0), np.full((17, 2), 15.0)]), confidence=np.ones((2, 17)),
            data={"xyxy": np.array([[80, 80, 100, 100], [5, 5, 25, 45.0]])},
        )

    rgb = np.zeros((200, 300, 3), dtype=np.uint8)
    box = np.array([[50, 60, 70, 100.0]])  # 20x40 box; pad 0.3 -> crop starts at (44, 48)
    xy, sc, found = run_rfdetr_crop(SimpleNamespace(predict=predict), rgb, box)
    assert calls == [(64, 32, 3)]
    assert found.tolist() == [True]
    assert np.allclose(xy[0, 0], [15 + 44, 15 + 48])


def test_oks_is_one_for_exact_prediction_and_ignores_invisible_joints():
    gt = np.random.RandomState(0).rand(17, 2) * 100
    vis = np.ones(17)
    vis[0] = 0
    pred = gt.copy()
    pred[0] += 1000  # invisible joint, must not count
    assert oks(pred, gt, vis, area=100 * 100) == 1.0
    assert len(SIGMAS) == 17
