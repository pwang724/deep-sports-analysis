import numpy as np
import pytest

from dsa.data.tennis_segmentation import CLASS_IDS, mask_to_boxes
from dsa.pose.bench import gt_recall, iou


def test_iou_basic():
    a = np.array([0, 0, 10, 10.0])
    boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30.0]])
    assert iou(a, boxes).round(3).tolist() == [1.0, 0.143, 0.0]
    assert len(iou(a, np.zeros((0, 4)))) == 0


def test_mask_to_boxes_drops_labels_covering_the_frame():
    mask = np.zeros((100, 200), dtype=np.uint8)
    mask[5:95, 5:195] = CLASS_IDS["bottom"]  # 85% of the frame: a mislabelled court
    mask[10:30, 50:60] = CLASS_IDS["top"]
    boxes = mask_to_boxes(mask)
    assert list(boxes) == ["top"]
    assert boxes["top"].tolist() == [50, 10, 60, 30]


def test_gt_recall_matches_by_iou_and_threshold():
    labels = [{"top": np.array([0, 0, 10, 20.0]), "bottom": np.array([100, 100, 150, 200.0])}]
    preds = [(np.array([[1, 1, 11, 21], [100, 100, 150, 200], [300, 300, 310, 310.0]]), np.array([0.9, 0.3, 0.9]))]
    r = gt_recall(preds, labels, threshold=0.25)
    assert r["top_recall"] == 1.0 and r["bottom_recall"] == 1.0 and r["dets_per_image"] == 3
    assert r["top_iou"] == pytest.approx(iou(labels[0]["top"], preds[0][0][:1])[0])
    r = gt_recall(preds, labels, threshold=0.5)
    assert r["bottom_recall"] == 0.0 and np.isnan(r["bottom_iou"]) and r["dets_per_image"] == 2
