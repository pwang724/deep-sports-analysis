import numpy as np
import pytest

from dsa.pose.backends import sanitize_boxes
from dsa.pose.detectors import make_detector


def test_sanitize_boxes_widens_thin_boxes_only():
    out = sanitize_boxes(np.array([[10, 10, 10.5, 50], [0, 0, 20, 30]]))
    assert out[0, 2] == 12 and out[0, 3] == 50
    assert np.array_equal(out[1], [0, 0, 20, 30])


def test_make_detector_rejects_unknown_names():
    with pytest.raises(ValueError):
        make_detector("rtdetr")
