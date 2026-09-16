import shutil
import subprocess

import numpy as np
import pytest

from dsa.video import H264Writer, codec_of

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_h264_writer_produces_a_decodable_h264_file(tmp_path):
    out = tmp_path / "clip.mp4"
    w = H264Writer(out, fps=30, size=(64, 48))
    for i in range(10):
        w.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    w.release()
    assert codec_of(out) == "h264"
    frames = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                             "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
                            capture_output=True, text=True).stdout.strip()
    assert frames == "10"
