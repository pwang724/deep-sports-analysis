from dsa.cloud.modal_app import VOL, _rel


def test_rel_strips_container_mount_prefix():
    """Volume client calls take paths relative to the volume root, not the /vol mount."""
    assert _rel(VOL / "models" / "x.pth") == "/models/x.pth"
    assert _rel(VOL / "runs") == "/runs"
