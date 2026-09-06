"""Adversarial regressions found in the M2.8.4 milestone review."""

from __future__ import annotations

import pose_estimation.cohort as cohort
import pose_estimation.video_io as video_io


class _OrientationRejectingCapture:
    def __init__(self) -> None:
        self.released = False

    def isOpened(self) -> bool:
        return True

    def set(self, _property: int, _value: int) -> bool:
        return False

    def release(self) -> None:
        self.released = True


def test_open_capture_refuses_failed_orientation_request(tmp_path, monkeypatch) -> None:
    source = tmp_path / "synthetic.mp4"
    source.touch()
    capture = _OrientationRejectingCapture()
    monkeypatch.setattr(video_io.cv2, "VideoCapture", lambda _source: capture)

    assert video_io.open_capture(str(source)) is None
    assert capture.released


def test_corrected_cohort_angles_publish_isotropic_unit() -> None:
    assert cohort.UNIT_DEG == "deg_image_plane"
