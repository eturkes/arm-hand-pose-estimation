#!/usr/bin/env python3
"""Build the synthetic demo clip: H.264 video, landmark CSV, and its metadata.

Patient recordings never enter this repository, so the player, its overlay and
its committed proof all run on a generated figure instead.  The CSV is written in
the shipped 2D schema — `body_<keypoint>_{x,y,z,vis}` plus both hands, coordinates
normalised by one scalar, `max(frame_w, frame_h)` — so the reader exercises the
same header derivation a real clip does.

Regenerate (prototype environment, `numpy` + `av` from the dev group)::

    uv run --directory prototype/review-ui python tools/make_fixture.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import av
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "fixtures"
WIDTH, HEIGHT, FPS, SECONDS = 640, 480, 30, 8
SCALE = float(max(WIDTH, HEIGHT))
BODY_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_pinky",
    "right_pinky", "left_index", "right_index", "left_thumb", "right_thumb", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]  # fmt: skip
INDEX = {name: i for i, name in enumerate(BODY_NAMES)}
#: Legs sit under the table in this seated task, so they carry a low visibility —
#: the value the overlay's confidence colouring is there to show.
LOW_VISIBILITY = {
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
}  # fmt: skip

INK = (232, 236, 244)
SKIN = (118, 132, 158)
TABLE = (46, 52, 68)
BACK_TOP = (22, 25, 34)
BACK_BOTTOM = (34, 39, 52)
TARGET = (214, 158, 84)


TABLE_Y = 344
TARGET_XY = (WIDTH / 2 + 152, 322.0)


def _pose(t: float) -> tuple[np.ndarray, float]:
    """One frame of a seated reach-to-grasp, in pixels.  `t` runs 0..1.

    The target is fixed on the table and the hand travels to it, so the clip
    carries real per-frame motion for the overlay and the scrubber to show.
    """
    sway = 5.0 * math.sin(2 * math.pi * t * 2)
    cx, shoulder_y = WIDTH / 2 + sway, 176.0
    reach = 0.5 - 0.5 * math.cos(2 * math.pi * t * 2)  # 0 at rest, 1 at the target

    points = np.zeros((33, 2), dtype=np.float64)

    def put(name: str, x: float, y: float) -> None:
        points[INDEX[name]] = (x, y)

    head_y = shoulder_y - 62 - 3 * math.sin(2 * math.pi * t * 4)
    put("nose", cx, head_y)
    for name, dx, dy in (
        ("left_eye_inner", 6, -8), ("left_eye", 11, -9), ("left_eye_outer", 16, -8),
        ("right_eye_inner", -6, -8), ("right_eye", -11, -9), ("right_eye_outer", -16, -8),
        ("left_ear", 24, -4), ("right_ear", -24, -4),
        ("mouth_left", 8, 11), ("mouth_right", -8, 11),
    ):  # fmt: skip
        put(name, cx + dx, head_y + dy)

    put("left_shoulder", cx + 52, shoulder_y)
    put("right_shoulder", cx - 52, shoulder_y)
    put("left_hip", cx + 40, shoulder_y + 118)
    put("right_hip", cx - 40, shoulder_y + 118)
    put("left_knee", cx + 46, shoulder_y + 196)
    put("right_knee", cx - 46, shoulder_y + 196)
    put("left_ankle", cx + 50, shoulder_y + 262)
    put("right_ankle", cx - 50, shoulder_y + 262)
    put("left_heel", cx + 44, shoulder_y + 272)
    put("right_heel", cx - 44, shoulder_y + 272)
    put("left_foot_index", cx + 72, shoulder_y + 268)
    put("right_foot_index", cx - 72, shoulder_y + 268)

    # Reaching (subject-left) arm drives the motion; the other stays near the lap.
    rest = np.array([cx + 56.0, shoulder_y + 126.0])
    wrist = rest + (np.array(TARGET_XY) - rest) * reach
    shoulder = points[INDEX["left_shoulder"]]
    midpoint = (shoulder + wrist) / 2.0
    span = wrist - shoulder
    bow = np.array([span[1], -span[0]])
    bow = bow / (float(np.hypot(*bow)) or 1.0)
    put("left_elbow", *(midpoint + bow * (30.0 - 16.0 * reach)))
    put("left_wrist", *wrist)
    put("right_elbow", cx - 78, shoulder_y + 70 + 6 * math.sin(2 * math.pi * t * 2))
    put("right_wrist", cx - 62, shoulder_y + 128)
    for side, direction in (("left", 1.0), ("right", -1.0)):
        wx, wy = points[INDEX[f"{side}_wrist"]]
        put(f"{side}_thumb", wx + 9 * direction, wy + 4)
        put(f"{side}_index", wx + 17 * direction, wy - 3)
        put(f"{side}_pinky", wx + 13 * direction, wy + 12)
    return points, reach


def _hand(wrist: np.ndarray, elbow: np.ndarray, direction: float) -> np.ndarray:
    """21 hand keypoints fanned along the forearm axis."""
    axis = wrist - elbow
    norm = float(np.hypot(*axis)) or 1.0
    forward = axis / norm
    lateral = np.array([-forward[1], forward[0]]) * direction
    points = np.zeros((21, 2), dtype=np.float64)
    points[0] = wrist
    for finger in range(5):
        spread = (finger - 2) * 0.36
        base = wrist + forward * 13 + lateral * spread * 13
        for joint in range(4):
            reach = 8.0 + joint * 7.5
            points[1 + finger * 4 + joint] = base + forward * reach + lateral * spread * reach * 0.45
    return points


def _canvas() -> np.ndarray:
    ramp = np.linspace(0.0, 1.0, HEIGHT, dtype=np.float32)[:, None]
    top = np.array(BACK_TOP, dtype=np.float32)
    bottom = np.array(BACK_BOTTOM, dtype=np.float32)
    frame = (top + (bottom - top) * ramp)[:, None, :] * np.ones((1, WIDTH, 1), dtype=np.float32)
    return frame.astype(np.uint8)


def _disc(frame: np.ndarray, centre, radius: float, colour) -> None:
    x0, y0 = int(centre[0] - radius) - 1, int(centre[1] - radius) - 1
    x1, y1 = int(centre[0] + radius) + 2, int(centre[1] + radius) + 2
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, WIDTH), min(y1, HEIGHT)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    mask = (xs - centre[0]) ** 2 + (ys - centre[1]) ** 2 <= radius**2
    frame[y0:y1, x0:x1][mask] = colour


def _bar(frame: np.ndarray, start, end, width: float, colour) -> None:
    steps = max(int(math.hypot(end[0] - start[0], end[1] - start[1]) / 2) + 1, 2)
    for step in range(steps + 1):
        ratio = step / steps
        _disc(
            frame,
            (start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio),
            width,
            colour,
        )


def _render(points: np.ndarray) -> np.ndarray:
    frame = _canvas()
    torso = [INDEX[n] for n in ("left_shoulder", "right_shoulder", "right_hip", "left_hip")]
    for a, b in zip(torso, torso[1:] + torso[:1], strict=True):
        _bar(frame, points[a], points[b], 13, SKIN)
    for chain in (
        ("left_shoulder", "left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow", "right_wrist"),
        ("left_hip", "left_knee", "left_ankle"),
        ("right_hip", "right_knee", "right_ankle"),
    ):
        for a, b in zip(chain, chain[1:], strict=False):
            _bar(frame, points[INDEX[a]], points[INDEX[b]], 9, SKIN)
    _disc(frame, points[INDEX["nose"]], 26, SKIN)
    for name in ("left_wrist", "right_wrist"):
        _disc(frame, points[INDEX[name]], 10, INK)
    # The table occludes the legs after the body is drawn, which is what makes
    # the figure read as seated — and it is why the leg keypoints carry a low
    # visibility in the CSV.
    frame[TABLE_Y:, :] = TABLE
    frame[TABLE_Y : TABLE_Y + 3, :] = INK
    _bar(frame, (TARGET_XY[0], TARGET_XY[1] - 9), (TARGET_XY[0], TARGET_XY[1] + 9), 12, TARGET)
    return frame


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    total = FPS * SECONDS
    header = ["video", "frame_idx", "timestamp_sec", "person_idx"]
    for name in BODY_NAMES:
        header += [f"body_{name}_x", f"body_{name}_y", f"body_{name}_z", f"body_{name}_vis"]
    for side in ("left", "right"):
        for i in range(21):
            header += [
                f"{side}_hand_{i}_x",
                f"{side}_hand_{i}_y",
                f"{side}_hand_{i}_z",
                f"{side}_hand_{i}_conf",
            ]

    rows = []
    container = av.open(str(OUT / "fixture.mp4"), mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    stream.options = {"crf": "20", "preset": "slow"}

    for frame_idx in range(total):
        t = frame_idx / total
        points, _reach = _pose(t)
        image = _render(points)
        packet_frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(packet_frame):
            container.mux(packet)

        row = {
            "video": "fixture/demo",
            "frame_idx": frame_idx,
            "timestamp_sec": round(frame_idx / FPS, 6),
            "person_idx": 0,
        }
        for name, (x, y) in zip(BODY_NAMES, points, strict=True):
            row[f"body_{name}_x"] = round(x / SCALE, 6)
            row[f"body_{name}_y"] = round(y / SCALE, 6)
            row[f"body_{name}_z"] = 0.0
            row[f"body_{name}_vis"] = 0.32 if name in LOW_VISIBILITY else 0.93
        for side, direction in (("left", 1.0), ("right", -1.0)):
            hand = _hand(points[INDEX[f"{side}_wrist"]], points[INDEX[f"{side}_elbow"]], direction)
            for i, (x, y) in enumerate(hand):
                row[f"{side}_hand_{i}_x"] = round(x / SCALE, 6)
                row[f"{side}_hand_{i}_y"] = round(y / SCALE, 6)
                row[f"{side}_hand_{i}_z"] = 0.0
                row[f"{side}_hand_{i}_conf"] = 0.88 if side == "left" else 0.61
        rows.append(row)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    with (OUT / "fixture.csv").open("w", newline="", encoding="utf-8") as stream_out:
        writer = csv.DictWriter(stream_out, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    (OUT / "fixture.json").write_text(
        json.dumps(
            {
                "generator": "prototype/review-ui/tools/make_fixture.py",
                "synthetic": True,
                "task": "cap",
                "side": "l",
                "view": "above",
                "width": WIDTH,
                "height": HEIGHT,
                "fps": FPS,
                "frames": total,
                "duration_s": round(total / FPS, 3),
                "codec": "h264",
                "coord_normalization": "image-isotropic-maxdim",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    video_bytes = (OUT / "fixture.mp4").stat().st_size
    print(f"fixture.mp4 {video_bytes} B, fixture.csv {len(rows)} rows x {len(header)} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
