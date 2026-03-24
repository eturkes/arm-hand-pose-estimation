"""CSV export of per-frame landmark data for downstream feature selection."""

import csv
import pathlib


ARM_KEYPOINT_NAMES = [
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_index_base", "right_index_base",
    "left_middle_base", "right_middle_base",
    "left_pinky_base", "right_pinky_base",
]

HAND_KEYPOINT_COUNT = 21


def make_csv_header():
    """Return the full list of column names."""
    cols = ["video", "frame_idx", "timestamp_sec", "person_idx"]

    for name in ARM_KEYPOINT_NAMES:
        cols.extend([f"arm_{name}_x", f"arm_{name}_y",
                     f"arm_{name}_z", f"arm_{name}_vis"])

    for side in ("left", "right"):
        for i in range(HAND_KEYPOINT_COUNT):
            cols.extend([f"{side}_hand_{i}_x", f"{side}_hand_{i}_y",
                         f"{side}_hand_{i}_z"])

    return cols


def _blank_hand_side(row, side):
    """Fill one hand side with empty strings."""
    for i in range(HAND_KEYPOINT_COUNT):
        row[f"{side}_hand_{i}_x"] = ""
        row[f"{side}_hand_{i}_y"] = ""
        row[f"{side}_hand_{i}_z"] = ""


def _fill_hand_side(row, side, hlm, frame_h, frame_w):
    """Fill one hand side with normalised landmark values."""
    for i in range(HAND_KEYPOINT_COUNT):
        row[f"{side}_hand_{i}_x"] = round(hlm[i, 0] / frame_w, 6)
        row[f"{side}_hand_{i}_y"] = round(hlm[i, 1] / frame_h, 6)
        row[f"{side}_hand_{i}_z"] = round(hlm[i, 2] / frame_w, 6)


def frame_to_rows(video_name, frame_idx, timestamp_sec, frame_h, frame_w,
                  body_landmarks, body_visibilities, hand_landmarks, matches,
                  hand_only=False):
    """Convert one frame's landmark data into CSV rows (one per person).

    Coordinates are normalised to [0, 1] by dividing by frame dimensions.
    Missing hand data is filled with empty strings (written as blank in CSV).

    When *hand_only* is True and no body was detected, a single row is
    emitted with blank arm columns and hand landmarks assigned left/right
    by wrist x-coordinate.
    """
    rows = []

    if body_landmarks:
        # Build a lookup: arm_idx → {4: hand_idx, 5: hand_idx}
        hand_map = {}
        for arm_idx, wrist_kp, hand_idx in matches:
            hand_map.setdefault(arm_idx, {})[wrist_kp] = hand_idx

        for person_idx, (lm, vis) in enumerate(
                zip(body_landmarks, body_visibilities)):
            row = {
                "video": video_name,
                "frame_idx": frame_idx,
                "timestamp_sec": round(timestamp_sec, 4),
                "person_idx": person_idx,
            }

            for kp_idx, name in enumerate(ARM_KEYPOINT_NAMES):
                row[f"arm_{name}_x"] = round(lm[kp_idx, 0] / frame_w, 6)
                row[f"arm_{name}_y"] = round(lm[kp_idx, 1] / frame_h, 6)
                row[f"arm_{name}_z"] = round(lm[kp_idx, 2] / frame_w, 6)
                row[f"arm_{name}_vis"] = round(vis[kp_idx], 4)

            matched_hands = hand_map.get(person_idx, {})
            for wrist_kp, side in [(4, "left"), (5, "right")]:
                hand_idx = matched_hands.get(wrist_kp)
                if hand_idx is not None:
                    _fill_hand_side(row, side, hand_landmarks[hand_idx],
                                    frame_h, frame_w)
                else:
                    _blank_hand_side(row, side)

            rows.append(row)

    elif hand_only and hand_landmarks:
        # No body detected — emit hand-only row with blank arm data.
        row = {
            "video": video_name,
            "frame_idx": frame_idx,
            "timestamp_sec": round(timestamp_sec, 4),
            "person_idx": 0,
        }

        for name in ARM_KEYPOINT_NAMES:
            row[f"arm_{name}_x"] = ""
            row[f"arm_{name}_y"] = ""
            row[f"arm_{name}_z"] = ""
            row[f"arm_{name}_vis"] = ""

        # Assign hands left/right by wrist x-coordinate (max 2).
        sorted_hands = sorted(hand_landmarks[:2],
                              key=lambda lm: lm[0, 0])
        sides = ["left", "right"]
        for i, hlm in enumerate(sorted_hands):
            _fill_hand_side(row, sides[i], hlm, frame_h, frame_w)
        for side in sides[len(sorted_hands):]:
            _blank_hand_side(row, side)

        rows.append(row)

    return rows


def open_csv_writer(output_path):
    """Open a CSV file for writing and return (file_handle, csv.DictWriter)."""
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = make_csv_header()
    fh = open(output_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=header)
    writer.writeheader()
    return fh, writer
