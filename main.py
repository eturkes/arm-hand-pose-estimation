#!/usr/bin/env python3
"""Pose estimation using OpenVINO and MediaPipe models.

Usage:
    python main.py                          # webcam 0, NPU device
    python main.py --source 1               # webcam 1
    python main.py --source video.mp4       # single video file
    python main.py --batch-dir videos/      # process all videos in a directory
    python main.py --device CPU             # run on CPU
    python main.py --no-flip                # disable mirror for front camera
    python main.py --tracking hands         # hands only
    python main.py --tracking body          # whole body + hands
"""

import argparse
import collections
import csv
import json
import pathlib
import time

import cv2
import numpy as np
import pygame

from models import download_and_compile_models
from detection import generate_anchors, PALM_INPUT_SIZE, POSE_INPUT_SIZE
from processing import (
    process_frame, match_hands_to_arms, select_primary_body,
    tracking_pose_indices,
    TRACKING_HANDS, TRACKING_HAND_ARM, TRACKING_BODY,
)
from smoothing import PoseSmoother
from constraints import (
    BoneLengthSmoother, clamp_joint_angles,
    BONE_SEGMENTS, BONE_SEGMENTS_BODY,
    ANGLE_LIMITS, ANGLE_LIMITS_BODY,
)
from drawing import draw_body_landmarks, draw_hand_landmarks, draw_arm_hand_bridges
from export import open_csv_writer, frame_to_rows

WINDOW_TITLE = "Pose Estimation"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DIAG_FIELDS = [
    "frame", "timestamp", "bodies_detected", "bodies_rendered",
    "hands_accepted", "hands_rendered", "body_carry",
    "body_track_ages", "hand_track_ages", "detections",
]


def frame_to_surface(frame):
    """Convert a BGR OpenCV frame to a pygame Surface."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))


def process_video(source, flip, models, palm_anchors, pose_anchors,
                  screen, csv_writer=None, video_name=None,
                  single_subject=False, diag_writer=None,
                  tracking=TRACKING_HAND_ARM):
    """Run pose estimation on a single video source with real-time display.

    Returns True if the user requested quit (ESC / window close),
    False if the video simply ended.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"  Cannot open: {source}")
        return False

    fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Optionally set resolution for cameras
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smoother = PoseSmoother()

    # Biomechanical constraints are only used when body is tracked
    use_body = tracking != TRACKING_HANDS
    if use_body:
        _, wrist_kps, shoulder_kps, _ = tracking_pose_indices(tracking)
        bone_segs = (BONE_SEGMENTS_BODY if tracking == TRACKING_BODY
                     else BONE_SEGMENTS)
        angle_lims = (ANGLE_LIMITS_BODY if tracking == TRACKING_BODY
                      else ANGLE_LIMITS)
        bone_smoother = BoneLengthSmoother(segments=bone_segs)

    processing_times = collections.deque(maxlen=200)
    track_state = None
    prev_hand_lm = None
    frame_idx = 0

    # Track age threshold: suppress detections that haven't persisted
    # for at least this many consecutive frames.
    min_track_age = max(1, int(fps_source * 0.1))   # ~3 frames @ 30 fps

    # Single-subject state
    last_body_lm = None
    last_body_vis = None
    frames_since_body = 0
    carry_limit = int(fps_source * 0.5)
    min_hand_age = max(1, int(fps_source * 0.2))


    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    cap.release()
                    return True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    cap.release()
                    return True

            ret, frame = cap.read()
            if not ret:
                break

            if flip:
                frame = cv2.flip(frame, 1)

            # Cap resolution for performance
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)

            frame_h, frame_w = frame.shape[:2]

            # Resize pygame window to match first frame
            if frame_idx == 0:
                screen = pygame.display.set_mode((frame_w, frame_h))
                if video_name:
                    pygame.display.set_caption(f"{WINDOW_TITLE} — {video_name}")

            # Inference — in multi-subject mode, disable synthetic and
            # re-crop hand detections to avoid cascading false positives
            # from spurious body tracks.
            start = time.time()
            body_lm, body_vis, hand_lm, track_state = process_frame(
                frame, models, palm_anchors, pose_anchors,
                prev_state=track_state,
                prev_hand_landmarks=(prev_hand_lm
                                     if single_subject or tracking == TRACKING_HANDS
                                     else None),
                synthesise_hands=single_subject and use_body,
                tracking=tracking,
            )
            elapsed = time.time() - start

            # Temporal smoothing
            t = time.time()

            if use_body:
                body_lm, body_vis, n_bodies_detected = smoother.smooth_bodies(
                    body_lm, body_vis, t,
                    shoulder_indices=shoulder_kps)
            else:
                n_bodies_detected = 0

            hand_lm, n_hands_active = smoother.smooth_hands(
                hand_lm, t, max_tracks=2 if single_subject else None)

            # Enforce biomechanical constraints on each body
            if use_body:
                for i, lm in enumerate(body_lm):
                    bone_smoother.update(i, lm)
                    clamp_joint_angles(lm, limits=angle_lims)
                bone_smoother.prune(range(len(body_lm)))

            if single_subject and use_body:
                # Filter transient hand tracks by age and cap at 2
                hand_ages = smoother.hand_track_ages()
                aged = sorted(
                    ((lm, age) for lm, age in zip(hand_lm, hand_ages)
                     if age >= min_hand_age),
                    key=lambda x: x[1], reverse=True,
                )
                hand_lm = [lm for lm, _ in aged[:2]]
            elif not use_body:
                # Hands-only: cap at 2 and filter by age
                hand_ages = smoother.hand_track_ages()
                aged = sorted(
                    ((lm, age) for lm, age in zip(hand_lm, hand_ages)
                     if age >= min_track_age),
                    key=lambda x: x[1], reverse=True,
                )
                hand_lm = [lm for lm, _ in aged[:2]]
            else:
                # Strip carry-forward ghosts and filter by minimum
                # track age to suppress transient false positives.
                body_ages = smoother.body_track_ages()
                hand_ages = smoother.hand_track_ages()
                body_keep = [
                    i for i, age in enumerate(body_ages)
                    if age >= min_track_age
                ]
                body_lm = [body_lm[i] for i in body_keep]
                body_vis = [body_vis[i] for i in body_keep]
                hand_lm = [
                    lm for lm, age
                    in zip(hand_lm[:n_hands_active], hand_ages)
                    if age >= min_track_age
                ]

            if use_body:
                matches = match_hands_to_arms(
                    body_lm, hand_lm,
                    wrist_kps=wrist_kps, shoulder_kps=shoulder_kps)
            else:
                matches = []

            if single_subject and use_body:
                # Use n_bodies_detected (not len(body_lm)) to distinguish
                # real detections from smoother carry-forward ghosts.
                if n_bodies_detected > 0:
                    # Only consider real detections for primary selection;
                    # carry-forward ghosts (appended after the first
                    # n_bodies_detected entries) are discarded.
                    real_lm = body_lm[:n_bodies_detected]
                    real_vis = body_vis[:n_bodies_detected]
                    real_matches = [
                        m for m in matches if m[0] < n_bodies_detected
                    ]
                    body_lm, body_vis, hand_lm, matches = select_primary_body(
                        real_lm, real_vis, hand_lm, real_matches)
                    last_body_lm = body_lm[0].copy()
                    last_body_vis = body_vis[0].copy()
                    frames_since_body = 0
                elif last_body_lm is not None and frames_since_body < carry_limit:
                    # No real body detection — use last known body
                    body_lm = [last_body_lm]
                    body_vis = [last_body_vis]
                    matches = match_hands_to_arms(
                        body_lm, hand_lm,
                        wrist_kps=wrist_kps, shoulder_kps=shoulder_kps)
                    frames_since_body += 1
                else:
                    # Body fully lost — hand-only fallback
                    body_lm = []
                    body_vis = []
                    matches = []
                    frames_since_body += 1

            # Store final hand landmarks for next frame's re-crop
            prev_hand_lm = [lm.copy() for lm in hand_lm] if hand_lm else None

            # Export landmarks
            if csv_writer is not None:
                timestamp_sec = frame_idx / fps_source
                rows = frame_to_rows(
                    video_name or str(source), frame_idx, timestamp_sec,
                    frame_h, frame_w, body_lm, body_vis, hand_lm, matches,
                    tracking=tracking,
                    hand_only=single_subject,
                )
                for row in rows:
                    csv_writer.writerow(row)

            # Draw overlays
            if use_body:
                frame = draw_body_landmarks(frame, body_lm, body_vis)
                frame = draw_arm_hand_bridges(frame, body_lm, hand_lm, matches)
            frame = draw_hand_landmarks(frame, hand_lm)

            # FPS / progress overlay
            processing_times.append(elapsed)
            avg_ms = np.mean(processing_times) * 1000
            fps = 1000 / avg_ms
            _, f_width = frame.shape[:2]
            label = f"Inference: {avg_ms:.1f}ms ({fps:.1f} FPS)"
            if total_frames > 0:
                pct = frame_idx / total_frames * 100
                label += f"  |  Frame {frame_idx}/{total_frames} ({pct:.0f}%)"
            cv2.putText(frame, label, (20, 40),
                        cv2.FONT_HERSHEY_COMPLEX, f_width / 1000,
                        (0, 0, 255), 1, cv2.LINE_AA)

            # Write per-frame diagnostic row
            if diag_writer is not None:
                hand_diag = track_state.get("hand_diag", []) if track_state else []
                # Detect whether body is carried forward
                if single_subject and use_body:
                    body_carry = (
                        len(body_lm) > 0
                        and n_bodies_detected == 0
                        and frames_since_body > 0
                    )
                else:
                    body_carry = False
                diag_writer.writerow({
                    "frame": frame_idx,
                    "timestamp": round(frame_idx / fps_source, 4),
                    "bodies_detected": n_bodies_detected,
                    "bodies_rendered": len(body_lm),
                    "hands_accepted": sum(1 for d in hand_diag if d.get("accepted")),
                    "hands_rendered": len(hand_lm),
                    "body_carry": body_carry,
                    "body_track_ages": json.dumps(smoother.body_track_ages()),
                    "hand_track_ages": json.dumps(smoother.hand_track_ages()),
                    "detections": json.dumps(hand_diag),
                })

            screen.blit(frame_to_surface(frame), (0, 0))
            pygame.display.flip()
            frame_idx += 1

    finally:
        cap.release()

    return False


def collect_video_files(directory):
    """Return sorted list of video file paths in a directory."""
    d = pathlib.Path(directory)
    if not d.is_dir():
        raise RuntimeError(f"Not a directory: {directory}")
    files = sorted(
        p for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not files:
        raise RuntimeError(f"No video files found in: {directory}")
    return files


def main():
    parser = argparse.ArgumentParser(description="Pose estimation")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--source", default=None,
        help="Video source: camera index (int) or file path (default: 0)")
    source_group.add_argument(
        "--batch-dir", default=None,
        help="Directory of video files to process sequentially")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for CSV output (default: output/)")
    parser.add_argument("--device", default="NPU",
                        help="OpenVINO inference device (default: NPU)")
    parser.add_argument("--no-flip", action="store_true",
                        help="Disable horizontal flip (useful for rear cameras)")
    parser.add_argument("--model-dir", default="model",
                        help="Directory for downloaded/converted models")
    parser.add_argument("--single-subject", action="store_true",
                        help="Keep only the most prominent body per frame")
    parser.add_argument(
        "--tracking", default="hand-arm",
        choices=["hands", "hand-arm", "body"],
        help=("Tracking scope: 'hands' (hands only), "
              "'hand-arm' (arms + hands, default), "
              "'body' (whole body + hands)"))
    parser.add_argument("--postprocess", action="store_true",
                        help="Apply Savitzky-Golay smoothing to CSVs after processing")
    parser.add_argument("--savgol-window", type=int, default=11,
                        help="Savitzky-Golay window length, must be odd (default: 11)")
    parser.add_argument("--savgol-polyorder", type=int, default=3,
                        help="Savitzky-Golay polynomial order (default: 3)")
    args = parser.parse_args()

    tracking = args.tracking

    # Download, convert, and compile models
    models = download_and_compile_models(args.model_dir, args.device)

    # Pre-generate detection anchors
    palm_anchors = generate_anchors(PALM_INPUT_SIZE, strides=[8, 16, 16, 16])
    pose_anchors = generate_anchors(POSE_INPUT_SIZE, strides=[8, 16, 32, 32, 32])

    pygame.init()
    # Placeholder size; process_video resizes on first frame
    screen = pygame.display.set_mode((640, 480))
    pygame.display.set_caption(WINDOW_TITLE)

    csv_paths = []

    try:
        if args.batch_dir:
            video_files = collect_video_files(args.batch_dir)
            print(f"Found {len(video_files)} video(s) in {args.batch_dir}")

            for i, vpath in enumerate(video_files, 1):
                print(f"\nProcessing {i}/{len(video_files)}: {vpath.name}")
                csv_path = pathlib.Path(args.output_dir) / f"{vpath.stem}.csv"
                fh, writer = open_csv_writer(csv_path, tracking=tracking)
                diag_path = pathlib.Path(args.output_dir) / f"{vpath.stem}_diag.csv"
                diag_fh = open(diag_path, "w", newline="")
                diag_w = csv.DictWriter(diag_fh, fieldnames=DIAG_FIELDS)
                diag_w.writeheader()
                try:
                    user_quit = process_video(
                        str(vpath), False, models, palm_anchors, pose_anchors,
                        screen, csv_writer=writer, video_name=vpath.name,
                        single_subject=args.single_subject,
                        diag_writer=diag_w,
                        tracking=tracking,
                    )
                finally:
                    fh.close()
                    diag_fh.close()
                print(f"  Saved: {csv_path}")
                print(f"  Diag:  {diag_path}")
                csv_paths.append(csv_path)
                if user_quit:
                    print("User quit — stopping batch.")
                    break

            print("\nBatch complete.")

        else:
            # Single source mode (camera or file)
            source_arg = args.source if args.source is not None else "0"
            try:
                source = int(source_arg)
                flip = not args.no_flip
            except ValueError:
                source = source_arg
                flip = False

            # For single file mode, also export CSV
            csv_writer = None
            fh = None
            diag_w = None
            diag_fh = None
            if isinstance(source, str):
                vpath = pathlib.Path(source)
                csv_path = pathlib.Path(args.output_dir) / f"{vpath.stem}.csv"
                fh, csv_writer = open_csv_writer(csv_path, tracking=tracking)
                diag_path = pathlib.Path(args.output_dir) / f"{vpath.stem}_diag.csv"
                diag_fh = open(diag_path, "w", newline="")
                diag_w = csv.DictWriter(diag_fh, fieldnames=DIAG_FIELDS)
                diag_w.writeheader()

            video_name = pathlib.Path(source).name if isinstance(source, str) else None
            print(f"Source: {source} | Device: {args.device} | "
                  f"Flip: {flip} | Tracking: {tracking}")
            print("Close the window or press ESC to exit.")

            try:
                process_video(source, flip, models, palm_anchors, pose_anchors,
                              screen, csv_writer=csv_writer,
                              video_name=video_name,
                              single_subject=args.single_subject,
                              diag_writer=diag_w,
                              tracking=tracking)
            finally:
                if fh is not None:
                    fh.close()
                    print(f"Saved: {csv_path}")
                    csv_paths.append(csv_path)
                if diag_fh is not None:
                    diag_fh.close()
                    print(f"Diag:  {diag_path}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        pygame.quit()

    # Post-process CSVs with Savitzky-Golay if requested
    if args.postprocess and csv_paths:
        from postprocess import savgol_smooth_csv

        print(f"\nPost-processing {len(csv_paths)} CSV(s) "
              f"(window={args.savgol_window}, polyorder={args.savgol_polyorder})")
        for csv_path in csv_paths:
            out = csv_path.with_name(f"{csv_path.stem}_smooth.csv")
            savgol_smooth_csv(csv_path, out,
                              window=args.savgol_window,
                              polyorder=args.savgol_polyorder)
            print(f"  {out}")
        print("Post-processing complete.")


if __name__ == "__main__":
    main()
