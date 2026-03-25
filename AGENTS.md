# Pose Estimation — Agent Context

See README.md for project overview, architecture, usage, and dependencies.

## Tracking Modes

The `--tracking` CLI flag controls body-part scope:

- `hands` — palm + hand landmarks only, no pose detection
- `hands-arms` — 12 arm keypoints + hands (default)
- `body` — all 33 pose keypoints + hands

Mode-specific constants (keypoint indices, wrist/shoulder pairs,
arm chains) are defined in `processing.py:tracking_pose_indices()`.
Export column prefix is `arm_` for hands-arms, `body_` for body mode.

## Git Conventions

- Commit subject line: under 50 characters
- Commit body lines: under 72 characters
- Before committing, check whether README.md, .gitignore, or other
  housekeeping files need updates to stay consistent with the changes

## Environment

- Python virtual environment in `.venv/`
- Host runs Linux with GNOME Wayland and Homebrew Python 3.14
- The `.venv` must be created on the host, not inside a container
  (absolute symlinks to the Python binary are not portable)

## Package Installation

- The user installs packages on the host (add to `requirements.txt`)
- The assistant can install packages inside its container for
  testing, but the user runs the project on the host, not in
  the container
- When installing R packages, use `renv` (not the global library)
- When installing Python or other language packages, use the
  available virtual environment (`.venv/`) rather than system-wide
