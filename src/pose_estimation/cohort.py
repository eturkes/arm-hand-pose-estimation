"""Cohort aggregate publisher — `(task, side)` statistics over the 2D corpus run.

Contract `.agent/archive/contract-m2u83.md`. This module carries the publisher's
contract-owned half: the domain exception, the label table and the rules that assign a
unit and a range to every published feature. The data-owned half — which columns are
published at all — is measured from the run at publish time (A02), and a disagreement
between the two is a refusal rather than a silent reconciliation.
"""

from __future__ import annotations

from dataclasses import dataclass

GENERATOR = "pose-estimation-cohort"
GENERATOR_VERSION = "v1"


class CohortError(Exception):
    """Every publisher and consumer refusal (A16). P17 asserts this class, not a supertype."""


# A09 + A20. Seven tokens, closed: P10 is a membership test, so growing the vocabulary
# weakens it. No token spells a bare `deg` — anisotropic normalisation was measured at a
# median 9.9 deg against the true image-plane angle, so a published angle is not the angle
# it names.
UNIT_DEG = "deg_image_plane_uncalibrated"
UNIT_FRAME_NORMALIZED = "frame_normalized"
UNIT_FRAME_NORMALIZED_PER_S = "frame_normalized_per_s"
UNIT_RATIO_SHOULDER_WIDTH = "ratio_shoulder_width"
UNIT_RATIO = "ratio"
UNIT_INDEX_SIGNED = "index_signed"
UNIT_DIMENSIONLESS = "dimensionless"

UNIT_VOCABULARY = frozenset(
    {
        UNIT_DEG,
        UNIT_FRAME_NORMALIZED,
        UNIT_FRAME_NORMALIZED_PER_S,
        UNIT_RATIO_SHOULDER_WIDTH,
        UNIT_RATIO,
        UNIT_INDEX_SIGNED,
        UNIT_DIMENSIONLESS,
    }
)

LEVELS = ("frame", "window")
EXCLUSION_REASONS = frozenset({"structurally_absent"})

# Bounds are the measurement's admissible domain read from `analysis/clinical_features.R`
# (A21), never a corpus extreme: D05 refused min/max because at n=15-16 an extreme is one
# identifiable subject, and a descriptor is schema that must not move when the data does.
# `None` is an open end. Sources: symmetry ratio and dominance index :322-324, SAL :366,
# normalized jerk :402, movement efficiency :430, unsigned trunk lean :753, signed lateral
# lean :773, wrapped trunk rotation :795, posture symmetry :812-816.
_FAMILIES: dict[str, tuple[str, str, str, float | None, float | None]] = {
    "elbow_angle_deg": ("肘角度", "Elbow angle", UNIT_DEG, 0.0, 180.0),
    "wrist_deviation_deg": ("手関節偏位角度", "Wrist deviation angle", UNIT_DEG, 0.0, 180.0),
    "finger_spread_deg": ("手指開扇角度", "Finger spread angle", UNIT_DEG, 0.0, 180.0),
    "reach_raw": ("リーチ距離", "Reach distance", UNIT_FRAME_NORMALIZED, 0.0, None),
    "reach_norm": (
        "正規化リーチ距離",
        "Normalized reach distance",
        UNIT_RATIO_SHOULDER_WIDTH,
        0.0,
        None,
    ),
    "grasp_aperture_thumb_index": (
        "母指-示指把持間隔",
        "Thumb-index grasp aperture",
        UNIT_FRAME_NORMALIZED,
        0.0,
        None,
    ),
    "grasp_aperture_thumb_pinky": (
        "母指-小指把持間隔",
        "Thumb-little finger grasp aperture",
        UNIT_FRAME_NORMALIZED,
        0.0,
        None,
    ),
    "wrist_displacement": ("手関節変位", "Wrist displacement", UNIT_FRAME_NORMALIZED, 0.0, None),
    "fingertip_displacement": (
        "指尖変位",
        "Fingertip displacement",
        UNIT_FRAME_NORMALIZED,
        0.0,
        None,
    ),
    "wrist_sal": (
        "手関節スペクトルアーク長",
        "Wrist spectral arc length",
        UNIT_DIMENSIONLESS,
        None,
        0.0,
    ),
    "wrist_velocity_mean": (
        "手関節平均速度",
        "Wrist mean velocity",
        UNIT_FRAME_NORMALIZED_PER_S,
        0.0,
        None,
    ),
    "wrist_velocity_peak": (
        "手関節最大速度",
        "Wrist peak velocity",
        UNIT_FRAME_NORMALIZED_PER_S,
        0.0,
        None,
    ),
    "wrist_normalized_jerk": (
        "手関節正規化ジャーク",
        "Wrist normalized jerk",
        UNIT_DIMENSIONLESS,
        0.0,
        None,
    ),
    "wrist_movement_efficiency": (
        "手関節運動効率",
        "Wrist movement efficiency",
        UNIT_RATIO,
        1.0,
        None,
    ),
    "fingertip_normalized_jerk": (
        "指尖正規化ジャーク",
        "Fingertip normalized jerk",
        UNIT_DIMENSIONLESS,
        0.0,
        None,
    ),
    "trunk_lean": ("体幹傾斜角度", "Trunk lean angle", UNIT_DEG, 0.0, 90.0),
    "trunk_lean_lateral": (
        "体幹側方傾斜角度",
        "Trunk lateral lean angle",
        UNIT_DEG,
        -180.0,
        180.0,
    ),
    "trunk_rotation": ("体幹回旋角度", "Trunk rotation angle", UNIT_DEG, -180.0, 180.0),
    "posture_symmetry": ("姿勢対称性", "Posture symmetry", UNIT_RATIO_SHOULDER_WIDTH, -1.0, 1.0),
    "compensatory_pattern_index": (
        "代償パターン指数",
        "Compensatory pattern index",
        UNIT_INDEX_SIGNED,
        -1.0,
        1.0,
    ),
}

# The R source spells the three trunk angles with a `_deg` suffix at frame level and
# without one in its window aggregates, so one family answers to both names.
_FAMILY_ALIASES = {
    "trunk_lean_deg": "trunk_lean",
    "trunk_lean_lateral_deg": "trunk_lean_lateral",
    "trunk_rotation_deg": "trunk_rotation",
}

_SIDES = {"left_": ("左", "Left"), "right_": ("右", "Right")}

# Suffix -> (ja template, en template). A bilateral ratio or index replaces its family's
# unit; every other derivation inherits it (A09).
_DERIVATIONS: dict[str, tuple[str, str, str | None]] = {
    "_symmetry_ratio": ("{}の左右対称比", "{} left-right symmetry ratio", UNIT_RATIO),
    "_dominance_index": ("{}の左右優位指数", "{} left-right dominance index", UNIT_INDEX_SIGNED),
    "_abs_diff": ("{}の左右差", "{} left-right absolute difference", None),
    "_mean": ("{}の平均", "{} mean", None),
    "_sd": ("{}の標準偏差", "{} standard deviation", None),
    "_range": ("{}の範囲", "{} range", None),
}


@dataclass(frozen=True)
class Feature:
    """One published feature. A08 deleted the separate id: `(level, column)` is the key."""

    level: str
    column: str
    ja: str
    en: str
    unit: str
    range: tuple[float | None, float | None]

    @property
    def raw(self) -> str:
        """`../rehab` descriptor key. D10's prefix is what keeps it off all 219 existing raws."""
        return f"pose_{self.level}_{self.column}"


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _decompose(column: str) -> tuple[str, str | None, str | None]:
    """`column` -> (family, side prefix, derivation suffix). Raises on an unknown column."""
    side = next((prefix for prefix in _SIDES if column.startswith(prefix)), None)
    stem = column[len(side) :] if side else column
    resolved = _FAMILY_ALIASES.get(stem, stem)
    if resolved in _FAMILIES:
        return resolved, side, None
    # Family match precedes suffix stripping, or `wrist_velocity_mean` decomposes as a
    # window aggregate of a `wrist_velocity` family that does not exist.
    for suffix in _DERIVATIONS:
        if not stem.endswith(suffix):
            continue
        base = stem[: -len(suffix)]
        base = _FAMILY_ALIASES.get(base, base)
        if base in _FAMILIES:
            return base, side, suffix
    raise CohortError(f"no label family for column {column!r}")


def describe(level: str, column: str) -> Feature:
    """Build one feature's labels, unit and range by rule (A09, A20, A21)."""
    if level not in LEVELS:
        raise CohortError(f"unknown level {level!r}")
    family, side, derivation = _decompose(column)
    ja, en, unit, low, high = _FAMILIES[family]
    if side:
        side_ja, side_en = _SIDES[side]
        ja, en = f"{side_ja}{ja}", f"{side_en} {_lower_first(en)}"
    if derivation:
        ja_template, en_template, override = _DERIVATIONS[derivation]
        ja, en = ja_template.format(ja), en_template.format(en)
        if override is not None:
            unit = override
            low, high = (0.0, 1.0) if override == UNIT_RATIO else (-1.0, 1.0)
        elif derivation in ("_abs_diff", "_range"):
            # A magnitude over the family's own span: zero to its width when bounded.
            low, high = 0.0, (None if low is None or high is None else high - low)
        elif derivation == "_sd":
            low, high = 0.0, None
    if unit not in UNIT_VOCABULARY:
        raise CohortError(f"unit {unit!r} for {level}/{column} is outside the vocabulary")
    return Feature(level=level, column=column, ja=ja, en=en, unit=unit, range=(low, high))


# The published schema: contract-owned and therefore frozen (A10), while the published
# PARTITION is measured from the run (A02). A column the run stops emitting, or newly
# emits, fails the cross-check by name rather than silently moving the product.
_FRAME_BILATERAL = (
    "elbow_angle_deg",
    "wrist_deviation_deg",
    "finger_spread_deg",
    "reach_raw",
    "reach_norm",
    "grasp_aperture_thumb_index",
    "grasp_aperture_thumb_pinky",
    "wrist_displacement",
    "fingertip_displacement",
)
_WINDOW_BILATERAL = (
    "wrist_sal",
    "wrist_velocity_mean",
    "wrist_velocity_peak",
    "wrist_normalized_jerk",
    "wrist_movement_efficiency",
    "fingertip_normalized_jerk",
)
_BILATERAL_SUFFIXES = ("_symmetry_ratio", "_dominance_index", "_abs_diff")

FRAME_COLUMNS: tuple[str, ...] = (
    *(f"{side}{base}" for side in ("left_", "right_") for base in _FRAME_BILATERAL),
    *(f"{base}{suffix}" for base in _FRAME_BILATERAL for suffix in _BILATERAL_SUFFIXES),
    "trunk_lean_deg",
    "trunk_lean_lateral_deg",
    "trunk_rotation_deg",
    "posture_symmetry",
)
WINDOW_COLUMNS: tuple[str, ...] = (
    *(f"{side}{base}" for side in ("left_", "right_") for base in _WINDOW_BILATERAL),
    *(f"{base}{suffix}" for base in _WINDOW_BILATERAL for suffix in _BILATERAL_SUFFIXES),
    "compensatory_pattern_index",
    "trunk_lean_mean",
    "trunk_lean_sd",
    "trunk_lean_range",
    "trunk_lean_lateral_mean",
    "trunk_lean_lateral_sd",
    "trunk_rotation_mean",
    "trunk_rotation_sd",
    "posture_symmetry_mean",
    "posture_symmetry_sd",
)

# `trunk_lean_sagittal_deg` and its `_mean`/`_sd` aggregates are absent by measurement:
# `clinical_features.R:1038` assigns NA_real_ on the 2D branch because sagittal lean is
# out of plane, so they carry zero finite values over the whole corpus and A02 sends them
# to `excluded`. They are not listed here, and P03 re-derives that partition from the run.
FEATURES: tuple[Feature, ...] = tuple(
    describe(level, column)
    for level, columns in (("frame", FRAME_COLUMNS), ("window", WINDOW_COLUMNS))
    for column in sorted(columns)
)

FEATURE_KEYS: frozenset[tuple[str, str]] = frozenset(
    (feature.level, feature.column) for feature in FEATURES
)
