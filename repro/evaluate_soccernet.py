#!/usr/bin/env python3
"""Evaluate native BroadTrack camera JSON on SoccerNet GameState.

This implements the published ProCC/SoccerNet-calibration protocol used for
BroadTrack's frame-wise JaC at 5 and 10 pixels and camera-parameter completeness
rate.  It also reports point-, frame-, and field-element-weighted reprojection
diagnostics.  The paper does not publish the scalar MRE/MedRE reduction, so none
of those diagnostic reductions is presented as the exact paper metric.
Evaluation is fixed to the paper's 1920x1080 image domain.

Example::

    python repro/evaluate_soccernet.py \
      --dataset /data/SoccerNetGS \
      --predictions /results/broadtrack \
      --json-out /results/broadtrack-metrics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# The official evaluator imports matplotlib even though this script does not use
# it.  Give matplotlib/fontconfig a writable cache before importing that module.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "broadtrack-mpl")
)
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np


PAPER_WIDTH = 1920
PAPER_HEIGHT = 1080
SAMPLING_FACTOR = 0.9
THRESHOLDS = (5.0, 10.0)
WORKERS = min(8, os.cpu_count() or 1)


def _import_official_evaluator():
    """Import the repository-local SoccerNet calibration implementation."""
    workspace = Path(__file__).resolve().parents[2]
    local_plugin = workspace / "AuxFlow" / "plugins" / "calibration"
    if local_plugin.is_dir():
        sys.path.insert(0, str(local_plugin))
    try:
        from sn_calibration_baseline.evaluate_camera import (  # type: ignore
            evaluate_camera_prediction,
            get_polylines,
        )
    except ImportError as exc:  # pragma: no cover - environment error
        raise SystemExit(
            "Cannot import sn_calibration_baseline. Expected the local package at "
            f"{local_plugin} (or an installed package)."
        ) from exc
    return get_polylines, evaluate_camera_prediction


GET_POLYLINES, OFFICIAL_EVALUATE_CAMERA_PREDICTION = _import_official_evaluator()


# Exact malformed labels called out by the BroadTrack author in issue #1.
LABEL_FIXES = {
    "Big rect. right mai": "Big rect. right main",
    "Big rect.  left bottom": "Big rect. left bottom",
    "Goal left post left ": "Goal left post left",
}

# Central projection through the pitch centre.  This is the official
# SoccerPitch.symetric_classes mapping, with its accidental trailing space
# removed from "Goal left post left".
SYMMETRIC_CLASS = {
    "Side line top": "Side line bottom",
    "Side line bottom": "Side line top",
    "Side line left": "Side line right",
    "Side line right": "Side line left",
    "Middle line": "Middle line",
    "Big rect. left top": "Big rect. right bottom",
    "Big rect. left bottom": "Big rect. right top",
    "Big rect. left main": "Big rect. right main",
    "Big rect. right top": "Big rect. left bottom",
    "Big rect. right bottom": "Big rect. left top",
    "Big rect. right main": "Big rect. left main",
    "Small rect. left top": "Small rect. right bottom",
    "Small rect. left bottom": "Small rect. right top",
    "Small rect. left main": "Small rect. right main",
    "Small rect. right top": "Small rect. left bottom",
    "Small rect. right bottom": "Small rect. left top",
    "Small rect. right main": "Small rect. left main",
    "Circle left": "Circle right",
    "Circle central": "Circle central",
    "Circle right": "Circle left",
    "Goal left crossbar": "Goal right crossbar",
    "Goal left post left": "Goal right post left",
    "Goal left post right": "Goal right post right",
    "Goal right crossbar": "Goal left crossbar",
    "Goal right post left": "Goal left post left",
    "Goal right post right": "Goal left post right",
    "Goal unknown": "Goal unknown",
    "Line unknown": "Line unknown",
}

SEQUENCE_RE = re.compile(r"SNGS[-_](\d+)", re.IGNORECASE)
REQUIRED_CAMERA_KEYS = {
    "sensorResolutionWidthPixels",
    "sensorResolutionHeightPixels",
    "horizontalFieldOfViewDegrees",
    "panDegrees",
    "tiltDegrees",
    "rollDegrees",
    "positionXMeters",
    "positionYMeters",
    "positionZMeters",
    "normalizedRadialDistortionCoefficients",
}


class EvaluationError(RuntimeError):
    """Input violates the fixed evaluation protocol."""


@dataclass(frozen=True)
class SequenceTask:
    sequence: int
    labels_path: str
    predictions: dict[int, dict[str, Any]]


@dataclass
class SequenceResult:
    sequence: int
    total_frames: int
    completed_frames: int
    invalid_frames: int
    jac5: np.ndarray
    jac10: np.ndarray
    reprojection_errors: np.ndarray
    invalid_examples: list[str]
    # The official evaluator exposes point errors grouped by semantic class but
    # does not prescribe how BroadTrack reduced them to scalar MRE/MedRE.  Keep
    # enough sufficient statistics to report transparent sensitivity analyses.
    frame_point_means: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    element_point_means: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )


def canonical_label(name: str) -> str:
    return LABEL_FIXES.get(name, name)


def canonicalize_lines(
    lines: Mapping[str, Sequence[Mapping[str, float]]], width: int, height: int
) -> dict[str, list[dict[str, float]]]:
    """Fix schema typos and scale normalized annotations exactly as SoccerNet."""
    out: dict[str, list[dict[str, float]]] = {}
    for raw_name, points in lines.items():
        name = canonical_label(raw_name)
        scaled = [
            {"x": float(p["x"]) * (width - 1), "y": float(p["y"]) * (height - 1)}
            for p in points
        ]
        if scaled:
            if name in out:
                raise EvaluationError(
                    f"Pitch labels {raw_name!r} and another key both canonicalize to {name!r}"
                )
            out[name] = scaled
    return out


def canonicalize_projected_lines(
    lines: Mapping[str, Sequence[Mapping[str, float]]],
) -> dict[str, list[Mapping[str, float]]]:
    """Put projected model labels in the same canonical namespace as annotations."""
    out: dict[str, list[Mapping[str, float]]] = {}
    raw_names: dict[str, str] = {}
    for raw_name, points in lines.items():
        name = canonical_label(raw_name)
        if name in out:
            raise EvaluationError(
                f"Projected pitch labels {raw_names[name]!r} and {raw_name!r} both "
                f"canonicalize to {name!r}"
            )
        out[name] = list(points)
        raw_names[name] = raw_name
    return out


def mirror_lines(
    lines: Mapping[str, list[dict[str, float]]],
) -> dict[str, list[dict[str, float]]]:
    missing = sorted(set(lines) - set(SYMMETRIC_CLASS))
    if missing:
        raise EvaluationError(
            f"No central-symmetry mapping for pitch labels: {missing}"
        )
    return {SYMMETRIC_CLASS[name]: points for name, points in lines.items()}


def point_to_polyline_distances(
    points: Sequence[Mapping[str, float]], polyline: Sequence[Mapping[str, float]]
) -> np.ndarray:
    """Vectorized equivalent of SoccerNet's distance_to_polyline."""
    p = np.asarray([[x["x"], x["y"]] for x in points], dtype=np.float64)
    q = np.asarray([[x["x"], x["y"]] for x in polyline], dtype=np.float64)
    if not len(p):
        return np.empty(0, dtype=np.float64)
    if not len(q):
        raise EvaluationError("A projected class contains an empty polyline")
    if len(q) == 1:
        return np.linalg.norm(p - q[0], axis=1)

    starts = q[:-1]
    vectors = q[1:] - starts
    lengths2 = np.einsum("ij,ij->i", vectors, vectors)
    offsets = p[:, None, :] - starts[None, :, :]
    coefficients = np.divide(
        np.einsum("nmi,mi->nm", offsets, vectors),
        lengths2[None, :],
        out=np.zeros((len(p), len(starts)), dtype=np.float64),
        where=lengths2[None, :] > 0,
    )
    coefficients = np.clip(coefficients, 0.0, 1.0)
    residuals = offsets - coefficients[:, :, None] * vectors[None, :, :]
    distances2 = np.einsum("nmi,nmi->nm", residuals, residuals)
    return np.sqrt(np.min(distances2, axis=1))


@dataclass(frozen=True)
class OrientationEvaluation:
    detected_classes: frozenset[str]
    groundtruth_classes: frozenset[str]
    distances: dict[str, np.ndarray]

    def jac(self, threshold: float) -> float:
        detected = self.detected_classes
        groundtruth = self.groundtruth_classes
        common = detected & groundtruth
        correct = sum(bool(np.all(self.distances[name] < threshold)) for name in common)
        false_positives = len(detected - groundtruth) + len(common) - correct
        false_negatives = len(groundtruth - detected)
        denominator = correct + false_positives + false_negatives
        return correct / denominator if denominator else 0.0

    def flattened_errors(self) -> np.ndarray:
        if not self.distances:
            return np.empty(0, dtype=np.float64)
        return np.concatenate([self.distances[name] for name in sorted(self.distances)])


def evaluate_orientation(
    projected: Mapping[str, Sequence[Mapping[str, float]]],
    groundtruth: Mapping[str, Sequence[Mapping[str, float]]],
) -> OrientationEvaluation:
    detected = frozenset(projected)
    annotated = frozenset(groundtruth)
    distances = {
        name: point_to_polyline_distances(groundtruth[name], projected[name])
        for name in sorted(detected & annotated)
    }
    return OrientationEvaluation(detected, annotated, distances)


def broadtrack_camera_to_soccernet(camera: Mapping[str, Any]) -> dict[str, Any]:
    """Convert BroadTrack's native camera schema to sn-calibration's schema."""
    missing = sorted(REQUIRED_CAMERA_KEYS - set(camera))
    if missing:
        raise EvaluationError(f"BroadTrack camera is missing keys: {missing}")

    try:
        width = float(camera["sensorResolutionWidthPixels"])
        height = float(camera["sensorResolutionHeightPixels"])
        hfov = math.radians(float(camera["horizontalFieldOfViewDegrees"]))
        pan = float(camera["panDegrees"])
        tilt = float(camera["tiltDegrees"])
        roll = float(camera["rollDegrees"])
        position = [
            float(camera["positionXMeters"]),
            float(camera["positionYMeters"]),
            float(camera["positionZMeters"]),
        ]
        normalized_distortion = [
            float(value) for value in camera["normalizedRadialDistortionCoefficients"]
        ]
    except (TypeError, ValueError) as exc:
        raise EvaluationError(f"Non-numeric BroadTrack camera value: {exc}") from exc

    numeric = [width, height, hfov, pan, tilt, roll, *position, *normalized_distortion]
    if not all(math.isfinite(value) for value in numeric):
        raise EvaluationError("BroadTrack camera contains NaN or infinity")
    if not 0 < hfov < math.pi:
        raise EvaluationError("Invalid horizontal field of view")
    if len(normalized_distortion) > 3:
        raise EvaluationError(
            "BroadTrack polynomial distortion with >3 coefficients is unsupported"
        )
    if not (
        math.isclose(width, PAPER_WIDTH) and math.isclose(height, PAPER_HEIGHT)
    ):
        raise EvaluationError(
            f"camera resolution {width:g}x{height:g} does not match the paper's "
            f"{PAPER_WIDTH}x{PAPER_HEIGHT}"
        )

    focal = width / (2.0 * math.tan(hfov / 2.0))
    normalized_focal = height / focal
    radial = [0.0] * 6
    for index, value in enumerate(normalized_distortion):
        radial[index] = value / normalized_focal ** (2 * (index + 1))

    # Camera.from_json_parameters recomputes rotation from these Euler angles.
    return {
        "pan_degrees": pan,
        "tilt_degrees": tilt,
        "roll_degrees": roll,
        "x_focal_length": focal,
        "y_focal_length": focal,
        "principal_point": [width / 2.0, height / 2.0],
        "position_meters": position,
        "radial_distortion": radial,
        "tangential_distortion": [0.0, 0.0],
        "thin_prism_distortion": [0.0, 0.0, 0.0, 0.0],
    }


def load_pitch_annotations(labels_path: str) -> list[tuple[int, Mapping[str, Any]]]:
    with open(labels_path, encoding="utf-8") as stream:
        labels = json.load(stream)

    images = {image["image_id"]: image for image in labels.get("images", [])}
    frames: list[tuple[int, Mapping[str, Any]]] = []
    seen: set[int] = set()
    for annotation in labels.get("annotations", []):
        if annotation.get("supercategory") != "pitch" and not (
            annotation.get("category_id") == 5 and "lines" in annotation
        ):
            continue
        image = images.get(annotation.get("image_id"))
        if image is None:
            raise EvaluationError(
                f"Pitch annotation has unknown image_id in {labels_path}"
            )
        try:
            width = int(image["width"])
            height = int(image["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationError(
                f"Missing/invalid image resolution for {image.get('file_name')} in {labels_path}"
            ) from exc
        if (width, height) != (PAPER_WIDTH, PAPER_HEIGHT):
            raise EvaluationError(
                f"{labels_path}: label image {image.get('file_name')} is {width}x{height}; "
                f"BroadTrack Table 1 requires {PAPER_WIDTH}x{PAPER_HEIGHT}"
            )
        stem = Path(str(image["file_name"])).stem
        if not stem.isdigit():
            raise EvaluationError(
                f"Non-numeric SoccerNet frame name: {image['file_name']}"
            )
        frame = int(stem)
        if frame in seen:
            raise EvaluationError(
                f"Duplicate pitch annotation for frame {frame} in {labels_path}"
            )
        seen.add(frame)
        frames.append((frame, annotation["lines"]))
    if not frames:
        raise EvaluationError(f"No pitch annotations found in {labels_path}")
    return sorted(frames)


def evaluate_sequence(task: SequenceTask) -> SequenceResult:
    annotated_frames = load_pitch_annotations(task.labels_path)
    jac5: list[float] = []
    jac10: list[float] = []
    all_errors: list[np.ndarray] = []
    frame_point_means: list[float] = []
    element_point_means: list[float] = []
    completed = 0
    invalid = 0
    invalid_examples: list[str] = []

    for frame, raw_lines in annotated_frames:
        camera = task.predictions.get(frame)
        if camera is None:
            continue
        # Dataset/schema errors are protocol errors, not failed camera outputs;
        # keep them outside the invalid-camera handler so evaluation aborts.
        groundtruth = canonicalize_lines(raw_lines, PAPER_WIDTH, PAPER_HEIGHT)
        mirrored_groundtruth = mirror_lines(groundtruth)
        try:
            converted = broadtrack_camera_to_soccernet(camera)
            raw_projected = GET_POLYLINES(
                converted, PAPER_WIDTH, PAPER_HEIGHT, sampling_factor=SAMPLING_FACTOR
            )
        except Exception as exc:  # invalid camera output contributes to incomplete CR
            invalid += 1
            if len(invalid_examples) < 5:
                invalid_examples.append(f"frame {frame:06d}: {exc}")
            continue

        # A projected-label collision is an evaluator/dependency contract error,
        # not an invalid camera.  Keep it outside the camera failure handler.
        projected = canonicalize_projected_lines(raw_projected)
        direct = evaluate_orientation(projected, groundtruth)
        mirrored = evaluate_orientation(projected, mirrored_groundtruth)

        completed += 1
        direct5, mirror5 = direct.jac(THRESHOLDS[0]), mirrored.jac(THRESHOLDS[0])
        direct10, mirror10 = direct.jac(THRESHOLDS[1]), mirrored.jac(THRESHOLDS[1])

        # Match official evaluate_camera.py exactly: each threshold independently
        # chooses the direct result only when it is strictly greater; ties mirror.
        jac5.append(direct5 if direct5 > mirror5 else mirror5)
        jac10.append(direct10 if direct10 > mirror10 else mirror10)
        mre_orientation = direct if direct5 > mirror5 else mirrored
        errors = mre_orientation.flattened_errors()
        if len(errors):
            all_errors.append(errors)
            frame_point_means.append(float(np.mean(errors)))
            element_point_means.extend(
                float(np.mean(class_errors))
                for class_errors in mre_orientation.distances.values()
                if len(class_errors)
            )

    reprojection_errors = (
        np.concatenate(all_errors) if all_errors else np.empty(0, dtype=np.float64)
    )
    return SequenceResult(
        sequence=task.sequence,
        total_frames=len(annotated_frames),
        completed_frames=completed,
        invalid_frames=invalid,
        jac5=np.asarray(jac5, dtype=np.float64),
        jac10=np.asarray(jac10, dtype=np.float64),
        reprojection_errors=reprojection_errors,
        invalid_examples=invalid_examples,
        frame_point_means=np.asarray(frame_point_means, dtype=np.float64),
        element_point_means=np.asarray(element_point_means, dtype=np.float64),
    )


def _sequence_from_text(value: str) -> int | None:
    match = SEQUENCE_RE.search(value)
    return int(match.group(1)) if match else None


def _expand_prediction_paths(values: Sequence[str]) -> list[Path]:
    paths: set[Path] = set()
    for value in values:
        expanded = (
            glob.glob(value, recursive=True) if glob.has_magic(value) else [value]
        )
        if not expanded:
            raise EvaluationError(f"Prediction path/glob matched nothing: {value}")
        for item in expanded:
            path = Path(item)
            if path.is_dir():
                paths.update(
                    candidate
                    for candidate in path.rglob("*.json")
                    if candidate.is_file()
                )
            elif path.is_file():
                paths.add(path)
            else:
                raise EvaluationError(f"Prediction path does not exist: {path}")
    return sorted(paths, key=lambda path: str(path))


def load_native_predictions(
    values: Sequence[str],
) -> tuple[dict[int, dict[int, dict[str, Any]]], list[str]]:
    """Load one or more unmodified JsonCalibDict outputs."""
    predictions: dict[int, dict[int, dict[str, Any]]] = {}
    provenance: dict[tuple[int, int], Path] = {}
    ignored: list[str] = []

    for path in _expand_prediction_paths(values):
        try:
            with open(path, encoding="utf-8") as stream:
                document = json.load(stream)
        except (OSError, json.JSONDecodeError):
            ignored.append(str(path))
            continue
        if not isinstance(document, dict):
            ignored.append(str(path))
            continue

        native_entries = [
            (key, value)
            for key, value in document.items()
            if isinstance(key, str)
            and isinstance(value, dict)
            and isinstance(value.get("cp"), dict)
        ]
        if not native_entries:
            ignored.append(str(path))
            continue

        # Runner outputs use the stable name ``broadtrack.json`` inside an
        # ``SNGS-xxx`` directory, while ad-hoc outputs often encode the
        # sequence in the filename.  Accept either.
        file_sequence = _sequence_from_text(str(path))
        for image_path, record in native_entries:
            sequence = _sequence_from_text(image_path) or file_sequence
            stem = Path(image_path).stem
            if sequence is None or not stem.isdigit():
                raise EvaluationError(
                    f"Cannot infer SNGS sequence/frame from {image_path!r} in {path}"
                )
            frame = int(stem)
            key = (sequence, frame)
            camera = record["cp"]
            if key in provenance:
                previous = predictions[sequence][frame]
                if previous != camera:
                    raise EvaluationError(
                        f"Conflicting camera for SNGS-{sequence:03d} frame {frame:06d}: "
                        f"{provenance[key]} and {path}. Pass only the final per-sequence JSON files."
                    )
                continue
            predictions.setdefault(sequence, {})[frame] = camera
            provenance[key] = path

    if not predictions:
        raise EvaluationError(
            "No native BroadTrack {'image': {'cp': ...}} predictions found"
        )
    return predictions, ignored


def discover_labels(dataset: str, split: str) -> dict[int, Path]:
    split_dir = Path(dataset) / split
    if not split_dir.is_dir():
        raise EvaluationError(f"SoccerNet split directory does not exist: {split_dir}")
    labels: dict[int, Path] = {}
    for path in sorted(split_dir.glob("SNGS-*/Labels-GameState.json")):
        sequence = _sequence_from_text(path.parent.name)
        if sequence is not None:
            labels[sequence] = path
    if not labels:
        raise EvaluationError(
            f"No SNGS-*/Labels-GameState.json files found under {split_dir}"
        )
    return labels


def validate_prediction_resolutions(
    predictions: Mapping[int, Mapping[int, Mapping[str, Any]]],
) -> Counter[tuple[float, float]]:
    resolutions: Counter[tuple[float, float]] = Counter()
    for frames in predictions.values():
        for camera in frames.values():
            try:
                width = float(camera["sensorResolutionWidthPixels"])
                height = float(camera["sensorResolutionHeightPixels"])
            except (KeyError, TypeError, ValueError):
                continue  # reported as an invalid/incomplete frame by the worker
            resolutions[(width, height)] += 1
    mismatches = {
        resolution: count
        for resolution, count in resolutions.items()
        if resolution != (float(PAPER_WIDTH), float(PAPER_HEIGHT))
    }
    if mismatches:
        summary = ", ".join(
            f"{w:g}x{h:g}: {n}" for (w, h), n in sorted(mismatches.items())
        )
        raise EvaluationError(
            f"Non-HD BroadTrack cameras detected ({summary}). The paper evaluates at "
            f"{PAPER_WIDTH}x{PAPER_HEIGHT}; rerun on HD frames."
        )
    return resolutions


def _safe_mean(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if len(values) else None


def _safe_median(values: np.ndarray) -> float | None:
    return float(np.median(values)) if len(values) else None


def summarize(result: SequenceResult) -> dict[str, Any]:
    return {
        "JaC5_percent": None
        if not len(result.jac5)
        else 100.0 * float(np.mean(result.jac5)),
        "JaC10_percent": None
        if not len(result.jac10)
        else 100.0 * float(np.mean(result.jac10)),
        "MRE_pixels": _safe_mean(result.reprojection_errors),
        "MedRE_pixels": _safe_median(result.reprojection_errors),
        "MRE_aggregation": "inferred point-pooled diagnostic",
        "frame_macro_MRE_pixels": _safe_mean(result.frame_point_means),
        "frame_macro_MedRE_pixels": _safe_median(result.frame_point_means),
        "element_macro_MRE_pixels": _safe_mean(result.element_point_means),
        "element_macro_MedRE_pixels": _safe_median(result.element_point_means),
        "CR_percent": 100.0 * result.completed_frames / result.total_frames,
        "completed_frames": result.completed_frames,
        "total_frames": result.total_frames,
        "invalid_frames": result.invalid_frames,
        "reprojection_points": int(len(result.reprojection_errors)),
        "reprojection_frames": int(len(result.frame_point_means)),
        "reprojection_elements": int(len(result.element_point_means)),
        "invalid_examples": result.invalid_examples,
    }


def aggregate_results(results: Sequence[SequenceResult]) -> SequenceResult:
    def concatenate(attribute: str) -> np.ndarray:
        arrays = [
            getattr(result, attribute)
            for result in results
            if len(getattr(result, attribute))
        ]
        return np.concatenate(arrays) if arrays else np.empty(0, dtype=np.float64)

    return SequenceResult(
        sequence=-1,
        total_frames=sum(result.total_frames for result in results),
        completed_frames=sum(result.completed_frames for result in results),
        invalid_frames=sum(result.invalid_frames for result in results),
        jac5=concatenate("jac5"),
        jac10=concatenate("jac10"),
        reprojection_errors=concatenate("reprojection_errors"),
        invalid_examples=[
            f"SNGS-{result.sequence:03d} {example}"
            for result in results
            for example in result.invalid_examples
        ][:20],
        frame_point_means=concatenate("frame_point_means"),
        element_point_means=concatenate("element_point_means"),
    )


def _format_number(value: Any, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def print_table(results: Sequence[SequenceResult]) -> None:
    print("sequence   JaC@5   JaC@10   MRE(px)   MedRE(px)      CR       frames")
    print("-" * 78)
    for result in results:
        summary = summarize(result)
        print(
            f"SNGS-{result.sequence:03d}  "
            f"{_format_number(summary['JaC5_percent']):>6}  "
            f"{_format_number(summary['JaC10_percent']):>7}  "
            f"{_format_number(summary['MRE_pixels']):>8}  "
            f"{_format_number(summary['MedRE_pixels']):>10}  "
            f"{_format_number(summary['CR_percent'], '%'):>7}  "
            f"{result.completed_frames:4d}/{result.total_frames:<4d}"
        )
    overall = summarize(aggregate_results(results))
    print("-" * 78)
    print(
        f"OVERALL   {_format_number(overall['JaC5_percent']):>6}  "
        f"{_format_number(overall['JaC10_percent']):>7}  "
        f"{_format_number(overall['MRE_pixels']):>8}  "
        f"{_format_number(overall['MedRE_pixels']):>10}  "
        f"{_format_number(overall['CR_percent'], '%'):>7}  "
        f"{overall['completed_frames']:4d}/{overall['total_frames']:<4d}"
    )


def run_self_check() -> None:
    """Small dependency-free checks against the official confusion matrix."""

    def check(condition: bool, message: str) -> None:
        if not condition:
            raise EvaluationError(f"self-check failed: {message}")

    check(
        canonical_label("Big rect. right mai") == "Big rect. right main",
        "right-main label",
    )
    check(
        canonical_label("Big rect.  left bottom") == "Big rect. left bottom",
        "left-bottom label",
    )
    check(
        canonical_label("Goal left post left ") == "Goal left post left",
        "goal-post label",
    )
    check(
        all(
            SYMMETRIC_CLASS[opposite] == name
            for name, opposite in SYMMETRIC_CLASS.items()
        ),
        "symmetry map is not involutive",
    )

    projected = {
        "Middle line": [{"x": 0.0, "y": 0.0}, {"x": 10.0, "y": 0.0}],
        "Side line top": [{"x": 0.0, "y": 10.0}, {"x": 10.0, "y": 10.0}],
    }
    groundtruth = {
        "Middle line": [{"x": 2.0, "y": 0.0}, {"x": 8.0, "y": 0.0}],
        "Side line bottom": [{"x": 2.0, "y": 20.0}, {"x": 8.0, "y": 20.0}],
    }
    ours = evaluate_orientation(projected, groundtruth)
    for threshold in THRESHOLDS:
        official, _, _ = OFFICIAL_EVALUATE_CAMERA_PREDICTION(
            projected, groundtruth, threshold
        )
        expected = float(official[0, 0] / official.sum())
        # The reference confusion matrix is float32.
        check(
            math.isclose(ours.jac(threshold), expected, abs_tol=1e-7),
            f"JaC@{threshold:g} differs from the official confusion matrix",
        )

    edge = evaluate_orientation(
        {"Middle line": projected["Middle line"]},
        {"Middle line": [{"x": 5.0, "y": 5.0}]},
    )
    check(edge.jac(5.0) == 0.0, "threshold comparison must be strict")
    check(edge.jac(10.0) == 1.0, "threshold comparison above the edge")

    native = {
        "sensorResolutionWidthPixels": PAPER_WIDTH,
        "sensorResolutionHeightPixels": PAPER_HEIGHT,
        "horizontalFieldOfViewDegrees": 60,
        "panDegrees": 0,
        "tiltDegrees": 80,
        "rollDegrees": 0,
        "positionXMeters": 0,
        "positionYMeters": 55,
        "positionZMeters": -12,
        "normalizedRadialDistortionCoefficients": [0.01],
    }
    converted = broadtrack_camera_to_soccernet(native)
    check(converted["principal_point"] == [960.0, 540.0], "principal point")
    expected_focal = PAPER_WIDTH / (2.0 * math.tan(math.radians(60.0) / 2.0))
    check(math.isclose(converted["x_focal_length"], expected_focal), "focal length")
    legacy = dict(
        native,
        sensorResolutionWidthPixels=960,
        sensorResolutionHeightPixels=540,
    )
    try:
        broadtrack_camera_to_soccernet(legacy)
    except EvaluationError:
        pass
    else:
        raise EvaluationError("self-check failed: non-HD camera was not rejected")
    print("BroadTrack SoccerNet evaluator self-check: OK")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", help="SoccerNet GameState root containing <split>/SNGS-*"
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--predictions",
        nargs="+",
        help="native BroadTrack JSON file(s), directory/directories, or glob(s)",
    )
    parser.add_argument(
        "--json-out", help="optional path for full machine-readable results"
    )
    parser.add_argument(
        "--self-check", action="store_true", help="run small evaluator checks and exit"
    )
    args = parser.parse_args(argv)
    if not args.self_check and (not args.dataset or not args.predictions):
        parser.error(
            "--dataset and --predictions are required unless --self-check is used"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_check:
        run_self_check()
        return 0

    try:
        predictions, ignored = load_native_predictions(args.predictions)
        labels = discover_labels(args.dataset, args.split)
        if not set(labels) & set(predictions):
            raise EvaluationError("Prediction and label sequence sets do not overlap")

        # The label split defines the evaluation universe.  Missing sequences
        # remain present with zero completeness.  Drop predictions outside that
        # universe before validation so they cannot affect the selected metrics.
        predictions = {sequence: predictions.get(sequence, {}) for sequence in labels}
        resolutions = validate_prediction_resolutions(predictions)
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    tasks = [
        SequenceTask(
            sequence=sequence,
            labels_path=str(labels[sequence]),
            predictions=predictions.get(sequence, {}),
        )
        for sequence in sorted(labels)
    ]

    try:
        if WORKERS == 1:
            results = [evaluate_sequence(task) for task in tasks]
        else:
            # executor.map preserves task order; the aggregate floating-point
            # reduction is consequently identical across worker counts.
            with ProcessPoolExecutor(max_workers=WORKERS) as executor:
                results = list(executor.map(evaluate_sequence, tasks, chunksize=1))
    except EvaluationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print_table(results)
    overall_result = aggregate_results(results)
    payload = {
        "protocol": {
            "name": "ProCC / SoccerNet-calibration JaC",
            "resolution": [PAPER_WIDTH, PAPER_HEIGHT],
            "thresholds_pixels": list(THRESHOLDS),
            "projected_pitch_sampling_factor_m": SAMPLING_FACTOR,
            "element_rule": "all annotated points have distance < threshold",
            "ambiguity": "best of direct and central-mirrored labels per frame and threshold",
            "MRE_aggregation": "point-pooled diagnostic; orientation selected by JaC@5",
            "MRE_aggregation_is_inferred": True,
            "MRE_aggregation_note": (
                "The public SoccerNet evaluator exposes per-point errors but does not publish "
                "the scalar MRE/MedRE reduction used by BroadTrack. Aggregate output also "
                "reports frame-macro and semantic-element-macro sensitivity reductions."
            ),
        },
        "input": {
            "dataset": str(Path(args.dataset).resolve()),
            "split": args.split,
            "camera_resolutions": {
                f"{width:g}x{height:g}": count
                for (width, height), count in sorted(resolutions.items())
            },
            "ignored_non_native_json_count": len(ignored),
        },
        "aggregate": summarize(overall_result),
        "per_sequence": {
            f"SNGS-{result.sequence:03d}": summarize(result) for result in results
        },
    }

    invalid = overall_result.invalid_frames
    if invalid:
        print(
            f"warning: {invalid} invalid camera frame(s) counted as incomplete",
            file=sys.stderr,
        )
        for example in overall_result.invalid_examples:
            print(f"  {example}", file=sys.stderr)
    if args.json_out:
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        print(f"Wrote {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
