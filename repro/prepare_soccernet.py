#!/usr/bin/env python3
"""Prepare deterministic SoccerNet-GSR inputs for BroadTrack.

The source dataset is treated as read-only.  Generated player masks and the
per-game tripod files from BroadTrack issue #1 are written below a separate
artifact root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TEST_SEQUENCE_NAMES = tuple(
    [f"SNGS-{number:03d}" for number in range(116, 151)]
    + [f"SNGS-{number:03d}" for number in range(187, 201)]
)
HUMAN_ROLES = frozenset({"goalkeeper", "other", "player", "referee"})
ROLE_CATEGORY_IDS = {
    "player": 1,
    "goalkeeper": 2,
    "referee": 3,
    "ball": 4,
    "other": 7,
}
ARTIFACT_FORMAT_VERSION = 1


class PreparationError(RuntimeError):
    """Raised when source data or generated artifacts are inconsistent."""


@dataclass(frozen=True)
class Expectations:
    sequence_names: tuple[str, ...] = TEST_SEQUENCE_NAMES
    frames_per_sequence: int = 750
    image_width: int = 1920
    image_height: int = 1080


@dataclass(frozen=True)
class HumanBox:
    role: str
    coordinates: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class FrameData:
    image_id: str
    file_name: str
    image_path: Path
    boxes: tuple[HumanBox, ...]


@dataclass(frozen=True)
class SequenceData:
    name: str
    game_id: str
    labels_path: Path
    frames_directory: Path
    frames: tuple[FrameData, ...]


@dataclass(frozen=True)
class SourceData:
    dataset_root: Path
    dataset_version: str
    sequences: tuple[SequenceData, ...]
    tripods: Mapping[str, Mapping[str, Any]]
    role_counts: Mapping[str, int]

    @property
    def frame_count(self) -> int:
        return sum(len(sequence.frames) for sequence in self.sequences)

    @property
    def bbox_count(self) -> int:
        return sum(
            len(frame.boxes)
            for sequence in self.sequences
            for frame in sequence.frames
        )


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except FileNotFoundError as exc:
        raise PreparationError(f"required file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PreparationError(f"invalid JSON in {path}: {exc}") from exc


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def _as_finite_number(value: Any, context: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{context} must be numeric, got {value!r}",
    )
    number = float(value)
    _require(math.isfinite(number), f"{context} must be finite, got {value!r}")
    return number


def _normalise_sphere(payload: Any, context: str) -> dict[str, Any]:
    _require(isinstance(payload, dict), f"{context} must be a JSON object")
    _require(set(payload) == {"sphere"}, f"{context} must contain only 'sphere'")
    sphere = payload["sphere"]
    _require(isinstance(sphere, dict), f"{context}.sphere must be an object")
    _require(
        set(sphere) == {"center", "radius"},
        f"{context}.sphere must contain center and radius",
    )
    center = sphere["center"]
    _require(
        isinstance(center, list) and len(center) == 3,
        f"{context}.sphere.center must contain three coordinates",
    )
    normalised_center = [
        _as_finite_number(value, f"{context}.sphere.center[{index}]")
        for index, value in enumerate(center)
    ]
    radius = _as_finite_number(sphere["radius"], f"{context}.sphere.radius")
    _require(radius > 0.0, f"{context}.sphere.radius must be positive")
    return {"sphere": {"center": normalised_center, "radius": radius}}


def _validate_sequence_list(
    sequences_info: Any, expectations: Expectations
) -> tuple[str, list[dict[str, Any]]]:
    _require(isinstance(sequences_info, dict), "sequences_info.json must be an object")
    version = sequences_info.get("version")
    _require(isinstance(version, str), "sequences_info.json has no string version")
    entries = sequences_info.get("test")
    _require(isinstance(entries, list), "sequences_info.json has no test list")
    names = tuple(entry.get("name") for entry in entries if isinstance(entry, dict))
    _require(
        names == expectations.sequence_names,
        "test sequence list/order differs from the expected SoccerNet-GSR test split",
    )
    for index, entry in enumerate(entries):
        _require(entry.get("id") == index, f"test sequence id at index {index} is invalid")
        _require(
            entry.get("n_frames") == expectations.frames_per_sequence,
            f"{entry.get('name')} has unexpected n_frames={entry.get('n_frames')}",
        )
    return version, entries


def _validate_bbox(
    annotation: Mapping[str, Any],
    width: int,
    height: int,
    context: str,
) -> tuple[float, float, float, float, float]:
    bbox = annotation.get("bbox_image")
    _require(isinstance(bbox, dict), f"{context} has no bbox_image object")
    _require(
        {"x", "y", "w", "h"}.issubset(bbox),
        f"{context}.bbox_image lacks x/y/w/h",
    )
    x = _as_finite_number(bbox["x"], f"{context}.bbox_image.x")
    y = _as_finite_number(bbox["y"], f"{context}.bbox_image.y")
    box_width = _as_finite_number(bbox["w"], f"{context}.bbox_image.w")
    box_height = _as_finite_number(bbox["h"], f"{context}.bbox_image.h")
    _require(box_width > 0 and box_height > 0, f"{context} has a non-positive bbox")
    x2, y2 = x + box_width, y + box_height
    _require(
        0 <= x < x2 <= width and 0 <= y < y2 <= height,
        f"{context} bbox {(x, y, x2, y2)} is outside {width}x{height}",
    )
    # BroadTrack consumes absolute pixel LTRB and ignores the confidence value.
    return (x, y, x2, y2, 1.0)


def _load_sequence(
    dataset_root: Path,
    name: str,
    version: str,
    expected_game_id: str,
    expectations: Expectations,
) -> tuple[SequenceData, Counter[str]]:
    sequence_root = dataset_root / name
    labels_path = sequence_root / "Labels-GameState.json"
    labels = _load_json(labels_path)
    _require(isinstance(labels, dict), f"{labels_path} must contain an object")
    info = labels.get("info")
    _require(isinstance(info, dict), f"{labels_path} has no info object")
    _require(info.get("version") == version, f"{name} label version mismatch")
    _require(info.get("name") == name, f"{name} label name mismatch")
    _require(str(info.get("id")) == name.removeprefix("SNGS-"), f"{name} id mismatch")
    _require(str(info.get("game_id")) == expected_game_id, f"{name} game_id mismatch")
    _require(
        info.get("seq_length") == expectations.frames_per_sequence,
        f"{name} seq_length mismatch",
    )
    _require(info.get("im_dir") == "img1", f"{name} im_dir must be img1")
    _require(info.get("im_ext") == ".jpg", f"{name} im_ext must be .jpg")

    images = labels.get("images")
    annotations = labels.get("annotations")
    _require(isinstance(images, list), f"{labels_path} has no images list")
    _require(isinstance(annotations, list), f"{labels_path} has no annotations list")
    expected_names = [
        f"{frame_number:06d}.jpg"
        for frame_number in range(1, expectations.frames_per_sequence + 1)
    ]
    actual_names = [image.get("file_name") for image in images if isinstance(image, dict)]
    _require(actual_names == expected_names, f"{name} image filenames/order are misaligned")

    frames_directory = sequence_root / "img1"
    _require(frames_directory.is_dir(), f"missing frames directory: {frames_directory}")
    disk_jpegs = sorted(path.name for path in frames_directory.glob("*.jpg"))
    _require(disk_jpegs == expected_names, f"{name}/img1 JPEG set is incomplete or has extras")

    images_by_id: dict[str, dict[str, Any]] = {}
    for index, image in enumerate(images):
        _require(isinstance(image, dict), f"{name} image[{index}] must be an object")
        image_id = str(image.get("image_id"))
        _require(image_id not in images_by_id, f"{name} duplicate image_id {image_id}")
        _require(image.get("width") == expectations.image_width, f"{name} width mismatch")
        _require(image.get("height") == expectations.image_height, f"{name} height mismatch")
        image_path = frames_directory / image["file_name"]
        _require(image_path.is_file(), f"missing image: {image_path}")
        _require(image_path.stat().st_size > 0, f"empty image: {image_path}")
        images_by_id[image_id] = image

    boxes_by_image: dict[str, list[HumanBox]] = defaultdict(list)
    role_counts: Counter[str] = Counter()
    for index, annotation in enumerate(annotations):
        _require(isinstance(annotation, dict), f"{name} annotation[{index}] must be an object")
        if annotation.get("supercategory") != "object":
            continue
        attributes = annotation.get("attributes")
        _require(isinstance(attributes, dict), f"{name} annotation[{index}] lacks attributes")
        role = attributes.get("role")
        _require(isinstance(role, str), f"{name} annotation[{index}] has no string role")
        expected_category = ROLE_CATEGORY_IDS.get(role)
        _require(expected_category is not None, f"{name} has unsupported object role {role!r}")
        _require(
            annotation.get("category_id") == expected_category,
            f"{name} annotation[{index}] category/role mismatch",
        )
        image_id = str(annotation.get("image_id"))
        _require(image_id in images_by_id, f"{name} annotation references unknown image {image_id}")
        if role == "ball":
            continue
        _require(role in HUMAN_ROLES, f"{name} has unsupported human role {role!r}")
        image = images_by_id[image_id]
        coordinates = _validate_bbox(
            annotation,
            expectations.image_width,
            expectations.image_height,
            f"{name} annotation[{index}]",
        )
        boxes_by_image[image_id].append(HumanBox(role=role, coordinates=coordinates))
        role_counts[role] += 1

    frames: list[FrameData] = []
    for image in images:
        image_id = str(image["image_id"])
        boxes = tuple(boxes_by_image.get(image_id, ()))
        _require(boxes, f"{name}/{image['file_name']} has no annotated humans")
        frames.append(
            FrameData(
                image_id=image_id,
                file_name=image["file_name"],
                image_path=(frames_directory / image["file_name"]).resolve(),
                boxes=boxes,
            )
        )
    return (
        SequenceData(
            name=name,
            game_id=expected_game_id,
            labels_path=labels_path.resolve(),
            frames_directory=frames_directory.resolve(),
            frames=tuple(frames),
        ),
        role_counts,
    )


def load_source_data(
    dataset_test_root: Path,
    match_info_path: Path,
    tripod_info_path: Path,
    expectations: Expectations = Expectations(),
) -> SourceData:
    """Load and strictly validate the dataset and issue #1 attachments."""

    dataset_root = dataset_test_root.expanduser().resolve()
    _require(dataset_root.is_dir(), f"dataset test root is not a directory: {dataset_root}")
    sequences_info = _load_json(dataset_root / "sequences_info.json")
    version, entries = _validate_sequence_list(sequences_info, expectations)
    match_info = _load_json(match_info_path.expanduser().resolve())
    tripod_info = _load_json(tripod_info_path.expanduser().resolve())
    _require(isinstance(match_info, dict), "match_info.json must contain an object")
    _require(isinstance(tripod_info, dict), "tripod_info JSON must contain an object")

    sequences: list[SequenceData] = []
    all_role_counts: Counter[str] = Counter()
    sequences_by_game: dict[str, list[str]] = defaultdict(list)
    normalised_sequence_tripods: dict[str, dict[str, Any]] = {}
    for entry in entries:
        name = entry["name"]
        _require(name in match_info, f"match_info has no entry for {name}")
        game_id = str(match_info[name])
        _require(name in tripod_info, f"tripod_info has no entry for {name}")
        normalised_sequence_tripods[name] = _normalise_sphere(
            tripod_info[name], f"tripod_info[{name!r}]"
        )
        sequence, counts = _load_sequence(
            dataset_root, name, version, game_id, expectations
        )
        sequences.append(sequence)
        all_role_counts.update(counts)
        sequences_by_game[game_id].append(name)

    game_tripods: dict[str, dict[str, Any]] = {}
    for game_id, names in sequences_by_game.items():
        reference = normalised_sequence_tripods[names[0]]
        for name in names[1:]:
            _require(
                normalised_sequence_tripods[name] == reference,
                f"tripod differs within game {game_id}: {names[0]} vs {name}",
            )
        game_tripods[game_id] = reference

    return SourceData(
        dataset_root=dataset_root,
        dataset_version=version,
        sequences=tuple(sequences),
        tripods=game_tripods,
        role_counts=dict(sorted(all_role_counts.items())),
    )


def _bbox_payload(frame: FrameData) -> dict[str, Any]:
    return {
        "image": str(frame.image_path),
        "bboxes": [list(box.coordinates) for box in frame.boxes],
    }


def _game_sort_key(game_id: str) -> tuple[int, Any]:
    return (0, int(game_id)) if game_id.isdigit() else (1, game_id)


def build_manifest(source: SourceData) -> dict[str, Any]:
    sequences_by_game: dict[str, list[str]] = defaultdict(list)
    for sequence in source.sequences:
        sequences_by_game[sequence.game_id].append(sequence.name)
    games = {
        game_id: {
            "sequences": sequences_by_game[game_id],
            "tripod": f"tripods/game-{game_id}.json",
        }
        for game_id in sorted(sequences_by_game, key=_game_sort_key)
    }
    sequences = {
        sequence.name: {
            "frames": len(sequence.frames),
            "frames_directory": str(sequence.frames_directory),
            "game_id": sequence.game_id,
            "human_bboxes": f"human-bboxes/{sequence.name}",
            "labels": str(sequence.labels_path),
        }
        for sequence in source.sequences
    }
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "dataset": {
            "root": str(source.dataset_root),
            "split": "test",
            "version": source.dataset_version,
        },
        "schema": {
            "bbox_coordinates": "[x1, y1, x2, y2, confidence]",
            "bbox_space": "absolute image pixels",
            "broadtrack_expansion_pixels": 100,
            "selection": "object annotations except role=ball",
        },
        "counts": {
            "bboxes": source.bbox_count,
            "frames": source.frame_count,
            "games": len(games),
            "human_roles": dict(source.role_counts),
            "sequences": len(source.sequences),
        },
        "games": games,
        "sequences": sequences,
    }


def _ensure_safe_artifact_root(dataset_root: Path, artifact_root: Path) -> Path:
    output = artifact_root.expanduser().resolve()
    _require(
        output != dataset_root and dataset_root not in output.parents,
        "artifact root must not be the dataset root or one of its descendants",
    )
    _require(not output.exists(), f"artifact root already exists: {output}")
    return output


def _validate_exact_names(directory: Path, expected: Iterable[str], context: str) -> None:
    _require(directory.is_dir(), f"missing {context} directory: {directory}")
    actual = sorted(path.name for path in directory.iterdir())
    wanted = sorted(expected)
    _require(actual == wanted, f"{context} file set mismatch")


def validate_artifacts(source: SourceData, artifact_root: Path) -> None:
    """Validate every generated JSON against its source frame and annotation."""

    root = artifact_root.expanduser().resolve()
    _require(root.is_dir(), f"artifact root is not a directory: {root}")
    _validate_exact_names(
        root, ("human-bboxes", "manifest.json", "tripods"), "artifact root"
    )

    manifest = _load_json(root / "manifest.json")
    _require(manifest == build_manifest(source), "manifest.json differs from source data")

    tripod_directory = root / "tripods"
    expected_tripod_names = [f"game-{game_id}.json" for game_id in source.tripods]
    _validate_exact_names(tripod_directory, expected_tripod_names, "tripod")
    for game_id, expected in source.tripods.items():
        actual = _load_json(tripod_directory / f"game-{game_id}.json")
        _require(actual == expected, f"game {game_id} tripod artifact mismatch")

    bboxes_root = root / "human-bboxes"
    _validate_exact_names(
        bboxes_root, (sequence.name for sequence in source.sequences), "bbox sequence"
    )
    checked_frames = 0
    checked_boxes = 0
    for sequence in source.sequences:
        sequence_bbox_root = bboxes_root / sequence.name
        expected_files = [Path(frame.file_name).with_suffix(".json").name for frame in sequence.frames]
        _validate_exact_names(sequence_bbox_root, expected_files, f"{sequence.name} bbox")
        for frame in sequence.frames:
            path = sequence_bbox_root / Path(frame.file_name).with_suffix(".json")
            payload = _load_json(path)
            _require(
                isinstance(payload, dict) and set(payload) == {"image", "bboxes"},
                f"invalid bbox schema in {path}",
            )
            expected_payload = _bbox_payload(frame)
            _require(payload == expected_payload, f"bbox/frame alignment mismatch in {path}")
            checked_frames += 1
            checked_boxes += len(payload["bboxes"])
    _require(checked_frames == source.frame_count, "validated frame count mismatch")
    _require(checked_boxes == source.bbox_count, "validated bbox count mismatch")


def prepare_artifacts(source: SourceData, artifact_root: Path) -> Path:
    """Atomically generate and validate the artifact tree."""

    output = _ensure_safe_artifact_root(source.dataset_root, artifact_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        for game_id in sorted(source.tripods, key=_game_sort_key):
            _write_json(staging / "tripods" / f"game-{game_id}.json", source.tripods[game_id])
        for sequence in source.sequences:
            for frame in sequence.frames:
                output_name = Path(frame.file_name).with_suffix(".json")
                _write_json(
                    staging / "human-bboxes" / sequence.name / output_name,
                    _bbox_payload(frame),
                )
        _write_json(staging / "manifest.json", build_manifest(source))
        validate_artifacts(source, staging)
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _summary(source: SourceData, artifact_root: Path, action: str) -> str:
    return (
        f"{action} {len(source.sequences)} sequences, {source.frame_count} frames, "
        f"{source.bbox_count} human boxes, and {len(source.tripods)} game tripods "
        f"at {artifact_root}"
    )


def create_parser() -> argparse.ArgumentParser:
    repro_root = Path(__file__).resolve().parent

    class HelpFormatter(
        argparse.ArgumentDefaultsHelpFormatter,
        argparse.RawDescriptionHelpFormatter,
    ):
        pass

    parser = argparse.ArgumentParser(
        description=(
            "Prepare BroadTrack inputs from the SoccerNet-GSR 2024 test split. "
            "The dataset is read-only; all generated files go to --artifact-root."
        ),
        formatter_class=HelpFormatter,
        epilog=(
            "Example:\n"
            "  python repro/prepare_soccernet.py --dataset-test-root "
            "/data/SoccerNetGS/test --artifact-root repro/artifacts/soccernet-test\n\n"
            "Use --check with the same arguments to validate an existing artifact tree "
            "without writing files."
        ),
    )
    parser.add_argument(
        "--dataset-test-root",
        "--dataset-test",
        required=True,
        type=Path,
        help="directory containing sequences_info.json and SNGS-* test folders",
    )
    parser.add_argument(
        "--match-info",
        type=Path,
        default=repro_root / "vendor" / "issue-1" / "match_info.json",
        help="match_info.json attachment from BroadTrack issue #1",
    )
    parser.add_argument(
        "--tripod-info",
        type=Path,
        default=repro_root / "vendor" / "issue-1" / "tripod_info_oleg.json",
        help="tripod_info_oleg.json attachment from BroadTrack issue #1",
    )
    parser.add_argument(
        "--artifact-root",
        "--output",
        required=True,
        type=Path,
        help="new output directory for manifest, tripods, and human bbox JSONs",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the source and an existing artifact root without writing",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        source = load_source_data(
            args.dataset_test_root, args.match_info, args.tripod_info
        )
        if args.check:
            validate_artifacts(source, args.artifact_root)
            print(_summary(source, args.artifact_root.resolve(), "Validated"))
        else:
            output = prepare_artifacts(source, args.artifact_root)
            print(_summary(source, output, "Prepared and validated"))
    except (PreparationError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
