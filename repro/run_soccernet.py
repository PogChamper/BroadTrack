#!/usr/bin/env python3
"""Run BroadTrack on the SoccerNet-GSR test sequences prepared earlier.

Each sequence runs in its own container with the dataset frames and the
prepared artifacts mounted read-only.  Results are written to
<output-root>/<sequence>/broadtrack.json next to the container log.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


def load_manifest(artifact_root: Path) -> dict[str, Any]:
    with (artifact_root / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format_version") != 1:
        raise SystemExit(f"unsupported artifact format in {artifact_root}")
    return manifest


def resolve_image(image: str) -> str:
    """Pin the tag to an image ID so a rebuild mid-run cannot change the code."""
    inspect = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True, text=True,
    )
    if inspect.returncode != 0:
        raise SystemExit(f"docker image {image} not found")
    return inspect.stdout.strip()


def mount(source: Path, target: str, read_only: bool) -> list[str]:
    option = f"type=bind,source={source},target={target}"
    return ["--mount", f"{option},readonly" if read_only else option]


def build_command(
    image: str, frames: Path, bboxes: Path, tripod: Path, result_directory: Path
) -> list[str]:
    return [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--network", "none",
        "--user", f"{os.getuid()}:{os.getgid()}",
        *mount(frames, "/input/frames", True),
        *mount(bboxes, "/input/human-bboxes", True),
        *mount(tripod, "/input/tripod.json", True),
        *mount(result_directory, "/output", False),
        image,
        "broadtrack",
        "--f", "/input/frames",
        "--b", "/input/human-bboxes",
        "--t", "/input/tripod.json",
        "--o", "/output/broadtrack.partial.json",
    ]


def all_finite(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(all_finite(item) for item in value)
    return True


def check_output(path: Path, frames: int) -> str | None:
    """Return None when the camera trajectory looks complete, else a reason."""

    try:
        with path.open(encoding="utf-8") as stream:
            cameras = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        return f"unreadable output: {exc}"
    if not isinstance(cameras, dict):
        return "output is not a JSON object"
    if len(cameras) != frames:
        return f"{len(cameras)} camera records, expected {frames}"
    if not all_finite(cameras):
        return "output contains non-finite numbers"
    return None


def run_sequence(command: Sequence[str], result_directory: Path) -> int:
    with (result_directory / "run.log").open("w", encoding="utf-8") as log:
        log.write(f"{shlex.join(command)}\n")
        log.flush()
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, text=True
        )
    return completed.returncode


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="broadtrack", help="BroadTrack Docker image")
    parser.add_argument(
        "--dataset-test-root", required=True, type=Path, help="SoccerNet-GSR test split"
    )
    parser.add_argument(
        "--artifact-root", required=True, type=Path, help="prepare_soccernet.py output"
    )
    parser.add_argument(
        "--output-root", required=True, type=Path, help="directory for the results"
    )
    parser.add_argument("--sequences", nargs="+", help="run only these sequences")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_test_root.expanduser().resolve()
    artifact_root = args.artifact_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest = load_manifest(artifact_root)
    image = resolve_image(args.image)
    sequences = sorted(manifest["sequences"])
    if args.sequences is not None:
        unknown = sorted(set(args.sequences) - set(sequences))
        if unknown:
            raise SystemExit(f"unknown sequences: {', '.join(unknown)}")
        sequences = sorted(set(args.sequences))

    done = skipped = failed = 0
    for name in sequences:
        entry = manifest["sequences"][name]
        result_directory = output_root / name
        if (result_directory / "broadtrack.json").is_file():
            print(f"{name}: skipped, result already exists")
            skipped += 1
            continue
        result_directory.mkdir(parents=True, exist_ok=True)
        command = build_command(
            image,
            dataset_root / name / "img1",
            artifact_root / entry["human_bboxes"],
            artifact_root / manifest["games"][entry["game_id"]]["tripod"],
            result_directory,
        )
        started = time.monotonic()
        exit_code = run_sequence(command, result_directory)
        elapsed = time.monotonic() - started
        partial = result_directory / "broadtrack.partial.json"
        if exit_code != 0:
            reason = f"docker exited with {exit_code}"
        else:
            reason = check_output(partial, entry["frames"])
        if reason is None:
            partial.rename(result_directory / "broadtrack.json")
            print(f"{name}: ok, {entry['frames']} frames in {elapsed:.0f}s")
            done += 1
        else:
            print(f"{name}: failed, {reason}", file=sys.stderr)
            failed += 1
    print(f"{done} completed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
