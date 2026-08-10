#!/usr/bin/env python3
"""Visualize native BroadTrack camera JSON on the frames it was estimated from.

Three modes: the pitch model projected onto the frame, the frame warped onto the
pitch plane, or both side by side.

Example::

    python repro/visualize.py \
      --frames /data/SNGS-116/img1 \
      --cameras /results/SNGS-116/broadtrack.json \
      --out /results/SNGS-116-side.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Importing the evaluator also puts sn_calibration_baseline on sys.path.
import evaluate_soccernet as evaluator  # noqa: E402

from sn_calibration_baseline.camera import Camera  # noqa: E402
from sn_calibration_baseline.soccerpitch import SoccerPitch  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

SAMPLE_STEP_M = 0.2
NEAR_PLANE_M = 0.5
# Samples further out than this many image half-diagonals are dropped: the
# radial polynomial is only meaningful near the image.
MAX_RADIUS_FACTOR = 3.0

LINE_COLOR = (0, 160, 255)
LINE_THICKNESS = 3
LINE_ALPHA = 0.85

TOPDOWN_SCALE = 10.0  # pixels per metre
TOPDOWN_MARGIN_M = 5.0
TOPDOWN_GRASS = (60, 125, 60)
TOPDOWN_MARKINGS = (245, 245, 245)
TOPDOWN_ALPHA = 0.85


def load_cameras(path: Path) -> dict[str, dict[str, Any]]:
    """Map frame basename to the native BroadTrack camera dict."""
    with path.open(encoding="utf-8") as stream:
        raw = json.load(stream)

    cameras: dict[str, dict[str, Any]] = {}
    for key, entry in raw.items():
        camera = entry.get("cp", entry) if isinstance(entry, Mapping) else None
        if not isinstance(camera, Mapping):
            raise SystemExit(f"{path}: entry {key!r} has no camera parameters")
        cameras[Path(key).name] = dict(camera)
    return cameras


def build_camera(native: Mapping[str, Any], frame_width: int) -> Camera:
    """Convert a native BroadTrack camera into an sn-calibration Camera."""
    width = float(native["sensorResolutionWidthPixels"])
    height = float(native["sensorResolutionHeightPixels"])

    # The evaluator pins its protocol to 1920x1080; visualization does not care.
    paper = evaluator.PAPER_WIDTH, evaluator.PAPER_HEIGHT
    evaluator.PAPER_WIDTH, evaluator.PAPER_HEIGHT = width, height
    try:
        parameters = evaluator.broadtrack_camera_to_soccernet(native)
    finally:
        evaluator.PAPER_WIDTH, evaluator.PAPER_HEIGHT = paper

    camera = Camera(width, height)
    camera.from_json_parameters(parameters)
    if frame_width != camera.image_width:
        camera.scale_resolution(frame_width / camera.image_width)
    return camera


def pitch_polylines() -> dict[str, np.ndarray]:
    """Dense 3D samples of every pitch element, in metres."""
    samples = SoccerPitch().sample_field_points(SAMPLE_STEP_M, SAMPLE_STEP_M)
    polylines = {
        name: np.asarray(points, dtype=float) for name, points in samples.items()
    }
    # sample_field_points stops one step short of closing the centre circle.
    circle = polylines["Circle central"]
    polylines["Circle central"] = np.vstack([circle, circle[:1]])
    return polylines


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Start/stop indices of the runs of True values in a boolean mask."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2], edges[1::2]))


def project_polyline(camera: Camera, points: np.ndarray) -> list[np.ndarray]:
    """Project a 3D polyline with the full camera model, distortion included.

    Samples behind the camera or far outside the frame break the polyline into
    several strokes rather than producing a spurious straight jump.
    """
    local = camera.rotation @ (points - camera.position).T
    depth = local[2]
    in_front = depth > NEAR_PLANE_M
    normalized = np.zeros((2, depth.size))
    np.divide(local[:2], depth, out=normalized, where=in_front)

    half_diagonal = np.hypot(
        camera.image_width / 2 / camera.xfocal_length,
        camera.image_height / 2 / camera.yfocal_length,
    )
    usable = in_front & (np.hypot(*normalized) <= MAX_RADIUS_FACTOR * half_diagonal)

    focal = np.array([camera.xfocal_length, camera.yfocal_length])
    center = np.asarray(camera.principal_point, dtype=float)
    strokes = []
    for start, stop in _true_runs(usable):
        if stop - start < 2:
            continue
        distorted = np.array(
            [camera.distort(point) for point in normalized[:, start:stop].T]
        )
        strokes.append(np.rint(distorted * focal + center).astype(np.int32))
    return strokes


def render_overlay(
    frame: np.ndarray, camera: Camera, polylines: Mapping[str, np.ndarray]
) -> np.ndarray:
    """Draw the projected pitch model over the frame."""
    canvas = frame.copy()
    for points in polylines.values():
        strokes = project_polyline(camera, points)
        if strokes:
            cv2.polylines(
                canvas, strokes, False, LINE_COLOR, LINE_THICKNESS, cv2.LINE_AA
            )
    return cv2.addWeighted(canvas, LINE_ALPHA, frame, 1.0 - LINE_ALPHA, 0.0)


@dataclass(frozen=True)
class TopDownView:
    """Template of the pitch seen from above, plus the metric grid behind it."""

    size: tuple[int, int]
    matrix: np.ndarray  # pitch metres -> template pixels
    template: np.ndarray
    x_meters: np.ndarray
    y_meters: np.ndarray


def build_topdown_view(polylines: Mapping[str, np.ndarray]) -> TopDownView:
    pitch = SoccerPitch()
    width = int(round((pitch.PITCH_LENGTH + 2 * TOPDOWN_MARGIN_M) * TOPDOWN_SCALE))
    height = int(round((pitch.PITCH_WIDTH + 2 * TOPDOWN_MARGIN_M) * TOPDOWN_SCALE))
    origin_x = (pitch.PITCH_LENGTH / 2 + TOPDOWN_MARGIN_M) * TOPDOWN_SCALE
    origin_y = (pitch.PITCH_WIDTH / 2 + TOPDOWN_MARGIN_M) * TOPDOWN_SCALE
    matrix = np.array(
        [
            [TOPDOWN_SCALE, 0.0, origin_x],
            [0.0, TOPDOWN_SCALE, origin_y],
            [0.0, 0.0, 1.0],
        ]
    )

    template = np.full((height, width, 3), TOPDOWN_GRASS, dtype=np.uint8)
    for name, points in polylines.items():
        if name.startswith("Goal"):  # posts and crossbars are not on the ground
            continue
        pixels = points[:, :2] * TOPDOWN_SCALE + matrix[:2, 2]
        cv2.polylines(
            template,
            [np.rint(pixels).astype(np.int32)],
            False,
            TOPDOWN_MARKINGS,
            2,
            cv2.LINE_AA,
        )

    columns, rows = np.meshgrid(np.arange(width), np.arange(height))
    x_meters = (columns - matrix[0, 2]) / TOPDOWN_SCALE
    y_meters = (rows - matrix[1, 2]) / TOPDOWN_SCALE
    return TopDownView((width, height), matrix, template, x_meters, y_meters)


def _opencv_distortion(camera: Camera) -> np.ndarray:
    """Camera.distort as an OpenCV rational+prism coefficient vector."""
    return np.array(
        [
            camera.radial_distortion[0],
            camera.radial_distortion[1],
            camera.tangential_disto[0],
            camera.tangential_disto[1],
            *camera.radial_distortion[2:],
            *camera.thin_prism_disto,
        ]
    )


def render_topdown(
    frame: np.ndarray, camera: Camera, view: TopDownView
) -> np.ndarray:
    """Warp the frame onto the pitch plane and blend it over the template."""
    undistorted = cv2.undistort(frame, camera.calibration, _opencv_distortion(camera))
    homography = view.matrix @ np.linalg.inv(camera.to_homography())
    warped = cv2.warpPerspective(undistorted, homography, view.size)
    # Eroded, otherwise the black border bleeds into the edge of the warp.
    covered = cv2.erode(
        cv2.warpPerspective(
            np.full(frame.shape[:2], 255, np.uint8), homography, view.size
        ),
        np.ones((5, 5), np.uint8),
    )

    # The homography also maps the plane behind the camera into the template.
    forward = camera.rotation[2]
    depth = (
        forward[0] * (view.x_meters - camera.position[0])
        + forward[1] * (view.y_meters - camera.position[1])
        - forward[2] * camera.position[2]
    )

    blended = cv2.addWeighted(
        warped, TOPDOWN_ALPHA, view.template, 1.0 - TOPDOWN_ALPHA, 0.0
    )
    image = view.template.copy()
    visible = (covered > 0) & (depth > NEAR_PLANE_M)
    image[visible] = blended[visible]
    return image


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    height = left.shape[0]
    width = int(round(right.shape[1] * height / right.shape[0]))
    return np.hstack([left, cv2.resize(right, (width, height))])


def render(
    frame: np.ndarray,
    camera: Camera,
    polylines: Mapping[str, np.ndarray],
    view: TopDownView,
    mode: str,
) -> np.ndarray:
    if mode == "overlay":
        return render_overlay(frame, camera, polylines)
    if mode == "topdown":
        return render_topdown(frame, camera, view)
    return side_by_side(
        render_overlay(frame, camera, polylines), render_topdown(frame, camera, view)
    )


def select_frames(directory: Path, start: int, count: int | None) -> list[Path]:
    """Frames sorted by name, sliced by 1-based index."""
    frames = sorted(
        path
        for path in directory.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()
    )
    first = max(start - 1, 0)
    return frames[first : None if count is None else first + count]


def open_video(path: Path, fps: float, image: np.ndarray) -> cv2.VideoWriter:
    height, width = image.shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise SystemExit(f"Cannot open {path} for writing")
    return writer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frames", type=Path, required=True, help="directory of frames")
    parser.add_argument("--cameras", type=Path, required=True, help="BroadTrack JSON")
    parser.add_argument(
        "--out", type=Path, required=True, help="output directory, or a .mp4 file"
    )
    parser.add_argument(
        "--mode", choices=("overlay", "topdown", "side"), default="side"
    )
    parser.add_argument("--fps", type=float, default=25.0, help="video frame rate")
    parser.add_argument("--start", type=int, default=1, help="1-based first frame")
    parser.add_argument("--count", type=int, default=None, help="number of frames")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    frames = select_frames(args.frames, args.start, args.count)
    if not frames:
        raise SystemExit(f"No frames selected in {args.frames}")
    cameras = load_cameras(args.cameras)
    polylines = pitch_polylines()
    view = build_topdown_view(polylines)

    to_video = args.out.suffix.lower() == ".mp4"
    (args.out.parent if to_video else args.out).mkdir(parents=True, exist_ok=True)

    writer = None
    written = 0
    skipped = 0
    for path in frames:
        native = cameras.get(path.name)
        frame = cv2.imread(str(path)) if native is not None else None
        if frame is None:
            skipped += 1
            continue
        camera = build_camera(native, frame.shape[1])
        image = render(frame, camera, polylines, view, args.mode)
        if to_video:
            writer = writer or open_video(args.out, args.fps, image)
            writer.write(image)
        else:
            cv2.imwrite(str(args.out / f"{path.stem}.png"), image)
        written += 1

    if writer is not None:
        writer.release()
    if skipped:
        print(
            f"skipped {skipped} frame(s): no camera in {args.cameras}, or unreadable",
            file=sys.stderr,
        )
    print(f"wrote {written} frame(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
