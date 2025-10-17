from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import trimesh


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute bounding boxes and centers of mass for GLB meshes."
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Folder that contains one or more .glb files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Path to the JSON file to write. Defaults to <directory>/glb_stats.json."
        ),
    )
    return parser.parse_args(argv)


def load_scene(path: Path) -> trimesh.Scene:
    scene = trimesh.load(path, force="scene")
    if not isinstance(scene, trimesh.Scene):
        scene = trimesh.Scene(scene)
    if not scene.geometry:
        raise ValueError(f"No geometry found in GLB file: {path}")
    return scene


def safe_vector(values: Iterable[float] | None) -> list[float] | None:
    if values is None:
        return None
    vector: list[float] = []
    try:
        for value in values:
            vector.append(float(value))
    except TypeError:
        return None
    if len(vector) != 3:
        return None
    if not all(math.isfinite(component) for component in vector):
        return None
    return vector


def round_vector(values: Iterable[float], digits: int = 5) -> list[float]:
    rounded = [round(float(value), digits) for value in values]
    if len(rounded) != 3:
        raise ValueError("Expected a 3D vector")
    return rounded


def vector_to_point(values: Iterable[float]) -> dict[str, float]:
    rounded = round_vector(values)
    axes = ("x", "y", "z")
    return {axis: rounded[idx] for idx, axis in enumerate(axes)}


def vector_to_axis_bounds(
    min_values: Iterable[float], max_values: Iterable[float]
) -> dict[str, dict[str, float]]:
    min_vector = round_vector(min_values)
    max_vector = round_vector(max_values)
    axes = ("x", "y", "z")
    return {
        axis: {"min": min_vector[idx], "max": max_vector[idx]}
        for idx, axis in enumerate(axes)
    }


def collect_stats(path: Path) -> dict:
    scene = load_scene(path)
    bounds = scene.bounds
    if bounds is None:
        raise ValueError(f"Unable to compute bounds for: {path}")
    bbox_min_raw, bbox_max_raw = bounds
    center_mass_vector = safe_vector(scene.center_mass)
    if center_mass_vector is None:
        center_mass_vector = safe_vector(scene.centroid)
    if center_mass_vector is None:
        center_mass_vector = [
            (float(bbox_min_raw[idx]) + float(bbox_max_raw[idx])) / 2.0
            for idx in range(3)
        ]
    min_vector = round_vector(bbox_min_raw)
    max_vector = round_vector(bbox_max_raw)
    extents = [max_vector[idx] - min_vector[idx] for idx in range(3)]
    bbox_area = round(extents[0] * extents[1], 5)
    try:
        volume = round(float(scene.volume), 5)
    except Exception as exc:  # pragma: no cover - trimesh volume failures
        raise ValueError(f"Unable to compute volume for: {path}") from exc
    return {
        "name": path.stem,
        "bounding_box": vector_to_axis_bounds(bbox_min_raw, bbox_max_raw),
        "bounding_box_area": bbox_area,
        "volume": volume,
        "center_of_mass": vector_to_point(center_mass_vector),
    }


def render_progress(current: int, total: int, name: str, *, bar_length: int = 30) -> None:
    if total <= 0:
        total = 1
    current = min(current, total)
    progress = current / total
    filled_length = int(bar_length * progress)
    bar = "#" * filled_length + "-" * (bar_length - filled_length)
    display_name = name[:25]
    sys.stdout.write(f"\r[{bar}] {current}/{total} {display_name.ljust(25)}")
    sys.stdout.flush()


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        raise SystemExit(f"Not a directory: {directory}")

    output_path = args.output
    if output_path is None:
        output_path = directory / "glb_stats.json"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    glb_files = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".glb"
    )

    if not glb_files:
        raise SystemExit(f"No .glb files found in: {directory}")

    results = []
    total_files = len(glb_files)
    for index, glb_path in enumerate(glb_files, start=1):
        try:
            stats = collect_stats(glb_path)
        except Exception as exc:  # pragma: no cover - surfaces errors to the caller
            sys.stdout.write("\n")
            raise SystemExit(f"Failed to process {glb_path.name}: {exc}") from exc
        results.append(stats)
        render_progress(index, total_files, glb_path.name)

    sys.stdout.write("\n")
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(results, outfile, indent=2)

    print(f"Wrote statistics for {len(results)} file(s) to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
