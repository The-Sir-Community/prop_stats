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
            "Path to the JSON file to write. Defaults to <directory>/prop_stats.json."
        ),
    )
    parser.add_argument(
        "-a",
        "--asset-types",
        type=Path,
        help="Path to the asset_types.json file with additional metadata.",
    )
    return parser.parse_args(argv)


def load_asset_types(path: Path) -> dict[str, dict]:
    """Load and parse asset_types.json file into a dictionary keyed by model name."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    asset_map = {}
    for asset in data.get("AssetTypes", []):
        model_name = asset.get("type")
        if not model_name:
            continue

        # Extract constants into a dict for easier lookup
        constants = {}
        for constant in asset.get("constants", []):
            constants[constant["name"]] = constant["value"]

        # Build the asset metadata
        asset_info = {
            "name": model_name,
            "path": asset.get("directory", ""),
            "physicsCost": constants.get("physicsCost"),
            "category": constants.get("category"),
            "levelRestrictions": asset.get("levelRestrictions", []),
        }

        asset_map[model_name] = asset_info

    return asset_map


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


def detect_mesh_validity_issues(scene: trimesh.Scene, bbox_volume: float) -> dict:
    """
    Detect mesh validity issues like large holes or missing faces.
    Returns a dict with validity metrics to help identify broken/invalid meshes.

    The goal is to flag meshes that are broken (e.g., missing bottom face on a cylinder)
    while NOT flagging valid meshes with architectural features (doors, windows, etc.)
    """
    # Combine all geometries in the scene
    combined_mesh = None
    for geometry in scene.geometry.values():
        if isinstance(geometry, trimesh.Trimesh):
            if combined_mesh is None:
                combined_mesh = geometry.copy()
            else:
                combined_mesh = trimesh.util.concatenate([combined_mesh, geometry])

    if combined_mesh is None or not isinstance(combined_mesh, trimesh.Trimesh):
        return {
            "is_watertight": True,
            "triangle_count": 0,
            "volume_ratio": None,
            "is_potentially_invalid": False
        }

    # Check if mesh is watertight (completely closed, no holes)
    is_watertight = combined_mesh.is_watertight

    # Get triangle/polygon count
    triangle_count = len(combined_mesh.faces)

    # Calculate volume ratio (mesh volume / bounding box volume)
    # Low ratio might indicate missing geometry
    try:
        mesh_volume = float(combined_mesh.volume)
        volume_ratio = round(mesh_volume / bbox_volume, 5) if bbox_volume > 0 else None
    except Exception:
        volume_ratio = None

    # Heuristic to detect potentially invalid meshes:
    # 1. Not watertight AND
    # 2. Simple mesh (< 1000 triangles) AND
    # 3. Very low volume ratio (< 0.3, suggesting 70%+ of bbox is empty)
    is_potentially_invalid = (
        not is_watertight and
        triangle_count < 1000 and
        volume_ratio is not None and
        volume_ratio < 0.3
    )

    return {
        "is_watertight": is_watertight,
        "triangle_count": triangle_count,
        "volume_ratio": volume_ratio,
        "is_potentially_invalid": is_potentially_invalid
    }


def collect_stats(path: Path, asset_metadata: dict | None = None) -> dict:
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

    # Calculate 3D bounding box volume (m³)
    bbox_volume = round(extents[0] * extents[1] * extents[2], 5)

    # Calculate 2D footprint (X × Y area in m²)
    footprint = round(extents[0] * extents[2], 5)

    # Height is the Y extent
    height = round(extents[1], 5)

    # Actual mesh volume
    try:
        volume = round(float(scene.volume), 5)
    except Exception as exc:  # pragma: no cover - trimesh volume failures
        raise ValueError(f"Unable to compute volume for: {path}") from exc

    # Check mesh validity
    try:
        validity_info = detect_mesh_validity_issues(scene, bbox_volume)
        is_watertight = validity_info["is_watertight"]
        triangle_count = validity_info["triangle_count"]
        volume_ratio = validity_info["volume_ratio"]
        is_potentially_invalid = validity_info["is_potentially_invalid"]
    except Exception:  # pragma: no cover - fallback if analysis fails
        is_watertight = None
        triangle_count = None
        volume_ratio = None
        is_potentially_invalid = None

    result = {
        "name": path.stem,
        "bounding_box": vector_to_axis_bounds(bbox_min_raw, bbox_max_raw),
        "bounding_box_volume": bbox_volume,
        "footprint": footprint,
        "height": height,
        "volume": volume,
        "volume_ratio": volume_ratio,
        "center_of_mass": vector_to_point(center_mass_vector),
        "is_watertight": is_watertight,
        "triangle_count": triangle_count,
        "is_potentially_invalid": is_potentially_invalid,
    }

    # Merge asset metadata if available
    if asset_metadata:
        result["path"] = asset_metadata.get("path", "")
        result["physicsCost"] = asset_metadata.get("physicsCost")
        result["category"] = asset_metadata.get("category")
        result["levelRestrictions"] = asset_metadata.get("levelRestrictions", [])

    return result


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
        output_path = directory / "prop_stats.json"
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load asset types if provided
    asset_map = {}
    if args.asset_types:
        asset_types_path = args.asset_types.expanduser().resolve()
        if not asset_types_path.is_file():
            raise SystemExit(f"Asset types file not found: {asset_types_path}")
        try:
            asset_map = load_asset_types(asset_types_path)
            print(f"Loaded {len(asset_map)} asset types from {asset_types_path}")
        except Exception as exc:
            raise SystemExit(f"Failed to load asset types: {exc}") from exc

    glb_files = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".glb"
    )

    if not glb_files:
        raise SystemExit(f"No .glb files found in: {directory}")

    results = []
    total_files = len(glb_files)
    for index, glb_path in enumerate(glb_files, start=1):
        try:
            # Look up asset metadata by the GLB filename (without extension)
            asset_metadata = asset_map.get(glb_path.stem)
            stats = collect_stats(glb_path, asset_metadata)
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
