#!/usr/bin/env python3
"""Visualize UMI-style cutlery object_poses.json on the XY plane.

This script does not import Isaac Sim.  It uses the cutlery task's anchor
mapping constants to show both:

1. raw anchor-frame ``tvec`` XY from object_poses.json
2. simulator/task world XY after applying ``ANCHOR_WORLD_POSE``
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Keep these aligned with:
# packages/simulator/src/simulator/tasks/cutlery_arrangement/cutlery_arrangement_env_cfg.py
ANCHOR_WORLD_POSE = (0.40, 0.10, 0.0)
PLATE_WORLD_POS = (0.50, -0.40, 0.05)

# Table footprint extracted from packages/simulator/assets/scenes/dining_room/scene.usd
# and expressed in task/world XY.
TABLE_XY = (0.0, 0.70, -0.65, 0.0)  # xmin, xmax, ymin, ymax

OBJECT_STYLE = {
    "fork": {"color": "#2f80ed", "marker": "^"},
    "knife": {"color": "#d64545", "marker": "s"},
    "plate": {"color": "#2c9c69", "marker": "o"},
}


def raw_xy_to_world_xy(raw_x: float, raw_y: float) -> tuple[float, float]:
    anchor_x, anchor_y, anchor_yaw = ANCHOR_WORLD_POSE
    cos_a = math.cos(anchor_yaw)
    sin_a = math.sin(anchor_yaw)
    return (
        anchor_x + cos_a * raw_x - sin_a * raw_y,
        anchor_y + sin_a * raw_x + cos_a * raw_y,
    )


def load_points(path: Path) -> tuple[dict[str, list[tuple[float, float]]], dict[str, list[tuple[float, float]]]]:
    with path.open() as f:
        episodes = json.load(f)
    if not isinstance(episodes, list):
        raise ValueError(f"{path}: expected top-level JSON list")

    raw_points = {name: [] for name in OBJECT_STYLE}
    world_points = {name: [] for name in OBJECT_STYLE}
    for ep_idx, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"{path}: episode {ep_idx} must be an object")
        if episode.get("status") != "full":
            continue
        objects = episode.get("objects")
        if not isinstance(objects, list):
            raise ValueError(f"{path}: episode {ep_idx} has no objects list")
        for obj_idx, obj in enumerate(objects):
            name = obj.get("object_name")
            if name not in raw_points:
                continue
            tvec = obj.get("tvec")
            if not isinstance(tvec, list) or len(tvec) != 3:
                raise ValueError(f"{path}: episode {ep_idx} object {obj_idx} has invalid tvec")
            raw_x, raw_y = float(tvec[0]), float(tvec[1])
            raw_points[name].append((raw_x, raw_y))
            world_points[name].append(raw_xy_to_world_xy(raw_x, raw_y))
    return raw_points, world_points


def _all_points(points_by_name: dict[str, list[tuple[float, float]]]) -> list[tuple[float, float]]:
    return [point for points in points_by_name.values() for point in points]


def _plot_object_points(ax, points_by_name: dict[str, list[tuple[float, float]]], *, raw: bool) -> None:
    for name, style in OBJECT_STYLE.items():
        points = points_by_name[name]
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        label = f"{name} raw" if raw else name
        ax.scatter(
            xs,
            ys,
            s=28,
            marker=style["marker"],
            c=style["color"],
            alpha=0.68,
            edgecolors="white",
            linewidths=0.35,
            label=label,
        )
        ax.scatter(
            [mean(xs)],
            [mean(ys)],
            s=155,
            marker="X",
            c=style["color"],
            edgecolors="black",
            linewidths=1.0,
        )


def plot(raw_points: dict[str, list[tuple[float, float]]], world_points: dict[str, list[tuple[float, float]]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), constrained_layout=True)
    fig.suptitle("Cutlery object poses on XY plane", fontsize=16, fontweight="bold")

    ax = axes[0]
    xmin, xmax, ymin, ymax = TABLE_XY
    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            facecolor="#cda66b",
            edgecolor="#8f642c",
            alpha=0.22,
            linewidth=2,
            label="table footprint",
        )
    )
    _plot_object_points(ax, world_points, raw=False)
    ax.scatter(
        [PLATE_WORLD_POS[0]],
        [PLATE_WORLD_POS[1]],
        s=220,
        marker="*",
        c="#f2c94c",
        edgecolors="black",
        linewidths=1.0,
        label="sim fixed plate cfg",
    )
    ax.set_title("Task/world XY")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.28)
    all_world = _all_points(world_points)
    if all_world:
        ax.set_xlim(min(xmin, min(p[0] for p in all_world)) - 0.05, max(xmax, max(p[0] for p in all_world)) + 0.05)
        ax.set_ylim(
            min(ymin, min(p[1] for p in all_world), PLATE_WORLD_POS[1]) - 0.05,
            max(ymax, max(p[1] for p in all_world), PLATE_WORLD_POS[1]) + 0.05,
        )
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    ax = axes[1]
    _plot_object_points(ax, raw_points, raw=True)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.35)
    ax.set_title("Raw anchor-frame XY")
    ax.set_xlabel("anchor-frame x [m]")
    ax.set_ylabel("anchor-frame y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def summarize(raw_points: dict[str, list[tuple[float, float]]], world_points: dict[str, list[tuple[float, float]]]) -> None:
    for label, points_by_name in (("raw", raw_points), ("world", world_points)):
        print(f"{label}:")
        for name in ("fork", "knife", "plate"):
            points = points_by_name[name]
            if not points:
                print(f"  {name}: n=0")
                continue
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            print(
                f"  {name}: n={len(points)}, "
                f"x=[{min(xs):.3f}, {max(xs):.3f}], "
                f"y=[{min(ys):.3f}, {max(ys):.3f}], "
                f"mean=({mean(xs):.3f}, {mean(ys):.3f})"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize cutlery object_poses.json raw/world XY positions.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/object_poses_wide_plate_keepout_200.json"),
        help="Input object_poses.json path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Defaults to '<input stem>_xy_overlay.png' next to input.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = args.input.with_name(f"{args.input.stem}_xy_overlay.png")

    raw_points, world_points = load_points(args.input)
    plot(raw_points, world_points, output)
    print(output)
    summarize(raw_points, world_points)


if __name__ == "__main__":
    main()
