#!/usr/bin/env python3
"""Visualize individual dining-cleanup object-pose episodes on world XY."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D


ANCHOR_WORLD_POSE = (0.40, 0.10, 0.0)
TABLE_X_RANGE = (0.0, 0.70)
TABLE_Y_RANGE = (-0.65, 0.0)
LEFT_TABLE_X_RANGE = (0.0, 0.22)
LEFT_TABLE_Y_RANGE = (-0.50, -0.10)
WIPE_LANES_X = (0.21, 0.18, 0.15, 0.11, 0.07)

TRAY_WORLD_POS = (0.57, -0.36)
TISSUE_WORLD_POS = (0.35, -0.12)
VASE_WORLD_POS = (0.35, -0.26)
CLOTH_WORLD_POS = (0.35, -0.43)

PER_OBJECT_YAW_OFFSET = {
    "bowl": 0.0,
    "spoon": 3.0 * math.pi / 2.0,
}

FOOTPRINT_SIZE = {
    "bowl": (0.140, 0.140),
    "spoon": (0.040, 0.194),
    "tray": (0.240, 0.260),
    "tissue": (0.073, 0.103),
    "vase": (0.100, 0.100),
    "cloth": (0.055, 0.115),
}

DEFAULT_INPUT = Path("data/dining_clean/dining_cleanup_spoon_random_yaw_200.json")
DEFAULT_OUTPUT_DIR = Path("data/dining_clean/spoon_random_yaw_first10")


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def raw_to_world_xy(raw_xy: list[float]) -> tuple[float, float]:
    anchor_x, anchor_y, anchor_yaw = ANCHOR_WORLD_POSE
    cos_a = math.cos(anchor_yaw)
    sin_a = math.sin(anchor_yaw)
    raw_x, raw_y = raw_xy
    return (
        anchor_x + cos_a * raw_x - sin_a * raw_y,
        anchor_y + sin_a * raw_x + cos_a * raw_y,
    )


def raw_to_world_yaw(name: str, raw_yaw: float) -> float:
    return normalize_angle(
        ANCHOR_WORLD_POSE[2] + raw_yaw + PER_OBJECT_YAW_OFFSET.get(name, 0.0)
    )


def load_full_entries(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return [entry for entry in data if entry.get("status") == "full"]


def entry_pose(entry: dict, name: str) -> tuple[tuple[float, float], float]:
    obj = next(obj for obj in entry["objects"] if obj["object_name"] == name)
    xy = raw_to_world_xy(obj["tvec"][:2])
    yaw = raw_to_world_yaw(name, float(obj["rvec"][2]))
    return xy, yaw


def add_axis_aligned_object(ax, name: str, xy: tuple[float, float], color: str) -> None:
    sx, sy = FOOTPRINT_SIZE[name]
    ax.add_patch(
        Rectangle(
            (xy[0] - sx / 2.0, xy[1] - sy / 2.0),
            sx,
            sy,
            facecolor=color,
            edgecolor="black",
            linewidth=1.1,
            alpha=0.28,
            zorder=4,
        )
    )
    ax.scatter([xy[0]], [xy[1]], s=70, color=color, edgecolor="black", linewidth=0.8, zorder=5)
    ax.text(xy[0] + 0.010, xy[1] + 0.010, name, fontsize=8, weight="bold", zorder=7)


def add_oriented_object(
    ax,
    name: str,
    xy: tuple[float, float],
    yaw: float,
    color: str,
    *,
    show_long_axis: bool = False,
) -> None:
    sx, sy = FOOTPRINT_SIZE[name]
    rect = Rectangle(
        (xy[0] - sx / 2.0, xy[1] - sy / 2.0),
        sx,
        sy,
        facecolor=color,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.36,
        zorder=8,
    )
    rect.set_transform(Affine2D().rotate_around(xy[0], xy[1], yaw) + ax.transData)
    ax.add_patch(rect)
    ax.scatter([xy[0]], [xy[1]], s=85, color=color, edgecolor="black", linewidth=0.9, zorder=9)

    # The spoon asset footprint is long in local y, so the visible long axis is
    # object yaw + 90 degrees.  Draw this axis explicitly for visual checks.
    if show_long_axis:
        axis_yaw = yaw + math.pi / 2.0
        half_len = sy * 0.58
        start = (
            xy[0] - 0.35 * half_len * math.cos(axis_yaw),
            xy[1] - 0.35 * half_len * math.sin(axis_yaw),
        )
        end = (
            xy[0] + half_len * math.cos(axis_yaw),
            xy[1] + half_len * math.sin(axis_yaw),
        )
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={"arrowstyle": "->", "color": "#7f1d1d", "lw": 2.2},
            zorder=10,
        )
        ax.text(
            xy[0] + 0.012,
            xy[1] - 0.025,
            f"spoon yaw={math.degrees(yaw):+.1f} deg",
            fontsize=8,
            color="#7f1d1d",
            weight="bold",
            zorder=11,
        )
    else:
        ax.text(xy[0] + 0.012, xy[1] + 0.012, name, fontsize=8, weight="bold", zorder=10)


def cloth_expected_path() -> list[tuple[float, float]]:
    y0, y1 = LEFT_TABLE_Y_RANGE
    points = [CLOTH_WORLD_POS, (WIPE_LANES_X[0], y0)]
    for idx, lane_x in enumerate(WIPE_LANES_X):
        points.append((lane_x, y1))
        if idx < len(WIPE_LANES_X) - 1:
            points.append((WIPE_LANES_X[idx + 1], y0))
    points.append(TRAY_WORLD_POS)
    return points


def add_scene_context(ax) -> None:
    ax.add_patch(
        Rectangle(
            (TABLE_X_RANGE[0], TABLE_Y_RANGE[0]),
            TABLE_X_RANGE[1] - TABLE_X_RANGE[0],
            TABLE_Y_RANGE[1] - TABLE_Y_RANGE[0],
            facecolor="#f6ead7",
            edgecolor="#5a4632",
            linewidth=2.0,
            label="table",
            zorder=1,
        )
    )
    ax.add_patch(
        Rectangle(
            (LEFT_TABLE_X_RANGE[0], LEFT_TABLE_Y_RANGE[0]),
            LEFT_TABLE_X_RANGE[1] - LEFT_TABLE_X_RANGE[0],
            LEFT_TABLE_Y_RANGE[1] - LEFT_TABLE_Y_RANGE[0],
            facecolor="#6cc3b5",
            edgecolor="#1f7a70",
            linewidth=1.8,
            alpha=0.16,
            label="dirty region",
            zorder=2,
        )
    )
    path = cloth_expected_path()
    xs, ys = zip(*path)
    ax.plot(xs, ys, color="#006d77", linewidth=1.4, alpha=0.72, zorder=3, label="cloth path")

    add_axis_aligned_object(ax, "tray", TRAY_WORLD_POS, "#8f4bd8")
    add_axis_aligned_object(ax, "tissue", TISSUE_WORLD_POS, "#cc8b00")
    add_axis_aligned_object(ax, "vase", VASE_WORLD_POS, "#5e8c31")
    add_axis_aligned_object(ax, "cloth", CLOTH_WORLD_POS, "#6b7cff")


def plot_episode(entry: dict, output: Path, episode_number: int) -> tuple[float, float, float]:
    bowl_xy, bowl_yaw = entry_pose(entry, "bowl")
    spoon_xy, spoon_yaw = entry_pose(entry, "spoon")

    fig, ax = plt.subplots(figsize=(8, 7), dpi=180)
    add_scene_context(ax)
    add_oriented_object(ax, "bowl", bowl_xy, bowl_yaw, "#2f80ed")
    add_oriented_object(ax, "spoon", spoon_xy, spoon_yaw, "#d64545", show_long_axis=True)

    ax.set_title(
        f"Dining Cleanup Pose Episode {episode_number:03d}",
        fontsize=13,
        weight="bold",
    )
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_xlim(TABLE_X_RANGE[0] - 0.05, TABLE_X_RANGE[1] + 0.05)
    ax.set_ylim(TABLE_Y_RANGE[0] - 0.08, TABLE_Y_RANGE[1] + 0.05)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return bowl_yaw, spoon_yaw, math.hypot(bowl_xy[0] - spoon_xy[0], bowl_xy[1] - spoon_xy[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize individual dining cleanup pose samples.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input object_poses JSON.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output image directory.")
    parser.add_argument("--start", type=int, default=1, help="1-based first full episode to visualize.")
    parser.add_argument("--count", type=int, default=10, help="Number of full episodes to visualize.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_full_entries(args.input)
    start_index = max(args.start - 1, 0)
    selected = entries[start_index : start_index + args.count]
    if not selected:
        raise ValueError(f"No full episodes selected from {args.input}")

    for offset, entry in enumerate(selected):
        episode_number = start_index + offset + 1
        output = args.output_dir / f"pose_ep_{episode_number:03d}.png"
        bowl_yaw, spoon_yaw, pair_dist = plot_episode(entry, output, episode_number)
        print(
            f"{output} | bowl_yaw={math.degrees(bowl_yaw):+6.1f} deg, "
            f"spoon_yaw={math.degrees(spoon_yaw):+6.1f} deg, "
            f"bowl_spoon_dist={pair_dist:.3f} m"
        )


if __name__ == "__main__":
    main()
