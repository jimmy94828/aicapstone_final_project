#!/usr/bin/env python3
"""Visualize dining-cleanup object occupancy on the table XY plane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ANCHOR_WORLD_POSE = (0.40, 0.10, 0.0)

TABLE_X_RANGE = (0.0, 0.70)
TABLE_Y_RANGE = (-0.65, 0.0)
TABLE_MID_X = 0.35

LEFT_TABLE_X_RANGE = (0.0, 0.22)
RIGHT_TABLE_X_RANGE = (0.38, 0.66)
LEFT_TABLE_Y_RANGE = (-0.50, -0.10)
WIPE_LANES_X = (0.21, 0.18, 0.15, 0.11, 0.07)

TRAY_WORLD_POS = (0.57, -0.36)
TISSUE_WORLD_POS = (0.35, -0.12)
VASE_WORLD_POS = (0.35, -0.26)
CLOTH_WORLD_POS = (0.35, -0.43)

# Top-down footprints after applying the task spawn scale in
# DiningCleanupEnvCfg.  These are measured from the updated USD assets and are
# the sizes used for planning, visualization, and object-pose overlap rejection.
FOOTPRINT_SIZE = {
    "bowl": (0.140, 0.140),
    "spoon": (0.040, 0.194),
    "tray": (0.240, 0.260),
    "tissue": (0.073, 0.103),
    "vase": (0.100, 0.100),
    "cloth": (0.055, 0.115),
}

DEFAULT_INPUT = Path("data/dining_clean/dining_cleanup_object_poses_500.json")
DEFAULT_OUTPUT = Path("data/dining_clean/dining_cleanup_layout_xy.png")


def world_xy_from_raw(raw_xy: list[float]) -> tuple[float, float]:
    return raw_xy[0] + ANCHOR_WORLD_POSE[0], raw_xy[1] + ANCHOR_WORLD_POSE[1]


def rect_from_ranges(x_range: tuple[float, float], y_range: tuple[float, float]) -> tuple[float, float, float, float]:
    return x_range[0], y_range[0], x_range[1] - x_range[0], y_range[1] - y_range[0]


def load_points(path: Path) -> dict[str, list[tuple[float, float]]]:
    data = json.loads(path.read_text())
    points = {"bowl": [], "spoon": []}
    for entry in data:
        if entry.get("status") != "full":
            continue
        for obj in entry["objects"]:
            name = obj["object_name"]
            if name in points:
                points[name].append(world_xy_from_raw(obj["tvec"][:2]))
    return points


def add_range_box(ax, points: list[tuple[float, float]], *, color: str, label: str, object_name: str) -> None:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    sx, sy = FOOTPRINT_SIZE[object_name]
    rect = Rectangle(
        (min(xs) - sx / 2.0, min(ys) - sy / 2.0),
        max(xs) - min(xs) + sx,
        max(ys) - min(ys) + sy,
        facecolor=color,
        edgecolor=color,
        alpha=0.12,
        linewidth=2,
        label=label,
    )
    ax.add_patch(rect)


def add_fixed_object(
    ax,
    name: str,
    xy: tuple[float, float],
    color: str,
) -> None:
    sx, sy = FOOTPRINT_SIZE[name]
    ax.add_patch(
        Rectangle(
            (xy[0] - sx / 2.0, xy[1] - sy / 2.0),
            sx,
            sy,
            facecolor=color,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.28,
            zorder=5,
            label=f"{name} footprint",
        )
    )
    ax.scatter([xy[0]], [xy[1]], s=90, c=color, edgecolors="black", linewidths=0.8, zorder=6)
    ax.text(xy[0] + 0.012, xy[1] + 0.012, name, fontsize=9, weight="bold")


def cloth_expected_path() -> list[tuple[float, float]]:
    y0, y1 = LEFT_TABLE_Y_RANGE
    points = [CLOTH_WORLD_POS, (WIPE_LANES_X[0], y0)]
    for idx, lane_x in enumerate(WIPE_LANES_X):
        points.append((lane_x, y1))
        if idx < len(WIPE_LANES_X) - 1:
            points.append((WIPE_LANES_X[idx + 1], y0))
    points.append(TRAY_WORLD_POS)
    return points


def add_cloth_expected_path(ax) -> None:
    points = cloth_expected_path()
    xs, ys = zip(*points)
    ax.plot(
        xs,
        ys,
        color="#006d77",
        linewidth=2.0,
        marker="x",
        markersize=4,
        label="expected cloth path",
        zorder=8,
    )
    for start, end in zip(points, points[1:]):
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "->",
                "color": "#006d77",
                "lw": 1.6,
                "shrinkA": 3,
                "shrinkB": 3,
            },
            zorder=9,
        )


def plot_layout(points: dict[str, list[tuple[float, float]]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), dpi=180)

    ax.add_patch(
        Rectangle(
            (TABLE_X_RANGE[0], TABLE_Y_RANGE[0]),
            TABLE_X_RANGE[1] - TABLE_X_RANGE[0],
            TABLE_Y_RANGE[1] - TABLE_Y_RANGE[0],
            facecolor="#f6ead7",
            edgecolor="#5a4632",
            linewidth=2.0,
            label="table footprint",
        )
    )

    wipe_rect = Rectangle(
        (LEFT_TABLE_X_RANGE[0], LEFT_TABLE_Y_RANGE[0]),
        LEFT_TABLE_X_RANGE[1] - LEFT_TABLE_X_RANGE[0],
        LEFT_TABLE_Y_RANGE[1] - LEFT_TABLE_Y_RANGE[0],
        facecolor="#6cc3b5",
        edgecolor="#1f7a70",
        alpha=0.16,
        linewidth=2.0,
        label="dirty region",
    )
    ax.add_patch(wipe_rect)
    add_cloth_expected_path(ax)

    add_fixed_object(ax, "tray", TRAY_WORLD_POS, "#8f4bd8")
    add_fixed_object(ax, "tissue", TISSUE_WORLD_POS, "#cc8b00")
    add_fixed_object(ax, "vase", VASE_WORLD_POS, "#5e8c31")
    add_fixed_object(ax, "cloth", CLOTH_WORLD_POS, "#6b7cff")

    bowl_points = points["bowl"]
    spoon_points = points["spoon"]
    add_range_box(
        ax,
        bowl_points,
        color="#2f80ed",
        label="bowl occupied envelope",
        object_name="bowl",
    )
    add_range_box(
        ax,
        spoon_points,
        color="#d64545",
        label="spoon occupied envelope",
        object_name="spoon",
    )

    bx, by = zip(*bowl_points)
    sx, sy = zip(*spoon_points)
    ax.scatter(bx, by, s=12, c="#2f80ed", alpha=0.36, edgecolors="none", label="bowl starts")
    ax.scatter(sx, sy, s=12, c="#d64545", alpha=0.36, edgecolors="none", label="spoon starts")

    ax.text(0.06, -0.04, "-x left / dirty area", fontsize=9, color="#5a4632")
    ax.text(0.43, -0.04, "+x right / tray side", fontsize=9, color="#5a4632")
    ax.text(
        0.02,
        TABLE_Y_RANGE[0] - 0.045,
        "Rectangles show object occupied XY footprints/envelopes and the dirty region.",
        fontsize=8,
        color="#6b5846",
    )

    ax.set_title("Dining Cleanup Object Occupancy and Dirty Region (World XY)", fontsize=15, weight="bold")
    ax.set_xlabel("world x (m)")
    ax.set_ylabel("world y (m)")
    ax.set_xlim(TABLE_X_RANGE[0] - 0.05, TABLE_X_RANGE[1] + 0.05)
    ax.set_ylim(TABLE_Y_RANGE[0] - 0.10, TABLE_Y_RANGE[1] + 0.06)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def print_summary(points: dict[str, list[tuple[float, float]]]) -> None:
    for name in ("bowl", "spoon"):
        xs = [p[0] for p in points[name]]
        ys = [p[1] for p in points[name]]
        print(
            f"{name}: n={len(points[name])}, "
            f"x=[{min(xs):.3f}, {max(xs):.3f}], "
            f"y=[{min(ys):.3f}, {max(ys):.3f}]"
        )
    print(f"table: x=[{TABLE_X_RANGE[0]:.3f}, {TABLE_X_RANGE[1]:.3f}], y=[{TABLE_Y_RANGE[0]:.3f}, {TABLE_Y_RANGE[1]:.3f}]")
    print(f"dirty region: x=[{LEFT_TABLE_X_RANGE[0]:.3f}, {LEFT_TABLE_X_RANGE[1]:.3f}], y=[{LEFT_TABLE_Y_RANGE[0]:.3f}, {LEFT_TABLE_Y_RANGE[1]:.3f}]")
    print("expected cloth path:")
    for idx, (x, y) in enumerate(cloth_expected_path()):
        print(f"  {idx}: ({x:.3f}, {y:.3f})")
    for name in ("tray", "tissue", "vase", "cloth", "bowl", "spoon"):
        sx, sy = FOOTPRINT_SIZE[name]
        print(f"{name} scaled footprint: {sx:.3f} x {sy:.3f} m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize dining cleanup object occupancy.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input dining cleanup object_poses JSON.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points = load_points(args.input)
    plot_layout(points, args.output)
    print(args.output)
    print_summary(points)


if __name__ == "__main__":
    main()
