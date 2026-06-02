#!/usr/bin/env python3
"""Visualize dining-cleanup object occupancy on the table XY plane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


ANCHOR_WORLD_POSE = (0.40, 0.10, 0.0)

TABLE_X_RANGE = (0.0, 0.70)
TABLE_Y_RANGE = (-0.65, 0.0)
TABLE_MID_X = 0.35

LEFT_TABLE_X_RANGE = (0.08, 0.22)
RIGHT_TABLE_X_RANGE = (0.38, 0.66)
LEFT_TABLE_Y_RANGE = (-0.55, -0.10)
WIPE_LANES_X = (0.1, 0.15, 0.19)
WIPE_REQUIRED_IDEAL_FRACTION = 0.70

TRAY_WORLD_POS = (0.57, -0.36)
TISSUE_WORLD_POS = (0.35, -0.12)
VASE_WORLD_POS = (0.35, -0.26)
CLOTH_WORLD_POS = (0.35, -0.43)

TRAY_SUCCESS_X_HALF_WIDTH = 0.13
TRAY_SUCCESS_Y_HALF_WIDTH = 0.14
TRAY_BOWL_TARGET = (TRAY_WORLD_POS[0], TRAY_WORLD_POS[1] + 0.055)
TRAY_SPOON_TARGET = (TRAY_WORLD_POS[0], TRAY_WORLD_POS[1] - 0.055)
MIN_CLEARANCE = 0.040

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


def footprint_radius(name: str) -> float:
    sx, sy = FOOTPRINT_SIZE[name]
    return 0.5 * max(sx, sy)


def rect_union_area(rects: list[tuple[float, float, float, float]]) -> float:
    xs = sorted({value for rect in rects for value in (rect[0], rect[2])})
    area = 0.0
    for x0, x1 in zip(xs, xs[1:]):
        if x1 <= x0:
            continue
        mid_x = 0.5 * (x0 + x1)
        intervals = sorted((r[1], r[3]) for r in rects if r[0] <= mid_x <= r[2])
        merged: list[list[float]] = []
        for y0, y1 in intervals:
            if not merged or y0 > merged[-1][1]:
                merged.append([y0, y1])
            else:
                merged[-1][1] = max(merged[-1][1], y1)
        area += (x1 - x0) * sum(y1 - y0 for y0, y1 in merged)
    return area


def wipe_swept_rects(*, clipped: bool = False) -> list[tuple[float, float, float, float]]:
    cloth_x, cloth_y = FOOTPRINT_SIZE["cloth"]
    half_x = 0.5 * cloth_x
    half_y = 0.5 * cloth_y
    y0, y1 = LEFT_TABLE_Y_RANGE
    rects = []
    for x in WIPE_LANES_X:
        rect = (x - half_x, y0 - half_y, x + half_x, y1 + half_y)
        if clipped:
            rect = (
                max(LEFT_TABLE_X_RANGE[0], rect[0]),
                max(LEFT_TABLE_Y_RANGE[0], rect[1]),
                min(LEFT_TABLE_X_RANGE[1], rect[2]),
                min(LEFT_TABLE_Y_RANGE[1], rect[3]),
            )
        if rect[2] > rect[0] and rect[3] > rect[1]:
            rects.append(rect)
    return rects


def wipe_coverage_ratio() -> float:
    target_area = (LEFT_TABLE_X_RANGE[1] - LEFT_TABLE_X_RANGE[0]) * (
        LEFT_TABLE_Y_RANGE[1] - LEFT_TABLE_Y_RANGE[0]
    )
    if target_area <= 0.0:
        return 0.0
    return rect_union_area(wipe_swept_rects(clipped=True)) / target_area


def add_range_box(ax, points: list[tuple[float, float]], *, color: str, label: str, pad: float = 0.0) -> None:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    rect = Rectangle(
        (min(xs) - pad, min(ys) - pad),
        max(xs) - min(xs) + 2.0 * pad,
        max(ys) - min(ys) + 2.0 * pad,
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
    *,
    label_keepout: bool = False,
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
        )
    )
    ax.scatter([xy[0]], [xy[1]], s=90, c=color, edgecolors="black", linewidths=0.8, zorder=6)
    ax.add_patch(
        Circle(
            xy,
            footprint_radius(name) + MIN_CLEARANCE,
            fill=False,
            edgecolor=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label="keep-out radius (+clearance)" if label_keepout else "_nolegend_",
        )
    )
    ax.text(xy[0] + 0.012, xy[1] + 0.012, name, fontsize=9, weight="bold")


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
    ax.axvline(TABLE_MID_X, color="#5a4632", linestyle=":", linewidth=1.6, label="table midline")

    wipe_rect = Rectangle(
        (LEFT_TABLE_X_RANGE[0], LEFT_TABLE_Y_RANGE[0]),
        LEFT_TABLE_X_RANGE[1] - LEFT_TABLE_X_RANGE[0],
        LEFT_TABLE_Y_RANGE[1] - LEFT_TABLE_Y_RANGE[0],
        facecolor="#6cc3b5",
        edgecolor="#1f7a70",
        alpha=0.16,
        linewidth=2.0,
        label="wipe coverage region",
    )
    ax.add_patch(wipe_rect)

    for idx, rect in enumerate(wipe_swept_rects(clipped=False)):
        x0, y0, x1, y1 = rect
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="#1f7a70",
                edgecolor="#1f7a70",
                alpha=0.08,
                linewidth=1.0,
                label="cloth swept footprint" if idx == 0 else "_nolegend_",
            )
        )

    for idx, x in enumerate(WIPE_LANES_X):
        y0, y1 = LEFT_TABLE_Y_RANGE
        start_y, end_y = (y0, y1) if idx % 2 == 0 else (y1, y0)
        ax.annotate(
            "",
            xy=(x, end_y),
            xytext=(x, start_y),
            arrowprops={"arrowstyle": "->", "color": "#1f7a70", "lw": 1.2},
        )

    tray_zone = Rectangle(
        (
            TRAY_WORLD_POS[0] - TRAY_SUCCESS_X_HALF_WIDTH,
            TRAY_WORLD_POS[1] - TRAY_SUCCESS_Y_HALF_WIDTH,
        ),
        2.0 * TRAY_SUCCESS_X_HALF_WIDTH,
        2.0 * TRAY_SUCCESS_Y_HALF_WIDTH,
        facecolor="#dbb2ff",
        edgecolor="#6d3ba8",
        alpha=0.18,
        linewidth=2.0,
        label="tray success zone",
    )
    ax.add_patch(tray_zone)
    add_fixed_object(ax, "tray", TRAY_WORLD_POS, "#8f4bd8", label_keepout=True)
    ax.scatter([TRAY_BOWL_TARGET[0]], [TRAY_BOWL_TARGET[1]], s=80, marker="o", c="#2f80ed", edgecolors="white", zorder=8)
    ax.scatter([TRAY_SPOON_TARGET[0]], [TRAY_SPOON_TARGET[1]], s=80, marker="^", c="#d64545", edgecolors="white", zorder=8)
    ax.text(TRAY_BOWL_TARGET[0] + 0.012, TRAY_BOWL_TARGET[1], "bowl drop target", fontsize=8)
    ax.text(TRAY_SPOON_TARGET[0] + 0.012, TRAY_SPOON_TARGET[1], "spoon drop target", fontsize=8)

    add_fixed_object(ax, "tissue", TISSUE_WORLD_POS, "#cc8b00")
    add_fixed_object(ax, "vase", VASE_WORLD_POS, "#5e8c31")
    add_fixed_object(ax, "cloth", CLOTH_WORLD_POS, "#6b7cff")

    bowl_points = points["bowl"]
    spoon_points = points["spoon"]
    add_range_box(
        ax,
        bowl_points,
        color="#2f80ed",
        label="bowl occupied spawn envelope",
        pad=footprint_radius("bowl"),
    )
    add_range_box(
        ax,
        spoon_points,
        color="#d64545",
        label="spoon occupied spawn envelope",
        pad=footprint_radius("spoon"),
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
        "Rectangles show scaled USD XY bounds; dashed circles are generator keep-out radii with clearance.",
        fontsize=8,
        color="#6b5846",
    )
    ax.text(
        0.02,
        TABLE_Y_RANGE[0] - 0.072,
        "Planned cloth/table coverage over wipe region: "
        f"{100.0 * wipe_coverage_ratio():.1f}% "
        f"(success threshold: {100.0 * wipe_coverage_ratio() * WIPE_REQUIRED_IDEAL_FRACTION:.1f}%)",
        fontsize=8,
        color="#1f7a70",
    )

    ax.set_title("Dining Cleanup Table Occupancy (World XY)", fontsize=15, weight="bold")
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
    print(f"wipe region: x=[{LEFT_TABLE_X_RANGE[0]:.3f}, {LEFT_TABLE_X_RANGE[1]:.3f}], y=[{LEFT_TABLE_Y_RANGE[0]:.3f}, {LEFT_TABLE_Y_RANGE[1]:.3f}]")
    print(f"planned cloth/table coverage: {100.0 * wipe_coverage_ratio():.1f}%")
    print(f"coverage success threshold: {100.0 * wipe_coverage_ratio() * WIPE_REQUIRED_IDEAL_FRACTION:.1f}%")
    print(
        "tray success zone: "
        f"x=[{TRAY_WORLD_POS[0] - TRAY_SUCCESS_X_HALF_WIDTH:.3f}, "
        f"{TRAY_WORLD_POS[0] + TRAY_SUCCESS_X_HALF_WIDTH:.3f}], "
        f"y=[{TRAY_WORLD_POS[1] - TRAY_SUCCESS_Y_HALF_WIDTH:.3f}, "
        f"{TRAY_WORLD_POS[1] + TRAY_SUCCESS_Y_HALF_WIDTH:.3f}]"
    )
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
