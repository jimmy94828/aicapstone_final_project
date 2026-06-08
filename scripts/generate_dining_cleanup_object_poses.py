#!/usr/bin/env python3
"""Generate object_poses.json for the advanced dining cleanup task.

The generated file uses the same per-episode UMI-style schema as the other
tasks.  Only bowl and spoon are randomized because tray/tissue/vase/cloth are
fixed scene objects in DiningCleanupEnvCfg.

World layout:
    - table footprint: x=[0.0, 0.70], y=[-0.65, 0.0]
    - +x is the Franka-view right side; -x is the Franka-view left side
    - bowl/spoon initial region is a shared left-side dirty-area region

The loader converts raw anchor-frame XY to world XY by adding
ANCHOR_WORLD_POSE[:2], so this script writes raw tvec values.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ANCHOR_WORLD_POSE = (0.40, 0.10, 0.0)

DEFAULT_OUTPUT = Path("data/dining_clean/dining_cleanup_object_poses_500.json")
DEFAULT_VIDEO_NAME = "synthetic_dining_cleanup_poses.mp4"

OBJECT_WORLD_X_RANGE = (0.10, 0.24)
OBJECT_WORLD_Y_RANGE = (-0.50, -0.22)
MIN_CLEARANCE = 0.040
MAX_PAIR_DISTANCE = 0.28
SPOON_WORLD_YAW = math.pi / 4.0
SPOON_YAW_OFFSET = 3.0 * math.pi / 2.0
SPOON_RAW_YAW = (SPOON_WORLD_YAW - SPOON_YAW_OFFSET + math.pi) % (2.0 * math.pi) - math.pi

# Top-down footprints after applying the task spawn scale in
# DiningCleanupEnvCfg.  Bowl/spoon can yaw per episode, so overlap rejection
# uses each object's bounding-circle radius derived from this footprint.
FOOTPRINT_SIZE = {
    "bowl": (0.140, 0.140),
    "spoon": (0.040, 0.194),
    "tray": (0.240, 0.260),
    "tissue": (0.073, 0.103),
    "vase": (0.100, 0.100),
    "cloth": (0.055, 0.115),
}

# Fixed objects from DiningCleanupEnvCfg.  Bowl/spoon are sampled on the left
# side and kept away from these objects by footprint radius.
STATIC_WORLD_XY = (
    ("tray", (0.57, -0.36)),
    ("tissue", (0.35, -0.12)),
    ("vase", (0.35, -0.26)),
    ("cloth", (0.35, -0.43)),
)


def world_xy_to_raw_xy(world_xy: tuple[float, float]) -> tuple[float, float]:
    anchor_x, anchor_y, anchor_yaw = ANCHOR_WORLD_POSE
    cos_a = math.cos(anchor_yaw)
    sin_a = math.sin(anchor_yaw)
    dx = world_xy[0] - anchor_x
    dy = world_xy[1] - anchor_y
    return (
        cos_a * dx + sin_a * dy,
        -sin_a * dx + cos_a * dy,
    )


def dist_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def footprint_radius(name: str) -> float:
    sx, sy = FOOTPRINT_SIZE[name]
    return 0.5 * max(sx, sy)


def random_world_xy(
    rng: random.Random,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[float, float]:
    return rng.uniform(*x_range), rng.uniform(*y_range)


def footprints_clear(
    a_name: str,
    a_xy: tuple[float, float],
    b_name: str,
    b_xy: tuple[float, float],
    *,
    min_clearance: float,
) -> bool:
    min_dist = footprint_radius(a_name) + footprint_radius(b_name) + min_clearance
    return dist_xy(a_xy, b_xy) >= min_dist


def valid_object_xy(name: str, point: tuple[float, float], *, min_clearance: float) -> bool:
    return all(
        footprints_clear(name, point, static_name, static_xy, min_clearance=min_clearance)
        for static_name, static_xy in STATIC_WORLD_XY
    )


def sample_pair(
    rng: random.Random,
    *,
    object_world_x_range: tuple[float, float],
    object_world_y_range: tuple[float, float],
    min_clearance: float,
    max_pair_distance: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    min_pair_distance = footprint_radius("bowl") + footprint_radius("spoon") + min_clearance
    if max_pair_distance < min_pair_distance:
        raise ValueError(
            f"max_pair_distance={max_pair_distance:.3f} is smaller than required "
            f"minimum pair distance={min_pair_distance:.3f}"
        )
    for _ in range(5000):
        bowl_xy = random_world_xy(rng, x_range=object_world_x_range, y_range=object_world_y_range)
        pair_distance = rng.uniform(min_pair_distance, max_pair_distance)
        pair_theta = rng.uniform(-math.pi, math.pi)
        spoon_xy = (
            bowl_xy[0] + pair_distance * math.cos(pair_theta),
            bowl_xy[1] + pair_distance * math.sin(pair_theta),
        )
        if not (
            object_world_x_range[0] <= spoon_xy[0] <= object_world_x_range[1]
            and object_world_y_range[0] <= spoon_xy[1] <= object_world_y_range[1]
        ):
            continue
        if not valid_object_xy("bowl", bowl_xy, min_clearance=min_clearance) or not valid_object_xy(
            "spoon", spoon_xy, min_clearance=min_clearance
        ):
            continue
        if not footprints_clear("bowl", bowl_xy, "spoon", spoon_xy, min_clearance=min_clearance):
            continue
        return bowl_xy, spoon_xy
    raise RuntimeError("failed to sample a valid bowl/spoon pair")


def rvec_z_yaw(yaw: float) -> list[float]:
    return [0.0, 0.0, yaw]


def build_entries(
    count: int,
    seed: int,
    video_name: str,
    *,
    object_world_x_range: tuple[float, float],
    object_world_y_range: tuple[float, float],
    min_clearance: float,
    max_pair_distance: float,
) -> list[dict]:
    rng = random.Random(seed)
    entries: list[dict] = []
    frame_cursor = 0

    for idx in range(count):
        bowl_world_xy, spoon_world_xy = sample_pair(
            rng,
            object_world_x_range=object_world_x_range,
            object_world_y_range=object_world_y_range,
            min_clearance=min_clearance,
            max_pair_distance=max_pair_distance,
        )
        bowl_raw_xy = world_xy_to_raw_xy(bowl_world_xy)
        spoon_raw_xy = world_xy_to_raw_xy(spoon_world_xy)

        bowl_yaw = rng.uniform(-math.pi, math.pi)
        # The task loader adds the spoon USD yaw offset, so this raw yaw yields
        # a final world yaw of 45 degrees from +x in every generated episode.
        spoon_yaw = SPOON_RAW_YAW
        objects = [
            {
                "object_name": "bowl",
                "rvec": rvec_z_yaw(bowl_yaw),
                "tvec": [bowl_raw_xy[0], bowl_raw_xy[1], rng.uniform(0.04, 0.07)],
            },
            {
                "object_name": "spoon",
                "rvec": rvec_z_yaw(spoon_yaw),
                "tvec": [spoon_raw_xy[0], spoon_raw_xy[1], rng.uniform(0.04, 0.07)],
            },
        ]
        rng.shuffle(objects)

        episode_len = rng.randint(1250, 1800)
        entries.append(
            {
                "video_name": video_name,
                "episode_range": [frame_cursor, frame_cursor + episode_len],
                "objects": objects,
                "status": "full",
            }
        )
        frame_cursor += episode_len

    return entries


def summarize(entries: list[dict], *, min_clearance: float, max_pair_distance: float) -> None:
    print(f"episodes={len(entries)}")
    for name in ("bowl", "spoon"):
        xs, ys = [], []
        for entry in entries:
            obj = next(o for o in entry["objects"] if o["object_name"] == name)
            x_raw, y_raw = obj["tvec"][:2]
            xs.append(x_raw + ANCHOR_WORLD_POSE[0])
            ys.append(y_raw + ANCHOR_WORLD_POSE[1])
        print(
            f"{name}: world x=[{min(xs):.3f}, {max(xs):.3f}], "
            f"y=[{min(ys):.3f}, {max(ys):.3f}]"
        )

    pair_dists = []
    for entry in entries:
        points = {}
        for obj in entry["objects"]:
            raw_xy = obj["tvec"][:2]
            points[obj["object_name"]] = (
                raw_xy[0] + ANCHOR_WORLD_POSE[0],
                raw_xy[1] + ANCHOR_WORLD_POSE[1],
            )
        pair_dists.append(dist_xy(points["bowl"], points["spoon"]))
    print(
        "bowl-spoon world XY distance="
        f"[{min(pair_dists):.3f}, {max(pair_dists):.3f}], "
        f"mean={sum(pair_dists) / len(pair_dists):.3f}"
    )
    print(
        "scaled footprint clearance min="
        f"{footprint_radius('bowl') + footprint_radius('spoon') + min_clearance:.3f}"
    )
    print(f"configured bowl-spoon max distance={max_pair_distance:.3f}")
    for name in ("tray", "tissue", "vase", "cloth", "bowl", "spoon"):
        sx, sy = FOOTPRINT_SIZE[name]
        print(f"{name} scaled footprint: {sx:.3f} x {sy:.3f} m")


def parse_range(text: str, *, name: str) -> tuple[float, float]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"{name} must use 'min,max' format, got {text!r}")
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must contain numeric bounds, got {text!r}") from exc
    if hi <= lo:
        raise argparse.ArgumentTypeError(f"{name} max must be greater than min, got {text!r}")
    return lo, hi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dining cleanup object poses.")
    parser.add_argument("--count", type=int, default=500, help="Number of full episodes to generate.")
    parser.add_argument("--seed", type=int, default=2026053002, help="Deterministic random seed.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
    parser.add_argument("--video-name", default=DEFAULT_VIDEO_NAME)
    parser.add_argument(
        "--object-world-x-range",
        type=lambda value: parse_range(value, name="--object-world-x-range"),
        default=OBJECT_WORLD_X_RANGE,
        help="World-frame x range for both bowl and spoon starts, formatted as 'min,max'.",
    )
    parser.add_argument(
        "--object-world-y-range",
        type=lambda value: parse_range(value, name="--object-world-y-range"),
        default=OBJECT_WORLD_Y_RANGE,
        help="World-frame y range for both bowl and spoon starts, formatted as 'min,max'.",
    )
    parser.add_argument(
        "--min-clearance",
        type=float,
        default=MIN_CLEARANCE,
        help="Minimum XY clearance between object footprint radii.",
    )
    parser.add_argument(
        "--max-pair-distance",
        type=float,
        default=MAX_PAIR_DISTANCE,
        help="Maximum XY distance between bowl and spoon centers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = build_entries(
        args.count,
        args.seed,
        args.video_name,
        object_world_x_range=args.object_world_x_range,
        object_world_y_range=args.object_world_y_range,
        min_clearance=args.min_clearance,
        max_pair_distance=args.max_pair_distance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(entries, f, indent=4)
        f.write("\n")

    print(args.output)
    print(
        "configured bowl/spoon world range: "
        f"x=[{args.object_world_x_range[0]:.3f}, {args.object_world_x_range[1]:.3f}], "
        f"y=[{args.object_world_y_range[0]:.3f}, {args.object_world_y_range[1]:.3f}]"
    )
    summarize(entries, min_clearance=args.min_clearance, max_pair_distance=args.max_pair_distance)


if __name__ == "__main__":
    main()
