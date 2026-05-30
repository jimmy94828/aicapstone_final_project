#!/usr/bin/env python3
"""Generate synthetic UMI-style object_poses.json for cutlery arrangement.

The generated JSON uses raw anchor-frame ``tvec`` values.  For the cutlery task,
the simulator converts raw XY to world XY as:

    world_xy = raw_tvec_xy + ANCHOR_WORLD_POSE[:2]

Plate is fixed to the simulator config's ``PLATE_WORLD_POS`` by writing the
inverse raw anchor-frame pose into every episode:

    plate_raw_xy = PLATE_WORLD_POS[:2] - ANCHOR_WORLD_POSE[:2]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


# Keep these aligned with:
# packages/simulator/src/simulator/tasks/cutlery_arrangement/cutlery_arrangement_env_cfg.py
ANCHOR_WORLD_POSE = (0.40, 0.10, 0.0)
PLATE_WORLD_POS = (0.50, -0.40, 0.05)

# Table footprint previously extracted from dining_room/scene.usd in task/world XY:
# x=[0.0, 0.70], y=[-0.65, 0.0].  The raw ranges below are inverse-transformed
# with a small edge margin.
RAW_X_RANGE = (-0.34, 0.26)  # world x=[0.06, 0.66]
RAW_Y_RANGE = (-0.68, -0.10)  # world y=[-0.58, 0.00]
MIN_WORLD_Y_MARGIN = 0.005

DEFAULT_OUTPUT = Path("data/object_poses_plate_fixed_fork_knife_diverse_200.json")
DEFAULT_PAIR_DIST_BINS = "0.06:0.14,0.14:0.24,0.24:0.36,0.36:0.55"
NEAR_EVAL_PAIR_DIST_BINS = "0.05:0.10,0.10:0.16,0.16:0.24,0.24:0.34"

# eval/cutlery_arrangement_eval.py initial object positions.  These are world
# coordinates; the generator converts them back to raw anchor-frame tvec values.
EVAL_KNIFE_WORLD_XY = (0.50, -0.10)
EVAL_FORK_WORLD_XY = (0.55, -0.10)

PRESETS = {
    "generic": {
        "output": DEFAULT_OUTPUT,
        "video_name": "synthetic_plate_fixed_cutlery_poses.mp4",
        "min_pair_dist": 0.060,
        "min_plate_dist": 0.060,
        "max_plate_dist": 0.300,
        "near_plate_ratio": 0.70,
        "pair_dist_bins": DEFAULT_PAIR_DIST_BINS,
    },
    "wide-plate-keepout": {
        "output": Path("data/object_poses_wide_plate_keepout_200.json"),
        "video_name": "synthetic_wide_plate_keepout_cutlery_poses.mp4",
        "min_pair_dist": 0.060,
        "min_plate_dist": 0.150,
        "max_plate_dist": 0.320,
        "near_plate_ratio": 0.0,
        "pair_dist_bins": DEFAULT_PAIR_DIST_BINS,
        "min_world_y": -0.55,
    },
    "near-eval-init": {
        "output": Path("data/object_poses_near_eval_init_200.json"),
        "video_name": "synthetic_near_eval_init_cutlery_poses.mp4",
        "min_pair_dist": 0.050,
        "min_plate_dist": 0.100,
        "max_plate_dist": 0.300,
        "near_plate_ratio": 0.0,
        "pair_dist_bins": NEAR_EVAL_PAIR_DIST_BINS,
    },
}


def van_der_corput(index: int, base: int) -> float:
    value = 0.0
    denom = 1.0
    while index:
        index, rem = divmod(index, base)
        denom *= base
        value += rem / denom
    return value


def halton_2d(index: int, base_x: int, base_y: int, shift_x: float, shift_y: float) -> tuple[float, float]:
    # Use index + 1 so the first point is not pinned exactly at 0.
    x = (van_der_corput(index + 1, base_x) + shift_x) % 1.0
    y = (van_der_corput(index + 1, base_y) + shift_y) % 1.0
    return x, y


def scale01(value: float, lo: float, hi: float) -> float:
    return lo + (hi - lo) * value


def jittered_halton_points(
    count: int,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    bases: tuple[int, int],
    shifts: tuple[float, float],
    jitter: float,
    rng: random.Random,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for idx in range(count):
        u, v = halton_2d(idx, bases[0], bases[1], shifts[0], shifts[1])
        x = scale01(u, *x_range) + rng.uniform(-jitter, jitter)
        y = scale01(v, *y_range) + rng.uniform(-jitter, jitter)
        x = min(max(x, x_range[0]), x_range[1])
        y = min(max(y, y_range[0]), y_range[1])
        points.append((x, y))
    rng.shuffle(points)
    return points


def dist_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def raw_y_range_for_min_world_y(min_world_y: float | None) -> tuple[float, float]:
    if min_world_y is None:
        return RAW_Y_RANGE
    # ANCHOR_WORLD_POSE yaw is currently 0, so world_y = raw_y + anchor_y.
    # Add a small margin because the user constraint is strictly "greater than".
    min_raw_y = min_world_y - ANCHOR_WORLD_POSE[1] + MIN_WORLD_Y_MARGIN
    y_range = (max(RAW_Y_RANGE[0], min_raw_y), RAW_Y_RANGE[1])
    if y_range[0] >= y_range[1]:
        raise ValueError(f"--min-world-y={min_world_y} leaves no valid raw y range")
    return y_range


def random_xy(rng: random.Random, *, y_range: tuple[float, float] = RAW_Y_RANGE) -> tuple[float, float]:
    return rng.uniform(*RAW_X_RANGE), rng.uniform(*y_range)


def in_workspace(point: tuple[float, float], *, y_range: tuple[float, float] = RAW_Y_RANGE) -> bool:
    x, y = point
    return RAW_X_RANGE[0] <= x <= RAW_X_RANGE[1] and y_range[0] <= y <= y_range[1]


def random_xy_in_ellipse(
    rng: random.Random,
    *,
    center: tuple[float, float],
    x_radius: float,
    y_radius: float,
) -> tuple[float, float]:
    for _ in range(1000):
        radius = math.sqrt(rng.random())
        theta = rng.uniform(-math.pi, math.pi)
        point = (
            center[0] + x_radius * radius * math.cos(theta),
            center[1] + y_radius * radius * math.sin(theta),
        )
        if in_workspace(point):
            return point
    raise RuntimeError(f"failed to sample point in ellipse centered at {center}")


def jittered_halton_ellipse_points(
    count: int,
    *,
    center: tuple[float, float],
    x_radius: float,
    y_radius: float,
    bases: tuple[int, int],
    shifts: tuple[float, float],
    jitter: float,
    rng: random.Random,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    candidate_idx = 0
    max_candidates = max(1000, count * 80)
    while len(points) < count and candidate_idx < max_candidates:
        u, v = halton_2d(candidate_idx, bases[0], bases[1], shifts[0], shifts[1])
        radius = math.sqrt(u)
        theta = 2.0 * math.pi * v
        point = (
            center[0] + x_radius * radius * math.cos(theta) + rng.uniform(-jitter, jitter),
            center[1] + y_radius * radius * math.sin(theta) + rng.uniform(-jitter, jitter),
        )
        if in_workspace(point):
            points.append(point)
        candidate_idx += 1

    while len(points) < count:
        points.append(
            random_xy_in_ellipse(
                rng,
                center=center,
                x_radius=x_radius,
                y_radius=y_radius,
            )
        )

    rng.shuffle(points)
    return points


def random_xy_near_plate(
    rng: random.Random,
    *,
    min_plate_dist: float,
    max_plate_dist: float,
    y_range: tuple[float, float] = RAW_Y_RANGE,
) -> tuple[float, float]:
    plate_xy = tuple(plate_raw_tvec()[:2])
    for _ in range(1000):
        # sqrt keeps samples approximately uniform by annulus area.
        r_sq = rng.uniform(min_plate_dist * min_plate_dist, max_plate_dist * max_plate_dist)
        radius = math.sqrt(r_sq)
        theta = rng.uniform(-math.pi, math.pi)
        point = (plate_xy[0] + radius * math.cos(theta), plate_xy[1] + radius * math.sin(theta))
        if in_workspace(point, y_range=y_range):
            return point
    return random_xy(rng, y_range=y_range)


def parse_pair_dist_bins(spec: str) -> list[tuple[float, float]]:
    bins: list[tuple[float, float]] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            lo_s, hi_s = chunk.split(":", 1)
            lo, hi = float(lo_s), float(hi_s)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --pair-dist-bins entry {chunk!r}; expected 'lo:hi,lo:hi'"
            ) from exc
        if lo < 0.0 or hi <= lo:
            raise ValueError(f"Invalid --pair-dist-bins entry {chunk!r}; require 0 <= lo < hi")
        bins.append((lo, hi))
    if not bins:
        raise ValueError("--pair-dist-bins must contain at least one 'lo:hi' entry")
    return bins


def pair_distance_ok(distance: float, target_bin: tuple[float, float]) -> bool:
    return target_bin[0] <= distance <= target_bin[1]


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


def rvec_for_object(name: str, idx: int, rng: random.Random) -> list[float]:
    """Return diverse axis-angle-like metadata.

    Current cutlery datagen uses ``use_fixed_yaw=True`` for fork/knife, so the
    loader ignores these rvec yaws for spawned fork/knife orientation.  Keeping
    diverse rvecs still makes the raw data shape realistic.
    """
    yaw = -math.pi + 2.0 * math.pi * ((idx * 0.61803398875 + rng.uniform(-0.04, 0.04)) % 1.0)
    phase = 2.0 * math.pi * ((idx % 37) / 37.0)
    tilt = {"fork": 0.80, "knife": 0.60, "plate": 0.0}[name]
    rx = tilt * math.sin(phase) + rng.uniform(-0.16, 0.16)
    ry = tilt * math.cos(phase * 0.73) + rng.uniform(-0.16, 0.16)
    rz = yaw + rng.uniform(-0.08, 0.08)
    if name == "plate":
        return [0.0, 0.0, 0.0]
    return [rx, ry, rz]


def plate_raw_tvec() -> list[float]:
    return [
        PLATE_WORLD_POS[0] - ANCHOR_WORLD_POSE[0],
        PLATE_WORLD_POS[1] - ANCHOR_WORLD_POSE[1],
        PLATE_WORLD_POS[2],
    ]


def build_entries(
    *,
    count: int,
    seed: int,
    min_pair_dist: float,
    min_plate_dist: float,
    max_plate_dist: float,
    near_plate_ratio: float,
    pair_dist_bins: list[tuple[float, float]],
    min_world_y: float | None,
    video_name: str,
) -> list[dict]:
    if not 0.0 <= near_plate_ratio <= 1.0:
        raise ValueError("--near-plate-ratio must be between 0 and 1")
    if max_plate_dist <= min_plate_dist:
        raise ValueError("--max-plate-dist must be greater than --min-plate-dist")

    rng = random.Random(seed)
    y_range = raw_y_range_for_min_world_y(min_world_y)
    fork_candidates = jittered_halton_points(
        count,
        x_range=RAW_X_RANGE,
        y_range=y_range,
        bases=(2, 3),
        shifts=(0.13, 0.29),
        jitter=0.015,
        rng=rng,
    )
    knife_candidates = jittered_halton_points(
        count,
        x_range=RAW_X_RANGE,
        y_range=y_range,
        bases=(5, 7),
        shifts=(0.47, 0.11),
        jitter=0.015,
        rng=rng,
    )

    fixed_plate = tuple(plate_raw_tvec()[:2])
    entries: list[dict] = []
    frame_cursor = 0

    for idx in range(count):
        target_pair_bin = pair_dist_bins[idx % len(pair_dist_bins)]
        if target_pair_bin[0] < min_pair_dist:
            target_pair_bin = (min_pair_dist, target_pair_bin[1])
        if target_pair_bin[1] <= target_pair_bin[0]:
            raise ValueError(
                f"Pair distance bin {pair_dist_bins[idx % len(pair_dist_bins)]} "
                f"is incompatible with --min-pair-dist={min_pair_dist}"
            )

        def sample_object(base_point: tuple[float, float]) -> tuple[float, float]:
            if rng.random() < near_plate_ratio:
                return random_xy_near_plate(
                    rng,
                    min_plate_dist=min_plate_dist,
                    max_plate_dist=max_plate_dist,
                    y_range=y_range,
                )
            return base_point

        fork_xy = sample_object(fork_candidates[idx])
        knife_xy = sample_object(knife_candidates[(idx * 73 + 19) % count])

        for _ in range(1000):
            pair_dist = dist_xy(fork_xy, knife_xy)
            if (
                pair_dist >= min_pair_dist
                and pair_distance_ok(pair_dist, target_pair_bin)
                and dist_xy(fork_xy, fixed_plate) >= min_plate_dist
                and dist_xy(knife_xy, fixed_plate) >= min_plate_dist
            ):
                break

            # Resample both objects for pair-distance diversity.  This avoids
            # getting stuck in one local fork/knife distance range.
            if rng.random() < near_plate_ratio:
                fork_xy = random_xy_near_plate(
                    rng,
                    min_plate_dist=min_plate_dist,
                    max_plate_dist=max_plate_dist,
                    y_range=y_range,
                )
            else:
                fork_xy = random_xy(rng, y_range=y_range)
            if rng.random() < near_plate_ratio:
                knife_xy = random_xy_near_plate(
                    rng,
                    min_plate_dist=min_plate_dist,
                    max_plate_dist=max_plate_dist,
                    y_range=y_range,
                )
            else:
                knife_xy = random_xy(rng, y_range=y_range)
        else:
            raise RuntimeError(f"failed to sample non-overlapping fork/knife pose at index {idx}")

        objects = [
            {
                "object_name": "fork",
                "rvec": rvec_for_object("fork", idx, rng),
                "tvec": [fork_xy[0], fork_xy[1], rng.uniform(0.075, 0.225)],
            },
            {
                "object_name": "knife",
                "rvec": rvec_for_object("knife", idx, rng),
                "tvec": [knife_xy[0], knife_xy[1], rng.uniform(0.075, 0.225)],
            },
            {
                "object_name": "plate",
                "rvec": rvec_for_object("plate", idx, rng),
                "tvec": plate_raw_tvec(),
            },
        ]
        rng.shuffle(objects)

        episode_len = rng.randint(1350, 1800)
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


def build_near_eval_entries(
    *,
    count: int,
    seed: int,
    min_pair_dist: float,
    min_plate_dist: float,
    pair_dist_bins: list[tuple[float, float]],
    near_eval_x_radius: float,
    near_eval_y_radius: float,
    video_name: str,
) -> list[dict]:
    rng = random.Random(seed)
    fork_center = world_xy_to_raw_xy(EVAL_FORK_WORLD_XY)
    knife_center = world_xy_to_raw_xy(EVAL_KNIFE_WORLD_XY)

    fork_candidates = jittered_halton_ellipse_points(
        count,
        center=fork_center,
        x_radius=near_eval_x_radius,
        y_radius=near_eval_y_radius,
        bases=(2, 3),
        shifts=(0.19, 0.37),
        jitter=0.008,
        rng=rng,
    )
    knife_candidates = jittered_halton_ellipse_points(
        count,
        center=knife_center,
        x_radius=near_eval_x_radius,
        y_radius=near_eval_y_radius,
        bases=(5, 7),
        shifts=(0.41, 0.23),
        jitter=0.008,
        rng=rng,
    )

    fixed_plate = tuple(plate_raw_tvec()[:2])
    entries: list[dict] = []
    frame_cursor = 0

    for idx in range(count):
        target_pair_bin = pair_dist_bins[idx % len(pair_dist_bins)]
        if target_pair_bin[0] < min_pair_dist:
            target_pair_bin = (min_pair_dist, target_pair_bin[1])
        if target_pair_bin[1] <= target_pair_bin[0]:
            raise ValueError(
                f"Pair distance bin {pair_dist_bins[idx % len(pair_dist_bins)]} "
                f"is incompatible with --min-pair-dist={min_pair_dist}"
            )

        fork_xy = fork_candidates[idx]
        knife_xy = knife_candidates[(idx * 59 + 11) % count]
        for _ in range(1000):
            pair_dist = dist_xy(fork_xy, knife_xy)
            if (
                pair_dist >= min_pair_dist
                and pair_distance_ok(pair_dist, target_pair_bin)
                and dist_xy(fork_xy, fixed_plate) >= min_plate_dist
                and dist_xy(knife_xy, fixed_plate) >= min_plate_dist
            ):
                break
            fork_xy = random_xy_in_ellipse(
                rng,
                center=fork_center,
                x_radius=near_eval_x_radius,
                y_radius=near_eval_y_radius,
            )
            knife_xy = random_xy_in_ellipse(
                rng,
                center=knife_center,
                x_radius=near_eval_x_radius,
                y_radius=near_eval_y_radius,
            )
        else:
            raise RuntimeError(f"failed to sample near-eval fork/knife pose at index {idx}")

        objects = [
            {
                "object_name": "fork",
                "rvec": rvec_for_object("fork", idx, rng),
                "tvec": [fork_xy[0], fork_xy[1], rng.uniform(0.075, 0.225)],
            },
            {
                "object_name": "knife",
                "rvec": rvec_for_object("knife", idx, rng),
                "tvec": [knife_xy[0], knife_xy[1], rng.uniform(0.075, 0.225)],
            },
            {
                "object_name": "plate",
                "rvec": rvec_for_object("plate", idx, rng),
                "tvec": plate_raw_tvec(),
            },
        ]
        rng.shuffle(objects)

        episode_len = rng.randint(1350, 1800)
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


def summarize(entries: list[dict]) -> None:
    plate_raw = plate_raw_tvec()
    fork_knife_dists = []
    fork_plate_dists = []
    knife_plate_dists = []
    print(f"episodes={len(entries)}")
    print(
        "fixed plate raw tvec="
        f"({plate_raw[0]:.3f}, {plate_raw[1]:.3f}, {plate_raw[2]:.3f}); "
        "world="
        f"({PLATE_WORLD_POS[0]:.3f}, {PLATE_WORLD_POS[1]:.3f}, {PLATE_WORLD_POS[2]:.3f})"
    )
    for name in ("fork", "knife", "plate"):
        xs, ys, zs = [], [], []
        for entry in entries:
            obj = next(o for o in entry["objects"] if o["object_name"] == name)
            x, y, z = obj["tvec"]
            xs.append(x)
            ys.append(y)
            zs.append(z)
        print(
            f"{name}: raw x=[{min(xs):.3f}, {max(xs):.3f}], "
            f"y=[{min(ys):.3f}, {max(ys):.3f}], z=[{min(zs):.3f}, {max(zs):.3f}]"
        )
    for entry in entries:
        points = {o["object_name"]: tuple(o["tvec"][:2]) for o in entry["objects"]}
        fork_knife_dists.append(dist_xy(points["fork"], points["knife"]))
        fork_plate_dists.append(dist_xy(points["fork"], points["plate"]))
        knife_plate_dists.append(dist_xy(points["knife"], points["plate"]))
    print(
        "fork-knife raw XY distance="
        f"[{min(fork_knife_dists):.3f}, {max(fork_knife_dists):.3f}], "
        f"mean={sum(fork_knife_dists) / len(fork_knife_dists):.3f}"
    )
    print(
        "fork/knife-to-plate raw XY distance min="
        f"{min(min(fork_plate_dists), min(knife_plate_dists)):.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cutlery object_poses.json with fixed plate and diverse fork/knife raw poses."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="generic",
        help=(
            "Generation preset. 'wide-plate-keepout' enlarges the plate keep-out radius; "
            "'near-eval-init' samples fork/knife around eval/cutlery_arrangement_eval.py initial poses."
        ),
    )
    parser.add_argument("--count", type=int, default=200, help="Number of full episodes to generate.")
    parser.add_argument("--seed", type=int, default=2026052403, help="Deterministic random seed.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path.")
    parser.add_argument("--video-name", default=None)
    parser.add_argument(
        "--min-pair-dist",
        type=float,
        default=None,
        help="Minimum raw XY distance between fork and knife.",
    )
    parser.add_argument(
        "--min-plate-dist",
        type=float,
        default=None,
        help="Minimum raw XY distance from fork/knife to the fixed plate.",
    )
    parser.add_argument(
        "--max-plate-dist",
        type=float,
        default=None,
        help="Maximum raw XY distance from plate when near-plate sampling is selected.",
    )
    parser.add_argument(
        "--near-plate-ratio",
        type=float,
        default=None,
        help="Fraction of fork/knife samples drawn from an annulus around the fixed plate.",
    )
    parser.add_argument(
        "--pair-dist-bins",
        default=None,
        help=(
            "Comma-separated raw XY fork-knife distance bins. "
            "Episodes cycle through these bins, e.g. '0.06:0.14,0.14:0.24'."
        ),
    )
    parser.add_argument(
        "--min-world-y",
        type=float,
        default=None,
        help="Optional strict lower bound for generated fork/knife world-frame y values.",
    )
    parser.add_argument(
        "--near-eval-x-radius",
        type=float,
        default=0.14,
        help="Raw-frame ellipse x radius around eval initial fork/knife positions for --preset near-eval-init.",
    )
    parser.add_argument(
        "--near-eval-y-radius",
        type=float,
        default=0.18,
        help="Raw-frame ellipse y radius around eval initial fork/knife positions for --preset near-eval-init.",
    )
    return parser.parse_args()


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    defaults = PRESETS[args.preset]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def main() -> None:
    args = resolve_args(parse_args())
    pair_dist_bins = parse_pair_dist_bins(args.pair_dist_bins)
    if args.preset == "near-eval-init":
        entries = build_near_eval_entries(
            count=args.count,
            seed=args.seed,
            min_pair_dist=args.min_pair_dist,
            min_plate_dist=args.min_plate_dist,
            pair_dist_bins=pair_dist_bins,
            near_eval_x_radius=args.near_eval_x_radius,
            near_eval_y_radius=args.near_eval_y_radius,
            video_name=args.video_name,
        )
    else:
        entries = build_entries(
            count=args.count,
            seed=args.seed,
            min_pair_dist=args.min_pair_dist,
            min_plate_dist=args.min_plate_dist,
            max_plate_dist=args.max_plate_dist,
            near_plate_ratio=args.near_plate_ratio,
            pair_dist_bins=pair_dist_bins,
            min_world_y=args.min_world_y,
            video_name=args.video_name,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(entries, f, indent=4)
        f.write("\n")

    print(args.output)
    summarize(entries)


if __name__ == "__main__":
    main()
