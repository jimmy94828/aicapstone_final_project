"""State machine for the advanced dining cleanup task."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from isaaclab.utils.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
)

from pxr import Usd, UsdGeom, Sdf, Gf, Vt
from simulator.tasks.dining_cleanup.dining_cleanup_env_cfg import (
    _create_wipe_vis_mesh,
)


from leisaac.datagen.state_machine.base import StateMachineBase

_BOWL_NAME = "bowl"
_SPOON_NAME = "spoon"
_TRAY_NAME = "tray"
_CLOTH_NAME = "cloth"
_TISSUE_NAME = "tissue"
_VASE_NAME = "vase"
_EE_BODY_NAME = "panda_hand"
_FRANKA_ARM_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0

_MAX_CARTESIAN_DELTA = 0.02
_MAX_ROT_DELTA = 0.08
_IK_DLS_LAMBDA = 0.01

_HOVER_Z_OFFSET = 0.35
_LIFT_Z_OFFSET = 0.37
_RELEASE_Z_OFFSET = 0.2
_WIPE_CONTACT_Z = 0.06
_WIPE_HOVER_Z = 0.35

_GRIPPER_DOWN_ROLL_W = math.pi
_GRIPPER_DOWN_PITCH_W = 0.0
_GRIPPER_DOWN_YAW_OFFSET_RANGE = (-0.12, 0.12)
_GRASP_YAW_OFFSET = math.pi / 2.0


_GRASP_X_OFFSET_Bowl = -0.06  # bowl edge grasp: pull target from center toward robot
_GRASP_Y_OFFSET_Bowl = -0.03
_GRASP_RETREAT_PER_OBJECT: dict[str, float] = {
    # _BOWL_NAME: -0.075,  # bowl edge grasp: pull target from center toward robot
    _SPOON_NAME: 0.020,
    _CLOTH_NAME: 0.000,
}
_GRASP_Z_OFFSET_PER_OBJECT: dict[str, float] = {
    _BOWL_NAME: 0.15,
    _SPOON_NAME: 0.040,
    _CLOTH_NAME: 0.035,
}
_GRASP_Z_AT_CLOSE_PER_OBJECT: dict[str, float] = {
    _BOWL_NAME: 0.15,
    _SPOON_NAME: 0.030,
    _CLOTH_NAME: 0.035,
}

_DROP_X_OFFSET_PER_OBJECT: dict[str, float] = {
    _BOWL_NAME: -0.03,
    _SPOON_NAME: 0.02,
}
_DROP_Y_OFFSET_PER_OBJECT: dict[str, float] = {
    _BOWL_NAME: +0.055,
    _SPOON_NAME: -0.055,
}

# World-frame table regions.  In this advanced task, +x is the Franka-view
# right side and -x is the Franka-view left side.
_LEFT_TABLE_X_RANGE = (0.0, 0.22)
_LEFT_TABLE_Y_RANGE = (-0.50, -0.10)
# _WIPE_LANES_X = (0.12, 0.15, 0.2, 0.22)
_WIPE_LANES_X = (0.21, 0.18, 0.15, 0.11 , 0.07)  # wipe from right to left to reduce risk of pushing objects off the table
_CLOTH_FOOTPRINT_SIZE = (0.055, 0.115)
_WIPE_REQUIRED_IDEAL_FRACTION = 0.70
_WIPE_COVERAGE_RESOLUTION = 0.005
_WIPE_CONTACT_Z_RANGE = (0.05, 0.09)
_STATIC_OBJECT_XY_TOL = 0.035
_STATIC_OBJECT_INITIAL_XY = {
    _TISSUE_NAME: (0.35, -0.12),
    _VASE_NAME: (0.35, -0.26),
}
_TRAY_INITIAL_XY = (0.57, -0.36)
_TRAY_XY_TOL = 0.05
_FALL_THRESHOLD_Z = 0.0

_TRAY_SUCCESS_X_HALF_WIDTH = 0.13
_TRAY_SUCCESS_Y_HALF_WIDTH = 0.14

_FRANKA_REST_JOINT_POS = {
    "panda_joint1": 0.0,
    "panda_joint2": -math.pi / 4.0,
    "panda_joint3": 0.0,
    "panda_joint4": -3.0 * math.pi / 4.0,
    "panda_joint5": 0.0,
    "panda_joint6": math.pi / 2.0,
    "panda_joint7": math.pi / 4.0,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}


@dataclass(frozen=True)
class _EventSpec:
    kind: str
    subject: str
    duration: int


def _build_events() -> tuple[_EventSpec, ...]:
    events: list[_EventSpec] = []
    for obj_name in (_BOWL_NAME, _SPOON_NAME):
        events.extend(
            [
                _EventSpec("move_above_object", obj_name, 180),
                _EventSpec("approach_object", obj_name, 130),
                _EventSpec("grasp_object", obj_name, 25),
                _EventSpec("lift_object", obj_name, 130),
                _EventSpec("move_above_drop", obj_name, 170),
                _EventSpec("lower_to_release", obj_name, 80),
                _EventSpec("retreat_from_drop", obj_name, 100),
            ]
        )

    events.extend(
        [
            _EventSpec("move_above_object", _CLOTH_NAME, 130),
            _EventSpec("approach_object", _CLOTH_NAME, 130),
            _EventSpec("grasp_object", _CLOTH_NAME, 30),
            _EventSpec("lift_object", _CLOTH_NAME, 110),
            _EventSpec("move_above_wipe_start", _CLOTH_NAME, 120),
            _EventSpec("lower_to_wipe", _CLOTH_NAME, 80),
        ]
    )

    for lane_idx in range(len(_WIPE_LANES_X)):
        events.append(_EventSpec("wipe_sweep", str(lane_idx), 200))
        if lane_idx < len(_WIPE_LANES_X) - 1:
            events.append(_EventSpec("wipe_shift", str(lane_idx), 200))
    # events.append(_EventSpec("wipe_lift_finish", _CLOTH_NAME, 160))
    events.append(_EventSpec("lift_object", _CLOTH_NAME, 110))
    events.append(_EventSpec("move_above_drop", _CLOTH_NAME, 160))
    events.append(_EventSpec("lower_to_release", _CLOTH_NAME, 60))
    events.append(_EventSpec("retreat_from_drop", _CLOTH_NAME, 40))
    return tuple(events)


_EVENTS = _build_events()


def _rect_union_area(rects: list[tuple[float, float, float, float]]) -> float:
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


def _planned_wipe_coverage_ratio() -> float:
    x_min, x_max = _LEFT_TABLE_X_RANGE
    y_min, y_max = _LEFT_TABLE_Y_RANGE
    half_x = 0.5 * _CLOTH_FOOTPRINT_SIZE[0]
    half_y = 0.5 * _CLOTH_FOOTPRINT_SIZE[1]
    rects: list[tuple[float, float, float, float]] = []
    for lane_x in _WIPE_LANES_X:
        rect = (
            max(x_min, lane_x - half_x),
            max(y_min, y_min - half_y),
            min(x_max, lane_x + half_x),
            min(y_max, y_max + half_y),
        )
        if rect[2] > rect[0] and rect[3] > rect[1]:
            rects.append(rect)
    target_area = (x_max - x_min) * (y_max - y_min)
    if target_area <= 0.0:
        return 0.0
    return _rect_union_area(rects) / target_area


_WIPE_IDEAL_COVERAGE_RATIO = _planned_wipe_coverage_ratio()
_WIPE_COVERAGE_THRESHOLD = _WIPE_IDEAL_COVERAGE_RATIO * _WIPE_REQUIRED_IDEAL_FRACTION


def _constant_gripper(num_envs: int, device: torch.device, value: float) -> torch.Tensor:
    return torch.full((num_envs, 1), value, device=device)


def _clamp_delta(delta: torch.Tensor, max_norm: float = _MAX_CARTESIAN_DELTA) -> torch.Tensor:
    norm = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
    scale = torch.clamp(max_norm / norm, max=1.0)
    return delta * scale


def _shortest_quat(quat: torch.Tensor) -> torch.Tensor:
    return torch.where(quat[:, 0:1] < 0.0, -quat, quat)


def _retreat_xy_toward(
    target_pos_w: torch.Tensor,
    anchor_pos_w: torch.Tensor,
    distance: float,
) -> torch.Tensor:
    out = target_pos_w.clone()
    delta_xy = out[:, :2] - anchor_pos_w[:, :2]
    norm = torch.linalg.norm(delta_xy, dim=-1, keepdim=True).clamp_min(1e-6)
    out[:, :2] -= distance * (delta_xy / norm)
    return out


def _yaw_from_quat_wxyz(quat_wxyz: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _find_body_index(robot, body_name: str) -> int:
    if hasattr(robot, "find_bodies"):
        body_ids, _ = robot.find_bodies(body_name)
        if len(body_ids) > 0:
            return int(body_ids[0])

    body_names = getattr(robot.data, "body_names", None)
    if body_names is not None and body_name in body_names:
        return body_names.index(body_name)

    return -1


class DiningCleanupStateMachine(StateMachineBase):
    """Scripted Franka policy for clearing and wiping the dining table.

    Behavior:
    1. Grasp the bowl by an edge-facing target and place it in the tray.
    2. Grasp the spoon and place it in the tray.
    3. Grasp the cloth, lower it to the table, and sweep three y-axis lanes over
       the left table region where the bowl/spoon originally started.

    The action vector is ``[panda_joint1, ..., panda_joint7, gripper]``.
    """

    MAX_STEPS: int = sum(event.duration for event in _EVENTS) + 100

    def __init__(self) -> None:
        self._step_count: int = 0
        self._episode_done: bool = False
        self._wipe_complete: bool = False
        self._ee_body_idx: int = -1
        self._jacobi_body_idx: int = -1
        self._arm_joint_ids: list[int] = []
        self._jacobi_joint_ids: list[int] = []
        self._rest_joint_pos: torch.Tensor | None = None
        self._rest_ee_pos_w: torch.Tensor | None = None
        self._initial_ee_pos_w: torch.Tensor | None = None
        self._gripper_down_yaw_w: torch.Tensor | None = None
        self._gripper_down_yaw_offset_w: torch.Tensor | None = None
        self._event: int = 0
        self._lift_start_ee_xy_w: torch.Tensor | None = None
        self._wipe_covered: torch.Tensor | None = None

    def setup(self, env) -> None:
        robot = env.scene["robot"]
        self._ee_body_idx = _find_body_index(robot, _EE_BODY_NAME)
        joint_names = list(robot.data.joint_names)
        missing = [j for j in _FRANKA_ARM_JOINT_NAMES if j not in joint_names]
        if missing:
            raise ValueError(f"Missing Franka joints {missing} in {joint_names}")
        self._arm_joint_ids = [joint_names.index(j) for j in _FRANKA_ARM_JOINT_NAMES]

        if self._ee_body_idx < 0:
            raise ValueError(f"Could not find body '{_EE_BODY_NAME}' in Franka.")
        if robot.is_fixed_base:
            self._jacobi_body_idx = self._ee_body_idx - 1
            self._jacobi_joint_ids = self._arm_joint_ids
        else:
            self._jacobi_body_idx = self._ee_body_idx
            self._jacobi_joint_ids = [jid + 6 for jid in self._arm_joint_ids]

        self._rest_joint_pos = torch.zeros(env.num_envs, len(joint_names), device=env.device)
        for idx, name in enumerate(joint_names):
            if name in _FRANKA_REST_JOINT_POS:
                self._rest_joint_pos[:, idx] = _FRANKA_REST_JOINT_POS[name]

        robot.write_joint_state_to_sim(
            position=self._rest_joint_pos,
            velocity=torch.zeros_like(self._rest_joint_pos),
        )
        env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
        self._rest_ee_pos_w = self._ee_pos_w(robot).clone()

        # Wipe-coverage visualization mesh.
        
        stage = env.sim.stage
        x_bins = max(1, math.ceil((_LEFT_TABLE_X_RANGE[1] - _LEFT_TABLE_X_RANGE[0]) / _WIPE_COVERAGE_RESOLUTION))
        y_bins = max(1, math.ceil((_LEFT_TABLE_Y_RANGE[1] - _LEFT_TABLE_Y_RANGE[0]) / _WIPE_COVERAGE_RESOLUTION))
        self._wipe_vis_x_bins = x_bins
        self._wipe_vis_y_bins = y_bins

        for idx in range(env.num_envs):
            origin = env.scene.env_origins[idx]
            ox, oy = float(origin[0]), float(origin[1])
            mesh_path = f"/World/envs/env_{idx}/Scene/wipe_vis_plane"
            if not stage.GetPrimAtPath(mesh_path).IsValid():
                _create_wipe_vis_mesh(
                    stage,
                    mesh_path,
                    x_range=(ox + _LEFT_TABLE_X_RANGE[0], ox + _LEFT_TABLE_X_RANGE[1]),
                    y_range=(oy + _LEFT_TABLE_Y_RANGE[0], oy + _LEFT_TABLE_Y_RANGE[1]),
                    x_bins=x_bins,
                    y_bins=y_bins,
                    z= 0.045,
                )

    def check_success(self, env) -> bool:
        status = self._success_status(env)
        self._print_success_status(status)
        return bool(status["overall"].all().item())

    def _success_status(self, env) -> dict[str, torch.Tensor]:
        tray_pos = env.scene[_TRAY_NAME].data.root_pos_w - env.scene.env_origins
        bowl_pos = env.scene[_BOWL_NAME].data.root_pos_w - env.scene.env_origins
        spoon_pos = env.scene[_SPOON_NAME].data.root_pos_w - env.scene.env_origins
        cloth_pos = env.scene[_CLOTH_NAME].data.root_pos_w - env.scene.env_origins
        tissue_pos = env.scene[_TISSUE_NAME].data.root_pos_w - env.scene.env_origins
        vase_pos = env.scene[_VASE_NAME].data.root_pos_w - env.scene.env_origins


        timeline_done = torch.full((env.num_envs,), self._wipe_complete, dtype=torch.bool, device=env.device)
        bowl_in_tray_xy = torch.logical_and(
            torch.abs(bowl_pos[:, 0] - tray_pos[:, 0]) <= _TRAY_SUCCESS_X_HALF_WIDTH,
            torch.abs(bowl_pos[:, 1] - tray_pos[:, 1]) <= _TRAY_SUCCESS_Y_HALF_WIDTH,
        )
        spoon_in_tray_xy = torch.logical_and(
            torch.abs(spoon_pos[:, 0] - tray_pos[:, 0]) <= _TRAY_SUCCESS_X_HALF_WIDTH,
            torch.abs(spoon_pos[:, 1] - tray_pos[:, 1]) <= _TRAY_SUCCESS_Y_HALF_WIDTH,
        )
        tableware_done = torch.logical_and(bowl_in_tray_xy, spoon_in_tray_xy)

        coverage_ratio = self._update_wipe_coverage(env)
        coverage_done = coverage_ratio >= _WIPE_COVERAGE_THRESHOLD

        tray_expected = torch.tensor(_TRAY_INITIAL_XY, dtype=tray_pos.dtype, device=tray_pos.device)
        tray_dist = torch.norm(tray_pos[:, :2] - tray_expected, dim=1)
        tray_stable = tray_dist <= _TRAY_XY_TOL

        protected_stable = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        protected_dists: dict[str, torch.Tensor] = {}
        for obj_name, expected_xy in _STATIC_OBJECT_INITIAL_XY.items():
            obj_pos = env.scene[obj_name].data.root_pos_w - env.scene.env_origins
            expected = torch.tensor(expected_xy, dtype=obj_pos.dtype, device=obj_pos.device)
            obj_dist = torch.norm(obj_pos[:, :2] - expected, dim=1)
            protected_dists[obj_name] = obj_dist
            protected_stable = torch.logical_and(protected_stable, obj_dist <= _STATIC_OBJECT_XY_TOL)

        no_non_tableware_fall = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for obj_name in (_TRAY_NAME, _CLOTH_NAME, _TISSUE_NAME, _VASE_NAME):
            obj_pos = env.scene[obj_name].data.root_pos_w - env.scene.env_origins
            no_non_tableware_fall = torch.logical_and(no_non_tableware_fall, obj_pos[:, 2] >= _FALL_THRESHOLD_Z)

        cloth_in_tray_xy = torch.logical_and(
            torch.abs(cloth_pos[:, 0] - tray_pos[:, 0]) <= _TRAY_SUCCESS_X_HALF_WIDTH,
            torch.abs(cloth_pos[:, 1] - tray_pos[:, 1]) <= _TRAY_SUCCESS_Y_HALF_WIDTH,
        )

        wiping_done = torch.logical_and(coverage_done, cloth_in_tray_xy)

        bowl_tray_abs = torch.abs(bowl_pos[:, :2] - tray_pos[:, :2])
        spoon_tray_abs = torch.abs(spoon_pos[:, :2] - tray_pos[:, :2])
        cloth_tray_abs = torch.abs(cloth_pos[:, :2] - tray_pos[:, :2])
        cloth_contact = torch.logical_and(
            cloth_pos[:, 2] >= _WIPE_CONTACT_Z_RANGE[0],
            cloth_pos[:, 2] <= _WIPE_CONTACT_Z_RANGE[1],
        )
        non_tableware_min_z = torch.stack(
            [tray_pos[:, 2], cloth_pos[:, 2], tissue_pos[:, 2], vase_pos[:, 2]],
            dim=0,
        ).amin(dim=0)

        overall = timeline_done
        for term in (tableware_done, coverage_done, tray_stable, protected_stable, no_non_tableware_fall, wiping_done):
            overall = torch.logical_and(overall, term)

        return {
            "timeline": timeline_done,
            "bowl_xy": bowl_in_tray_xy,
            "spoon_xy": spoon_in_tray_xy,
            "cloth_xy": cloth_in_tray_xy,
            "tableware": tableware_done,
            "coverage": coverage_done,
            "wiping": wiping_done,
            "tray_stable": tray_stable,
            "protected": protected_stable,
            "no_non_tableware_fall": no_non_tableware_fall,
            "overall": overall,
            "coverage_ratio": coverage_ratio,
            "bowl_tray_dx": bowl_tray_abs[:, 0],
            "bowl_tray_dy": bowl_tray_abs[:, 1],
            "spoon_tray_dx": spoon_tray_abs[:, 0],
            "spoon_tray_dy": spoon_tray_abs[:, 1],
            "cloth_tray_dx": cloth_tray_abs[:, 0],
            "cloth_tray_dy": cloth_tray_abs[:, 1],
            "cloth_z": cloth_pos[:, 2],
            "cloth_contact": cloth_contact,
            "tray_dist": tray_dist,
            "tissue_dist": protected_dists[_TISSUE_NAME],
            "vase_dist": protected_dists[_VASE_NAME],
            "non_tableware_min_z": non_tableware_min_z,
        }

    def _print_success_status(self, status: dict[str, torch.Tensor]) -> None:
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"

        def word(name: str) -> str:
            ok = bool(status[name].all().item())
            color = GREEN if ok else RED
            text = "PASS" if ok else "FAIL"
            return f"{color}{text}{RESET}"

        def line(key: str, value: str, indent: int = 0) -> str:
            return f"{' ' * indent}{key:<24}: {value}"

        def values(name: str) -> str:
            return ", ".join(f"{value:.3f}" for value in status[name].detach().cpu().tolist())

        print(
            "\n".join(
                [
                    "[DiningCleanup FSM]",
                    line("timeline", word("timeline")),
                    line("tableware", word("tableware")),
                    line("bowl_xy", word("bowl_xy"), indent=2),
                    line("bowl tray dx/dy", f"{values('bowl_tray_dx')} / {values('bowl_tray_dy')}", indent=2),
                    line("spoon_xy", word("spoon_xy"), indent=2),
                    line("spoon tray dx/dy", f"{values('spoon_tray_dx')} / {values('spoon_tray_dy')}", indent=2),
                    line("cloth_xy", word("cloth_xy"), indent=2),
                    line("cloth tray dx/dy", f"{values('cloth_tray_dx')} / {values('cloth_tray_dy')}", indent=2),
                    line("cloth z/contact", f"{values('cloth_z')} / {word('cloth_contact')}", indent=2),
                    line("wiping", word("wiping")),
                    line("coverage", word("coverage"), indent=2),
                    line("coverage ratio", values("coverage_ratio"), indent=2),
                    line("threshold", f"{_WIPE_COVERAGE_THRESHOLD:.3f}", indent=2),
                    line("ideal", f"{_WIPE_IDEAL_COVERAGE_RATIO:.3f}", indent=2),
                    line(
                        "required",
                        f"{_WIPE_REQUIRED_IDEAL_FRACTION:.0%}",
                        indent=2,
                    ),
                    line("tray_dist", values("tray_dist")),
                    line("tray_stable", word("tray_stable")),
                    line("protected", word("protected")),
                    line("tissue_dist", values("tissue_dist"), indent=2),
                    line("vase_dist", values("vase_dist"), indent=2),
                    line("no_non_tableware_fall", word("no_non_tableware_fall")),
                    line("min non-tableware z", values("non_tableware_min_z"), indent=2),
                    line("overall", word("overall")),
                ]
            ),
            flush=True,
        )

    def pre_step(self, env) -> None:
        self._update_wipe_coverage(env)
        self._sync_wipe_vis(env) 

    def get_action(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)

        device = env.device
        num_envs = env.num_envs
        event = _EVENTS[self._event]

        active_obj_name = event.subject if event.subject in (_BOWL_NAME, _SPOON_NAME, _CLOTH_NAME) else _CLOTH_NAME
        active_obj = env.scene[active_obj_name]
        active_pos_w = active_obj.data.root_pos_w.clone()
        active_quat_w = active_obj.data.root_quat_w.clone()
        tray_pos_w = env.scene[_TRAY_NAME].data.root_pos_w.clone()
        robot_root_pos_w = robot.data.root_pos_w.clone()
        grasp_anchor_w = self._grasp_anchor_w(active_obj_name, active_pos_w, robot_root_pos_w)

        if self._step_count == 0 and event.kind in ("move_above_object", "move_above_wipe_start"):
            self._initial_ee_pos_w = self._ee_pos_w(robot).clone()

        target_quat_w = self._target_orientation(
            event,
            active_obj_name,
            active_quat_w,
            num_envs,
            device,
        )

        if event.kind == "move_above_object":
            target_pos_w, gripper_cmd = self._phase_move_above_object(grasp_anchor_w, num_envs, device)
        elif event.kind == "approach_object":
            target_pos_w, gripper_cmd = self._phase_approach_object(active_obj_name, grasp_anchor_w, num_envs, device)
        elif event.kind == "grasp_object":
            target_pos_w, gripper_cmd = self._phase_grasp(active_obj_name, grasp_anchor_w, num_envs, device)
        elif event.kind == "lift_object":
            target_pos_w, gripper_cmd = self._phase_lift(self._ee_pos_w(robot), active_pos_w, num_envs, device)
        elif event.kind == "move_above_drop":
            target_pos_w, gripper_cmd = self._phase_move_above_drop(
                self._drop_target_w(active_obj_name, tray_pos_w), num_envs, device
            )
        elif event.kind == "lower_to_release":
            target_pos_w, gripper_cmd = self._phase_lower_to_release(
                self._drop_target_w(active_obj_name, tray_pos_w), num_envs, device
            )
        elif event.kind == "retreat_from_drop":
            target_pos_w, gripper_cmd = self._phase_retreat_from_drop(
                self._drop_target_w(active_obj_name, tray_pos_w), num_envs, device
            )
        elif event.kind == "move_above_wipe_start":
            target_pos_w, gripper_cmd = self._phase_move_above_wipe_start(env)
        elif event.kind == "lower_to_wipe":
            target_pos_w, gripper_cmd = self._phase_lower_to_wipe(env)
        elif event.kind == "wipe_sweep":
            target_pos_w, gripper_cmd = self._phase_wipe_sweep(env, int(event.subject))
        elif event.kind == "wipe_shift":
            target_pos_w, gripper_cmd = self._phase_wipe_shift(env, int(event.subject))
        else:
            target_pos_w, gripper_cmd = self._phase_wipe_lift_finish(env)

        return self._joint_position_franka_action(env, target_pos_w, target_quat_w, gripper_cmd)

    def _target_orientation(
        self,
        event,
        obj_name,
        obj_quat_w,
        num_envs,
        device,
    ):
        # Align the utensil before dropping it.
        if (
            obj_name == _SPOON_NAME
            and event.kind in ("move_above_drop", "lower_to_release")
        ):
            roll = torch.full(
                (num_envs,),
                _GRIPPER_DOWN_ROLL_W,
                device=device,
                dtype=obj_quat_w.dtype,
            )

            pitch = torch.zeros_like(roll)

            # Face world +X.
            yaw = torch.zeros_like(roll)

            return quat_from_euler_xyz(
                roll,
                pitch,
                yaw,
            )

        return self._gripper_down_quat_w(
            obj_quat_w,
            obj_name,
            num_envs,
            device,
            obj_quat_w.dtype,
            yaw_offset=_GRASP_YAW_OFFSET,
        )
    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _phase_alpha(self) -> float:
        denom = max(_EVENTS[self._event].duration - 1, 1)
        return min(self._step_count / denom, 1.0)

    def _update_wipe_coverage(self, env) -> torch.Tensor:
        cloth_pos = env.scene[_CLOTH_NAME].data.root_pos_w - env.scene.env_origins
        x_min, x_max = _LEFT_TABLE_X_RANGE
        y_min, y_max = _LEFT_TABLE_Y_RANGE
        x_bins = max(1, math.ceil((x_max - x_min) / _WIPE_COVERAGE_RESOLUTION))
        y_bins = max(1, math.ceil((y_max - y_min) / _WIPE_COVERAGE_RESOLUTION))
        expected_shape = (env.num_envs, x_bins, y_bins)
        if self._wipe_covered is None or self._wipe_covered.shape != expected_shape:
            self._wipe_covered = torch.zeros(expected_shape, dtype=torch.bool, device=env.device)

        dx = (x_max - x_min) / x_bins
        dy = (y_max - y_min) / y_bins
        grid_x = torch.linspace(
            x_min + 0.5 * dx,
            x_max - 0.5 * dx,
            x_bins,
            dtype=cloth_pos.dtype,
            device=cloth_pos.device,
        )
        grid_y = torch.linspace(
            y_min + 0.5 * dy,
            y_max - 0.5 * dy,
            y_bins,
            dtype=cloth_pos.dtype,
            device=cloth_pos.device,
        )
        half_x = 0.5 * _CLOTH_FOOTPRINT_SIZE[0]
        half_y = 0.5 * _CLOTH_FOOTPRINT_SIZE[1]
        in_contact = torch.logical_and(
            cloth_pos[:, 2] >= _WIPE_CONTACT_Z_RANGE[0],
            cloth_pos[:, 2] <= _WIPE_CONTACT_Z_RANGE[1],
        )
        covered_now = torch.logical_and(
            torch.abs(cloth_pos[:, 0, None, None] - grid_x[None, :, None]) <= half_x,
            torch.abs(cloth_pos[:, 1, None, None] - grid_y[None, None, :]) <= half_y,
        )
        self._wipe_covered |= torch.logical_and(covered_now, in_contact[:, None, None])
        return self._wipe_covered.float().mean(dim=(1, 2))

    def _sync_wipe_vis(self, env) -> None:
        """Write wipe coverage state to USD vertex colors (env 0 only for perf)."""
        if self._wipe_covered is None:
            return
        from pxr import Gf, UsdGeom, Vt

        stage = env.sim.stage
        # Only visualize env 0 to avoid per-frame USD writes for all envs
        env_idx = 0
        mesh_path = f"/World/envs/env_{env_idx}/Scene/wipe_vis_plane"
        prim = stage.GetPrimAtPath(mesh_path)
        if not prim.IsValid():
            return

        state = self._wipe_covered[env_idx].float()   # (x_bins, y_bins)
        # heatmap: 0=dirty(blue) -> 1=clean(red-green peak)
        c = state.cpu()
        colors = []
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                # v = float(c[i, j])
                if c[i, j] <= 0.0:
                    colors.append(Gf.Vec3f(0.36, 0.25, 0.20))
                else:
                    colors.append(Gf.Vec3f(1.0, 1.0, 1.0))

        attr = UsdGeom.Mesh(prim).GetDisplayColorAttr()
        attr.Set(Vt.Vec3fArray(colors))

    def _grasp_anchor_w(self, obj_name: str, obj_pos_w: torch.Tensor, robot_root_pos_w: torch.Tensor) -> torch.Tensor:
        if obj_name == _CLOTH_NAME:
            return obj_pos_w.clone()
        if obj_name == _BOWL_NAME:
            target = obj_pos_w.clone()
            target[:, 0] += _GRASP_X_OFFSET_Bowl
            target[:, 1] += _GRASP_Y_OFFSET_Bowl
            return target
        return _retreat_xy_toward(
            obj_pos_w,
            robot_root_pos_w,
            _GRASP_RETREAT_PER_OBJECT.get(obj_name, 0.0),
        )

    def _drop_target_w(self, obj_name: str, tray_pos_w: torch.Tensor) -> torch.Tensor:
        target = tray_pos_w.clone()
        target[:, 0] += _DROP_X_OFFSET_PER_OBJECT.get(obj_name, 0.0)
        target[:, 1] += _DROP_Y_OFFSET_PER_OBJECT.get(obj_name, 0.0)
        return target

    def _phase_move_above_object(self, obj_pos_w, num_envs, device):
        target = obj_pos_w.clone()
        target[:, 2] += _HOVER_Z_OFFSET
        if self._initial_ee_pos_w is not None:
            alpha = self._phase_alpha()
            target = (1.0 - alpha) * self._initial_ee_pos_w + alpha * target
        return target, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def _phase_approach_object(self, obj_name, obj_pos_w, num_envs, device):
        target = obj_pos_w.clone()
        target[:, 2] += _GRASP_Z_OFFSET_PER_OBJECT.get(obj_name, 0.04)
        return target, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def _phase_grasp(self, obj_name, obj_pos_w, num_envs, device):
        target = obj_pos_w.clone()
        target[:, 2] += _GRASP_Z_AT_CLOSE_PER_OBJECT.get(obj_name, 0.03)
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_lift(self, ee_pos_w, obj_pos_w, num_envs, device):
        if self._step_count == 0 or self._lift_start_ee_xy_w is None:
            self._lift_start_ee_xy_w = ee_pos_w[:, :2].clone()
        target = obj_pos_w.clone()
        target[:, :2] = self._lift_start_ee_xy_w
        target[:, 2] += _LIFT_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_move_above_drop(self, drop_pos_w, num_envs, device):
        target = drop_pos_w.clone()
        target[:, 2] += _LIFT_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_lower_to_release(self, drop_pos_w, num_envs, device):
        target = drop_pos_w.clone()
        target[:, 2] += _RELEASE_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_retreat_from_drop(self, drop_pos_w, num_envs, device):
        target = drop_pos_w.clone()
        target[:, 2] += _LIFT_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def _wipe_point_w(self, env, x: float, y: float, z: float) -> torch.Tensor:
        target = env.scene.env_origins.clone()
        target[:, 0] += x
        target[:, 1] += y
        target[:, 2] += z
        return target

    def _wipe_lane_endpoint(self, lane_idx: int, at_end: bool) -> tuple[float, float]:
        x = _WIPE_LANES_X[lane_idx]
        y0, y1 = _LEFT_TABLE_Y_RANGE
        y = y1 if at_end else y0
        return x, y

    def _phase_move_above_wipe_start(self, env):
        x, y = self._wipe_lane_endpoint(0, at_end=False)
        target = self._wipe_point_w(env, x, y, _WIPE_HOVER_Z)
        return target, _constant_gripper(env.num_envs, env.device, _GRIPPER_CLOSE)

    def _phase_lower_to_wipe(self, env):
        x, y = self._wipe_lane_endpoint(0, at_end=False)
        if self._phase_alpha() == 1:
            print(x,y)
        return self._wipe_point_w(env, x, y, _WIPE_CONTACT_Z), _constant_gripper(
            env.num_envs, env.device, _GRIPPER_CLOSE
        )

    def _phase_wipe_sweep(self, env, lane_idx: int):
        x, y = self._wipe_lane_endpoint(lane_idx, at_end=True)
        if self._phase_alpha() == 1:
            print(x,y)
        return self._wipe_point_w(env, x, y, _WIPE_CONTACT_Z), _constant_gripper(
            env.num_envs, env.device, _GRIPPER_CLOSE
        )

    def _phase_wipe_shift(self, env, lane_idx: int):
        x, y = self._wipe_lane_endpoint(lane_idx + 1, at_end=False)
        if self._phase_alpha() == 1:
            print(x,y)
        return self._wipe_point_w(env, x, y, _WIPE_CONTACT_Z), _constant_gripper(
            env.num_envs, env.device, _GRIPPER_CLOSE
        )

    def _phase_wipe_lift_finish(self, env):
        x, y = self._wipe_lane_endpoint(len(_WIPE_LANES_X) - 1, at_end=True)
        return self._wipe_point_w(env, x, y, _WIPE_HOVER_Z), _constant_gripper(
            env.num_envs, env.device, _GRIPPER_CLOSE
        )

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def advance(self) -> None:
        if self._episode_done:
            return

        self._step_count += 1
        if self._step_count < _EVENTS[self._event].duration:
            return

        old_active = self._active_object_for_event(_EVENTS[self._event])
        self._event += 1
        self._step_count = 0

        if self._event >= len(_EVENTS):
            self._wipe_complete = True
            self._episode_done = True
            return

        new_active = self._active_object_for_event(_EVENTS[self._event])
        if new_active != old_active:
            self._initial_ee_pos_w = None
            self._gripper_down_yaw_w = None
            self._gripper_down_yaw_offset_w = None
            self._lift_start_ee_xy_w = None
        elif _EVENTS[self._event].kind in ("move_above_object", "move_above_wipe_start"):
            self._initial_ee_pos_w = None

    def reset(self) -> None:
        self._step_count = 0
        self._episode_done = False
        self._wipe_complete = False
        self._event = 0
        self._initial_ee_pos_w = None
        self._gripper_down_yaw_w = None
        self._gripper_down_yaw_offset_w = None
        self._lift_start_ee_xy_w = None
        self._wipe_covered = None

    def _active_object_for_event(self, event: _EventSpec) -> str:
        if event.subject in (_BOWL_NAME, _SPOON_NAME, _CLOTH_NAME):
            return event.subject
        return _CLOTH_NAME

    # ------------------------------------------------------------------
    # IK / control helpers
    # ------------------------------------------------------------------

    def _ee_pos_w(self, robot) -> torch.Tensor:
        body_idx = self._ee_body_idx if self._ee_body_idx >= 0 else -1
        return robot.data.body_pos_w[:, body_idx, :]

    def _ee_quat_w(self, robot) -> torch.Tensor:
        body_idx = self._ee_body_idx if self._ee_body_idx >= 0 else -1
        return robot.data.body_quat_w[:, body_idx, :]

    def _joint_position_franka_action(
        self,
        env,
        target_pos_w: torch.Tensor,
        target_quat_w: torch.Tensor,
        gripper_cmd: torch.Tensor,
    ) -> torch.Tensor:
        robot = env.scene["robot"]
        root_pos_w = robot.data.root_pos_w
        root_quat_w = robot.data.root_quat_w
        root_quat_inv = quat_inv(root_quat_w)

        target_pos_root = quat_apply(root_quat_inv, target_pos_w - root_pos_w)
        ee_pos_root = quat_apply(root_quat_inv, self._ee_pos_w(robot) - root_pos_w)
        delta_pos_root = _clamp_delta(target_pos_root - ee_pos_root)

        delta_quat_w = _shortest_quat(quat_mul(target_quat_w, quat_inv(self._ee_quat_w(robot))))
        delta_rot_w = axis_angle_from_quat(delta_quat_w)
        delta_rot_root = _clamp_delta(quat_apply(root_quat_inv, delta_rot_w), _MAX_ROT_DELTA)

        pose_delta_root = torch.cat([delta_pos_root, delta_rot_root], dim=-1)
        joint_pos_target = self._arm_joint_pos(robot) + self._compute_delta_joint_pos(
            pose_delta_root, self._ee_jacobian_root(robot)
        )
        joint_pos_target = self._clamp_arm_joint_pos(robot, joint_pos_target)
        return torch.cat([joint_pos_target, gripper_cmd], dim=-1)

    def _arm_joint_pos(self, robot) -> torch.Tensor:
        if not self._arm_joint_ids:
            raise RuntimeError("setup() must run before requesting actions.")
        return robot.data.joint_pos[:, self._arm_joint_ids]

    def _ee_jacobian_root(self, robot) -> torch.Tensor:
        if self._jacobi_body_idx < 0 or not self._jacobi_joint_ids:
            raise RuntimeError("setup() must run before requesting actions.")

        jacobian = robot.root_physx_view.get_jacobians()[
            :, self._jacobi_body_idx, :, self._jacobi_joint_ids
        ].clone()
        root_rot_matrix = matrix_from_quat(quat_inv(robot.data.root_quat_w))
        jacobian[:, :3, :] = torch.bmm(root_rot_matrix, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(root_rot_matrix, jacobian[:, 3:, :])
        return jacobian

    def _compute_delta_joint_pos(self, pose_delta: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        jacobian_t = torch.transpose(jacobian, dim0=1, dim1=2)
        lambda_matrix = (_IK_DLS_LAMBDA**2) * torch.eye(
            jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype
        )
        delta_joint_pos = (
            jacobian_t @ torch.inverse(jacobian @ jacobian_t + lambda_matrix) @ pose_delta.unsqueeze(-1)
        )
        return delta_joint_pos.squeeze(-1)

    def _clamp_arm_joint_pos(self, robot, joint_pos: torch.Tensor) -> torch.Tensor:
        joint_pos_limits = getattr(robot.data, "soft_joint_pos_limits", None)
        if joint_pos_limits is None:
            joint_pos_limits = getattr(robot.data, "joint_pos_limits", None)
        if joint_pos_limits is None:
            return joint_pos
        arm_joint_pos_limits = joint_pos_limits[:, self._arm_joint_ids, :]
        return torch.clamp(joint_pos, arm_joint_pos_limits[..., 0], arm_joint_pos_limits[..., 1])

    def _gripper_down_quat_w(
        self,
        obj_quat_w: torch.Tensor,
        obj_name: str,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
        yaw_offset: float = 0.0,
    ) -> torch.Tensor:
        if self._gripper_down_yaw_w is None or self._gripper_down_yaw_w.shape[0] != num_envs:
            base_yaw = _yaw_from_quat_wxyz(obj_quat_w).to(device=device, dtype=dtype)
            if obj_name in (_BOWL_NAME, _CLOTH_NAME):
                self._gripper_down_yaw_offset_w = torch.zeros(num_envs, device=device, dtype=dtype)
                # Bowl edge grasp and cloth wiping should be deterministic and vertical.
                base_yaw = torch.zeros_like(base_yaw)
            else:
                self._gripper_down_yaw_offset_w = torch.empty(num_envs, device=device, dtype=dtype).uniform_(
                    _GRIPPER_DOWN_YAW_OFFSET_RANGE[0],
                    _GRIPPER_DOWN_YAW_OFFSET_RANGE[1],
                )
            self._gripper_down_yaw_w = (
                base_yaw + yaw_offset + self._gripper_down_yaw_offset_w
            ).clone()

        roll = torch.full((num_envs,), _GRIPPER_DOWN_ROLL_W, device=device, dtype=dtype)
        pitch = torch.full((num_envs,), _GRIPPER_DOWN_PITCH_W, device=device, dtype=dtype)
        yaw = self._gripper_down_yaw_w.to(device=device, dtype=dtype)
        return quat_from_euler_xyz(roll, pitch, yaw)

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def task_object_names(self) -> tuple[str, ...]:
        return (_BOWL_NAME, _SPOON_NAME, _TRAY_NAME, _CLOTH_NAME, _TISSUE_NAME, _VASE_NAME)
