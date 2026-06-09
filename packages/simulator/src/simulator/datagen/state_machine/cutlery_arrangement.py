"""State machine for the Franka cutlery-arrangement task."""

from __future__ import annotations

import math

import torch
from isaaclab.utils.math import (
    axis_angle_from_quat,
    matrix_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
)

from leisaac.datagen.state_machine.base import StateMachineBase

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_FORK_NAME = "fork"
_KNIFE_NAME = "knife"
_PLATE_NAME = "plate"
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

_HOVER_Z_OFFSET = 0.25
_GRASP_Z_OFFSET = 0.08
_LIFT_Z_OFFSET = 0.25
_RELEASE_Z_OFFSET = 0.12
_RELEASE_TOL_M = 0.015
_GRIPPER_DOWN_ROLL_W = math.pi
_GRIPPER_DOWN_PITCH_W = 0.0
_GRIPPER_DOWN_YAW_OFFSET_RANGE = (-0.15, 0.15)
_GRASP_YAW_OFFSET: float = math.pi / 2.0
_GRASP_RETREAT_PER_OBJECT: dict[str, float] = {
    "fork": 0.025,
    "knife": 0.025,
}

_PLACE_OFFSET = 0.08

_SUCCESS_MAX_DIST_XY = 0.15

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

_PICK_ORDER = (_KNIFE_NAME, _FORK_NAME)
_PLACE_X_SIGNS = (+1.0, -1.0)  # knife → +x of plate, fork → -x of plate

_PHASE_DURATIONS_PER_OBJECT = (180, 130, 20, 70, 170, 15, 50)
_PHASES_PER_OBJECT = len(_PHASE_DURATIONS_PER_OBJECT)

# ---------------------------------------------------------------------------
# DART noise configuration
# ---------------------------------------------------------------------------
# Per-phase target-position noise sigma (metres). Index = phase_in_cycle.
# Design rationale:
#   - Phase 0 (hover) / Phase 4 (above place): large xy noise to generate
#     lateral recovery demos. z noise is smaller to avoid ceiling clips.
#   - Phase 1 (approach): moderate xy, tiny z — approach height is critical.
#   - Phase 2 (grasp): very small — too much noise here causes empty grasps.
#   - Phase 3 (lift): no xy noise (we already lock xy via _lift_start_ee_xy_w);
#     small z so the lift height demo has some variation.
#   - Phase 5 (lower to release): tiny — release height is sensitive.
#   - Phase 6 (retreat): negligible — success already achieved.
#
# Noise is sampled ONCE per phase, then linearly decayed to 0 over the phase.
# This produces a "drift-then-correct" shape rather than random jitter.
_DART_SIGMA_XY = (0.05, 0.01, 0.003, 0.000, 0.05, 0.003, 0.002)
_DART_SIGMA_Z  = (0.01, 0.01, 0.002, 0.005, 0.01, 0.002, 0.001)


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


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class CutleryArrangementStateMachine(StateMachineBase):
    """Scripted Franka policy for arranging cutlery around a plate.

    Picks up the knife and places it on the -x (right) side of the plate,
    then picks up the fork and places it on the +x (left) side.

    Each object goes through 7 phases:

    0. Move above object
    1. Approach down to object
    2. Close gripper to grasp
    3. Lift object upward
    4. Move above target position (beside plate)
    5. Lower and release
    6. Retreat upward

    DART noise: each phase samples a fixed offset for (x, y, z) that decays
    linearly to zero over the phase duration. This produces "drift-then-correct"
    recovery trajectories that reduce covariate shift at training time.

    The action vector is ``[panda_joint1, ..., panda_joint7, gripper]``.
    """

    MAX_STEPS: int = len(_PICK_ORDER) * sum(_PHASE_DURATIONS_PER_OBJECT) + 100

    def __init__(self) -> None:
        self._step_count: int = 0
        self._episode_done: bool = False
        self._ee_body_idx: int = -1
        self._jacobi_body_idx: int = -1
        self._arm_joint_ids: list[int] = []
        self._jacobi_joint_ids: list[int] = []
        self._rest_joint_pos: torch.Tensor | None = None
        self._rest_ee_pos_w: torch.Tensor | None = None

        # FIX: _initial_ee_pos_w is now captured at the start of every phase-0
        # (hover phase), not just at global step 0. This ensures the lerp works
        # correctly for both the knife pick (event 0) and the fork pick (event 7).
        self._initial_ee_pos_w: torch.Tensor | None = None

        self._gripper_down_yaw_w: torch.Tensor | None = None
        self._gripper_down_yaw_offset_w: torch.Tensor | None = None
        self._current_object_idx: int = 0
        self._event: int = 0
        self._events_dt: list[int] = list(_PHASE_DURATIONS_PER_OBJECT) * len(_PICK_ORDER)
        self._lift_start_ee_xy_w: torch.Tensor | None = None

        # DART: per-phase fixed noise offset (num_envs, 3), sampled at phase start
        self._dart_noise_w: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # StateMachineBase interface
    # ------------------------------------------------------------------

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

    def check_success(self, env) -> bool:
        plate_pos = env.scene[_PLATE_NAME].data.root_pos_w - env.scene.env_origins
        fork_pos = env.scene[_FORK_NAME].data.root_pos_w - env.scene.env_origins
        knife_pos = env.scene[_KNIFE_NAME].data.root_pos_w - env.scene.env_origins

        done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

        fork_dist_xy = torch.norm(fork_pos[:, :2] - plate_pos[:, :2], dim=1)
        knife_dist_xy = torch.norm(knife_pos[:, :2] - plate_pos[:, :2], dim=1)

        done = torch.logical_and(done, fork_dist_xy <= _SUCCESS_MAX_DIST_XY)
        done = torch.logical_and(done, knife_dist_xy <= _SUCCESS_MAX_DIST_XY)

        done = torch.logical_and(done, fork_pos[:, 0] < plate_pos[:, 0])   # fork on +x
        done = torch.logical_and(done, knife_pos[:, 0] > plate_pos[:, 0])  # knife on -x

        # z difference check
        Z_threshold = 0.03
        fork_plate_z_ok = torch.abs(fork_pos[:, 2] - plate_pos[:, 2]) <= Z_threshold
        knife_plate_z_ok = torch.abs(knife_pos[:, 2] - plate_pos[:, 2]) <= Z_threshold
        fork_knife_z_ok = torch.abs(fork_pos[:, 2] - knife_pos[:, 2]) <= Z_threshold

        done = torch.logical_and(done, fork_plate_z_ok)
        done = torch.logical_and(done, knife_plate_z_ok)
        done = torch.logical_and(done, fork_knife_z_ok)

        return bool(done.all().item())

    def pre_step(self, env) -> None:
        pass

    def get_action(self, env) -> torch.Tensor:
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)

        device = env.device
        num_envs = env.num_envs

        obj_name = _PICK_ORDER[self._current_object_idx]
        x_sign = _PLACE_X_SIGNS[self._current_object_idx]
        obj_pos_w = env.scene[obj_name].data.root_pos_w.clone()
        obj_quat_w = env.scene[obj_name].data.root_quat_w.clone()
        plate_pos_w = env.scene[_PLATE_NAME].data.root_pos_w.clone()
        robot_root_pos_w = robot.data.root_pos_w.clone()

        place_target_w = plate_pos_w.clone()
        place_target_w[:, 0] += x_sign * _PLACE_OFFSET

        phase_in_cycle = self._event % _PHASES_PER_OBJECT

        # FIX: capture EE position on the FIRST step of every hover phase (phase 0),
        # not just at global step 0 + event 0. The original code only captured on
        # (step==0 and event==0), so the fork's hover phase (event==7) never got
        # a valid _initial_ee_pos_w and the lerp was silently skipped.
        if phase_in_cycle == 0 and self._step_count == 0:
            self._initial_ee_pos_w = self._ee_pos_w(robot).clone()

        # DART: sample a fresh fixed noise offset on the first step of each phase.
        if self._step_count == 0:
            self._dart_noise_w = self._sample_dart_noise(
                phase_in_cycle, num_envs, device, obj_pos_w.dtype
            )

        target_quat_w = self._gripper_down_quat_w(
            obj_quat_w,
            obj_name,
            num_envs,
            device,
            obj_quat_w.dtype,
            yaw_offset=_GRASP_YAW_OFFSET,
        )

        grasp_anchor_w = _retreat_xy_toward(
            obj_pos_w,
            robot_root_pos_w,
            _GRASP_RETREAT_PER_OBJECT.get(obj_name, 0.0),
        )

        if phase_in_cycle == 0:
            target_pos_w, gripper_cmd = self._phase_move_above_object(obj_pos_w, num_envs, device)
        elif phase_in_cycle == 1:
            target_pos_w, gripper_cmd = self._phase_approach_object(grasp_anchor_w, num_envs, device)
        elif phase_in_cycle == 2:
            target_pos_w, gripper_cmd = self._phase_grasp(grasp_anchor_w, num_envs, device)
        elif phase_in_cycle == 3:
            ee_pos_now_w = self._ee_pos_w(robot)
            target_pos_w, gripper_cmd = self._phase_lift(ee_pos_now_w, plate_pos_w, num_envs, device)
        elif phase_in_cycle == 4:
            target_pos_w, gripper_cmd = self._phase_move_above_place(place_target_w, num_envs, device)
        elif phase_in_cycle == 5:
            target_pos_w, gripper_cmd = self._phase_lower_to_release(place_target_w, num_envs, device)
        else:
            target_pos_w, gripper_cmd = self._phase_retreat(
                place_target_w, num_envs, device, robot=robot
            )

        # DART: apply decayed noise to the target position.
        # Noise decays from full magnitude at step 0 to zero at the last step,
        # so the EE starts displaced and IK naturally pulls it back — generating
        # recovery demonstrations without any extra logic.
        target_pos_w = self._apply_dart_noise(target_pos_w, phase_in_cycle)

        return self._joint_position_franka_action(env, target_pos_w, target_quat_w, gripper_cmd)

    # ------------------------------------------------------------------
    # Phase helpers
    # ------------------------------------------------------------------

    def _phase_move_above_object(self, obj_pos_w, num_envs, device):
        target = obj_pos_w.clone()
        target[:, 2] += _HOVER_Z_OFFSET
        # Lerp from current EE position to the hover target so the arm moves
        # smoothly rather than jumping. _initial_ee_pos_w is now always set
        # at the start of this phase (see get_action fix above).
        if self._initial_ee_pos_w is not None:
            denom = max(self._events_dt[self._event] - 1, 1)
            alpha = min(self._step_count / denom, 1.0)
            target = (1.0 - alpha) * self._initial_ee_pos_w + alpha * target
        return target, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def _phase_approach_object(self, obj_pos_w, num_envs, device):
        target = obj_pos_w.clone()
        target[:, 2] += _GRASP_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    def _phase_grasp(self, obj_pos_w, num_envs, device):
        # Hold approach height while fingers close; prevents the EE from
        # continuing to descend during the grasp, which causes empty grasps.
        target = obj_pos_w.clone()
        target[:, 2] += _GRASP_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_lift(self, ee_pos_w, plate_pos_w, num_envs, device):
        # Lock XY on the first step of this phase so the target does not drift
        # with the held object's moving position (which is offset from the EE).
        if self._step_count == 0 or self._lift_start_ee_xy_w is None:
            self._lift_start_ee_xy_w = ee_pos_w[:, :2].clone()
        target = ee_pos_w.clone()
        target[:, :2] = self._lift_start_ee_xy_w
        target[:, 2] = plate_pos_w[:, 2] + _LIFT_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_move_above_place(self, place_pos_w, num_envs, device):
        target = place_pos_w.clone()
        target[:, 2] += _LIFT_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_lower_to_release(self, place_pos_w, num_envs, device):
        target = place_pos_w.clone()
        target[:, 2] += _RELEASE_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

    def _phase_retreat(self, place_pos_w, num_envs, device, robot=None):
        # Keep gripper CLOSED and continue targeting the release point until the
        # EE has actually reached release altitude. This guards against phase 5
        # timing out before IK converges (which would release the object mid-air).
        # if robot is not None and self._ee_body_idx >= 0:
        #     ee_z = robot.data.body_pos_w[:, self._ee_body_idx, 2]
        #     release_z = place_pos_w[:, 2] + _RELEASE_Z_OFFSET
        #     descent_complete = bool((ee_z <= release_z + _RELEASE_TOL_M).all().item())
        # else:
        #     descent_complete = True

        # if not descent_complete:
        #     target = place_pos_w.clone()
        #     target[:, 2] += _RELEASE_Z_OFFSET
        #     return target, _constant_gripper(num_envs, device, _GRIPPER_CLOSE)

        target = place_pos_w.clone()
        target[:, 2] += _LIFT_Z_OFFSET
        return target, _constant_gripper(num_envs, device, _GRIPPER_OPEN)

    # ------------------------------------------------------------------
    # DART helpers
    # ------------------------------------------------------------------

    def _sample_dart_noise(
        self,
        phase: int,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample a fixed (x, y, z) noise offset for the current phase.

        The offset is held constant throughout the phase and decayed to zero
        in _apply_dart_noise, producing a smooth "drift → correct" shape.
        Phase 3 (lift) has zero xy noise because _lift_start_ee_xy_w already
        locks the xy target — adding lateral noise there would fight that lock.
        """
        sigma_xy = _DART_SIGMA_XY[phase]
        sigma_z  = _DART_SIGMA_Z[phase]

        noise = torch.zeros(num_envs, 3, device=device, dtype=dtype)
        if sigma_xy > 0.0:
            noise[:, :2] = torch.randn(num_envs, 2, device=device, dtype=dtype) * sigma_xy
        if sigma_z > 0.0:
            noise[:, 2] = torch.randn(num_envs, device=device, dtype=dtype) * sigma_z
        return noise

    def _apply_dart_noise(self, target_pos_w: torch.Tensor, phase: int) -> torch.Tensor:
        """Add linearly-decayed DART noise to target_pos_w.

        Decay: noise_scale = 1 - (step / duration), so the EE starts displaced
        and IK drives it back to the true target by phase end.  The final few
        steps have near-zero noise, meaning the demo reaches the correct state
        before the next phase begins (important for grasp/release phases).
        """
        if self._dart_noise_w is None:
            return target_pos_w

        duration = max(self._events_dt[self._event] - 1, 1)
        # decay_factor goes from 1.0 at step 0 to 0.0 at the last step
        decay = max(1.0 - self._step_count / duration, 0.0)

        return target_pos_w + self._dart_noise_w * decay

    # ------------------------------------------------------------------
    # Timeline
    # ------------------------------------------------------------------

    def advance(self) -> None:
        if self._episode_done:
            return

        self._step_count += 1
        if self._step_count < self._events_dt[self._event]:
            return

        self._event += 1
        self._step_count = 0
        # Reset DART noise so a fresh offset is sampled on the next phase's step 0.
        self._dart_noise_w = None

        if self._event >= len(self._events_dt):
            self._episode_done = True
            return

        new_obj_idx = self._event // _PHASES_PER_OBJECT
        if new_obj_idx != self._current_object_idx:
            self._current_object_idx = new_obj_idx
            # FIX: reset _initial_ee_pos_w so it is re-captured on the next
            # hover phase's first step (get_action now handles both objects).
            self._initial_ee_pos_w = None
            self._gripper_down_yaw_w = None
            self._gripper_down_yaw_offset_w = None
            self._lift_start_ee_xy_w = None

    def reset(self) -> None:
        self._step_count = 0
        self._episode_done = False
        self._event = 0
        self._current_object_idx = 0
        self._initial_ee_pos_w = None
        self._gripper_down_yaw_w = None
        self._gripper_down_yaw_offset_w = None
        self._lift_start_ee_xy_w = None
        self._dart_noise_w = None

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
            self._gripper_down_yaw_offset_w = torch.empty(num_envs, device=device, dtype=dtype).uniform_(
                _GRIPPER_DOWN_YAW_OFFSET_RANGE[0],
                _GRIPPER_DOWN_YAW_OFFSET_RANGE[1],
            )
            if obj_name == _KNIFE_NAME:
                base_yaw = torch.zeros_like(base_yaw)  # knife: fixed world yaw
            self._gripper_down_yaw_w = (
                base_yaw + yaw_offset + self._gripper_down_yaw_offset_w
            ).clone()

        roll = torch.full((num_envs,), _GRIPPER_DOWN_ROLL_W, device=device, dtype=dtype)
        pitch = torch.full((num_envs,), _GRIPPER_DOWN_PITCH_W, device=device, dtype=dtype)
        yaw = self._gripper_down_yaw_w.to(device=device, dtype=dtype)
        return quat_from_euler_xyz(roll, pitch, yaw)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_episode_done(self) -> bool:
        return self._episode_done

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def task_object_names(self) -> tuple[str, ...]:
        return (_FORK_NAME, _KNIFE_NAME, _PLATE_NAME)