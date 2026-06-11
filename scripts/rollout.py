# Synchronous LeRobot rollout script for LeIsaac.
# Derived partially from upstream LeIsaac
# `scripts/evaluation/policy_inference.py`
# (https://github.com/LightwheelAI/leisaac/blob/main/scripts/evaluation/policy_inference.py
# @ SHA 6b933e80786a69eb27d47503d11725c9c846566e), trimmed to local LeRobot
# inference and extended with a dual-viewport setup, a debug shape printer,
# and an in-process LeRobotSyncPolicy. Entry point lives at the top of
# `scripts/` (NOT under `scripts/evaluation/`) per AUT-81.

"""Run local LeRobot policy inference in the same process as Isaac Sim."""

"""Launch Isaac Sim Simulator first."""
import json as _json
import multiprocessing
import os as _os
from pathlib import Path as _Path


# Fields that newer LeRobot adds at training time but the inference-side
# LeRobot installed in the worker image doesn't accept. They're all
# training-only (LoRA, torch.compile, image-preproc) and safe to strip
# from the checkpoint's config.json before from_pretrained() reads it.
# Extend whenever draccus.utils.DecodingError surfaces a new field.
_LEROBOT_INCOMPAT_CONFIG_FIELDS: tuple[str, ...] = (
    "use_peft",
    "resize_shape",
    "crop_ratio",
    "compile_model",
    "compile_mode",
)


def _patch_lerobot_config(checkpoint_dir: str) -> None:
    """Strip known-incompatible fields from <checkpoint>/config.json.

    Idempotent — running twice is fine. Errors are swallowed; if config.json
    is missing or unreadable the original from_pretrained call will still
    surface a helpful message.
    """
    cfg_path = _Path(checkpoint_dir) / "config.json"
    if not cfg_path.is_file():
        return
    try:
        with cfg_path.open("r") as f:
            cfg = _json.load(f)
    except (OSError, ValueError) as exc:
        print(f"[rollout] config.json read skipped: {exc}", flush=True)
        return
    stripped = [k for k in _LEROBOT_INCOMPAT_CONFIG_FIELDS if k in cfg]
    if not stripped:
        return
    for k in stripped:
        cfg.pop(k, None)
    try:
        with cfg_path.open("w") as f:
            _json.dump(cfg, f, indent=2)
        print(
            f"[rollout] stripped LeRobot-incompatible config fields: {stripped}",
            flush=True,
        )
    except OSError as exc:
        print(f"[rollout] config.json patch skipped: {exc}", flush=True)




if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Synchronous LeRobot inference for LeIsaac simulation."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--object_poses",
    type=str,
    default=None,
    help="Optional per-episode object_poses.json used to reset evaluation layouts.",
)
parser.add_argument(
    "--dining_cleanup_config",
    type=str,
    default=None,
    help="Optional Dining Cleanup JSON config. Provides asset/scale overrides and a default object_poses path.",
)
parser.add_argument(
    "--step_hz", type=int, default=60, help="Environment stepping rate in Hz."
)
parser.add_argument("--seed", type=int, default=None, help="Seed of the environment.")
parser.add_argument(
    "--episode_length_s", type=float, default=60.0, help="Episode length in seconds."
)
parser.add_argument(
    "--eval_rounds",
    type=int,
    default=0,
    help=(
        "Number of evaluation rounds. 0 means don't add time out termination, policy will run until success or manual"
        " reset."
    ),
)
parser.add_argument(
    "--policy_type",
    type=str,
    default="lerobot-smolvla",
    help="Local LeRobot policy type. Use lerobot-, for example lerobot-smolvla.",
)
parser.add_argument(
    "--policy_action_horizon",
    type=int,
    default=16,
    help="Number of actions to execute per policy call.",
)
parser.add_argument(
    "--policy_language_instruction",
    type=str,
    default=None,
    help="Language instruction of the policy.",
)
parser.add_argument(
    "--policy_checkpoint_path",
    type=str,
    required=True,
    help="Path to the local LeRobot checkpoint.",
)
parser.add_argument(
    "--debug_policy_shapes",
    action="store_true",
    help="Print observation and action tensor shapes around each local LeRobot inference call.",
)
parser.add_argument(
    "--show_wipe_mesh",
    action="store_true",
    help=(
        "Create and update the wipe-coverage visualization mesh in the USD stage. "
        "Mirrors the mesh maintained by the DiningCleanup FSM during data generation."
    ),
)
parser.add_argument(
    "--record_video",
    action="store_true",
    help="Record one mp4 per evaluation episode from policy camera observations.",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default="outputs/rollout/videos",
    help="Directory for per-episode rollout videos when --record_video is set.",
)
parser.add_argument(
    "--video_fps",
    type=int,
    default=10,
    help="Output video FPS. Frames are sampled from environment steps.",
)
parser.add_argument(
    "--video_cameras",
    type=str,
    default="wrist,front",
    help="Comma-separated camera names to concatenate horizontally in videos.",
)
parser.add_argument(
    "--disable_progress",
    action="store_true",
    help="Disable rollout progress bars and ETA output.",
)
parser.add_argument(
    "--progress_interval_s",
    type=float,
    default=5.0,
    help="Minimum seconds between progress bar refreshes.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.dining_cleanup_config:
    config_path = _Path(args_cli.dining_cleanup_config).expanduser()
    _os.environ["DINING_CLEANUP_CONFIG"] = str(config_path)
    with config_path.open("r") as f:
        dining_cleanup_config = _json.load(f)
    if args_cli.object_poses is None and dining_cleanup_config.get("object_poses"):
        args_cli.object_poses = dining_cleanup_config["object_poses"]
    print(f"[rollout] using Dining Cleanup config: {config_path}", flush=True)
    if args_cli.object_poses:
        print(f"[rollout] object_poses: {args_cli.object_poses}", flush=True)

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import time
from typing import Any

import omni.ui as ui
import omni.kit.app
import omni.kit.viewport.utility as vp_util

import carb
import gymnasium as gym
import numpy as np
import omni
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import Camera
from isaaclab_tasks.utils import parse_env_cfg
from lerobot.async_inference.helpers import raw_observation_to_observation
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import ACTION, OBS_IMAGES

from leisaac.utils.env_utils import (
    dynamic_reset_gripper_effort_limit_sim,
    get_task_type,
)
from leisaac.utils.robot_utils import (
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
)

import leisaac  # noqa: F401
import simulator.tasks  # noqa: F401
from simulator.tasks.external import resolve_task
from simulator.utils.object_poses_loader import load_episode_poses
from simulator import FRANKA_JOINT_NAMES

try:
    from tqdm import tqdm as _tqdm
except Exception:  # noqa: BLE001
    _tqdm = None


def setup_dual_viewports():
    """Setup dual viewports: main perspective view and GoPro camera view."""
    perspective_path = "/World/envs/env_0/Robot/panda_hand/wrist"

    # Get main viewport window
    v1_window = ui.Workspace.get_window("Viewport")
    if not v1_window:
        print("Error: Main viewport window not found")
        return

    v1_api = vp_util.get_viewport_from_window_name("Viewport")
    if v1_api:
        v1_api.camera_path = perspective_path

    # Get or create secondary viewport window
    v2_window = ui.Workspace.get_window("Viewport 2")
    if not v2_window:
        v2_window = vp_util.create_viewport_window("Viewport 2")
        # Important: Wait for UI to register the new window
        omni.kit.app.get_app().update()  # Synchronous frame update

    v2_api = vp_util.get_viewport_from_window_name("Viewport 2")
    if v2_api:
        v2_api.camera_path = f"/World/front_camera"

    # Ensure both windows exist before docking
    if v1_window and v2_window:
        # Wait for UI to stabilize before docking
        omni.kit.app.get_app().update()

        # Attempt docking with error handling
        try:
            v2_window.dock_in(v1_window, ui.DockPosition.RIGHT)
            print("Viewports docked: [Viewport (Persp)] | [Viewport 2 (Camera)]")
        except Exception as e:
            print(f"Docking failed: {str(e)}")
            # Alternative docking approach if direct docking fails
            try:
                # Try docking after another frame
                omni.kit.app.get_app().update()
                v2_window.dock_in(v1_window, ui.DockPosition.RIGHT)
                print("Viewports docked on second attempt")
            except Exception as e2:
                print(f"Second docking attempt failed: {str(e2)}")
    else:
        print("Error: Could not find one or both viewport windows for docking.")


class _EpisodeVideoRecorder:
    """Per-episode mp4 recorder for rollout debugging."""

    def __init__(
        self,
        out_dir: str,
        fps: int = 10,
        step_hz: int = 60,
        camera_names: list[str] | None = None,
    ):
        self.out_dir = out_dir
        self.fps = max(1, int(fps))
        self.capture_stride = max(1, int(step_hz) // self.fps)
        self.camera_names = camera_names or ["wrist", "front"]
        _os.makedirs(out_dir, exist_ok=True)
        self.proc = None
        self.tmp_path = None
        self._broken = False
        self._step_counter = 0
        self._frames = 0
        self._idx = None

    def start(self, idx: int) -> None:
        self._idx = idx
        self.tmp_path = _os.path.join(self.out_dir, f"ep_{idx:02d}_tmp.mp4")
        self.proc = None
        self._broken = False
        self._step_counter = 0
        self._frames = 0

    def _spawn(self, width: int, height: int) -> None:
        import subprocess as _sp

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            self.tmp_path,
        ]
        try:
            self.proc = _sp.Popen(cmd, stdin=_sp.PIPE)
        except (OSError, FileNotFoundError) as exc:
            print(f"[video] ffmpeg spawn failed: {exc}", flush=True)
            self._broken = True
            self.proc = None

    def add(self, env) -> None:
        if self._broken or self.proc is False:
            return
        self._step_counter += 1
        if self._step_counter % self.capture_stride != 0:
            return
        try:
            frames = []
            for camera_name in self.camera_names:
                sensor = env.scene[camera_name]
                rgb = sensor.data.output["rgb"]
                frame = rgb.detach().cpu().numpy()[0]
                if frame.shape[-1] > 3:
                    frame = frame[..., :3]
                if frame.dtype != np.uint8:
                    if np.nanmax(frame) <= 1.0:
                        frame = frame * 255.0
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
                frames.append(frame)
            frame = np.concatenate(frames, axis=1)
        except Exception as exc:  # noqa: BLE001
            print(f"[video] frame grab failed: {exc}", flush=True)
            self._broken = True
            return
        if self.proc is None:
            h, w = frame.shape[:2]
            self._spawn(w, h)
            if self._broken or self.proc is None:
                return
        try:
            self.proc.stdin.write(frame.tobytes())
            self._frames += 1
        except (BrokenPipeError, OSError) as exc:
            print(f"[video] pipe broken after {self._frames} frames: {exc}", flush=True)
            self._broken = True
            self.proc = None

    def finalize(self, outcome: str) -> None:
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=20)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        if self.tmp_path and _os.path.exists(self.tmp_path) and self._frames > 0:
            final_path = _os.path.join(self.out_dir, f"ep_{self._idx:02d}_{outcome}.mp4")
            _os.replace(self.tmp_path, final_path)
            print(f"[video] saved {final_path} ({self._frames} frames)", flush=True)
        elif self.tmp_path and _os.path.exists(self.tmp_path):
            _os.remove(self.tmp_path)
        self.proc = None
        self.tmp_path = None


class _NoOpProgress:
    """Fallback progress bar used when tqdm is unavailable or disabled."""

    def update(self, n: int = 1) -> None:
        pass

    def set_postfix(self, *args, **kwargs) -> None:
        pass

    def close(self) -> None:
        pass


def _make_progress(
    *,
    total: int | None,
    desc: str,
    unit: str,
    leave: bool,
    disable: bool,
    mininterval: float,
):
    if disable or _tqdm is None:
        return _NoOpProgress()
    return _tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        mininterval=max(0.1, float(mininterval)),
    )


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


class Controller:
    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )
        self.reset_state = False

    def __del__(self):
        if (
            hasattr(self, "_input")
            and hasattr(self, "_keyboard")
            and hasattr(self, "_keyboard_sub")
        ):
            self._input.unsubscribe_from_keyboard_events(
                self._keyboard, self._keyboard_sub
            )
            self._keyboard_sub = None

    def reset(self):
        self.reset_state = False

    def _on_keyboard_event(self, event, *args, **kwargs):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "R":
                self.reset_state = True
        return True


def _shape_summary(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    return type(value).__name__


def _print_mapping_shapes(title: str, values: dict[str, Any]) -> None:
    print(title)
    for key in sorted(values):
        print(f"  {key}: {_shape_summary(values[key])}")


class LeRobotSyncPolicy:
    """Local LeRobot inference path matching the async server pipeline."""

    def __init__(
        self,
        policy_type: str,
        pretrained_name_or_path: str,
        task_type: str,
        camera_infos: dict[str, tuple[int, int]],
        actions_per_chunk: int,
        device: str,
        debug_policy_shapes: bool = False,
    ):
        if actions_per_chunk <= 0:
            raise ValueError(
                f"policy_action_horizon must be positive, got {actions_per_chunk}."
            )

        self.task_type = task_type
        self.actions_per_chunk = actions_per_chunk
        self.device = device
        self.debug_policy_shapes = debug_policy_shapes

        if task_type == "so101leader":
            self.state_joint_names = SINGLE_ARM_JOINT_NAMES
            self.action_dim = len(SINGLE_ARM_JOINT_NAMES)
        elif task_type == "franka_panda":
            self.state_joint_names = FRANKA_JOINT_NAMES
            self.action_dim = 8
        else:
            raise ValueError(
                f"Task type {task_type} not supported for synchronous LeRobot inference yet."
            )

        self.env_camera_keys = list(camera_infos.keys())

        print(
            f"Loading local LeRobot policy '{policy_type}' from {pretrained_name_or_path}...",
            flush=True,
        )
        # Strip training-only fields that newer LeRobot adds but the
        # inference-side LeRobot doesn't accept. Safe because these flags
        # never affect inference. See _patch_lerobot_config above.
        _patch_lerobot_config(pretrained_name_or_path)
        policy_class = get_policy_class(policy_type)
        self.policy = policy_class.from_pretrained(pretrained_name_or_path, local_files_only=True)
        self.policy.to(device)
        self.policy.eval()
        self.camera_feature_map = self._map_camera_features(
            camera_infos, self.policy.config.image_features
        )
        self.lerobot_features = self._build_lerobot_features(
            camera_infos, self.camera_feature_map
        )

        device_override = {"device": device}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=pretrained_name_or_path,
            preprocessor_overrides={
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": {}},
            },
            postprocessor_overrides={"device_processor": device_override},
        )
        print("Local LeRobot policy is ready.", flush=True)

    def reset(self):
        policy_reset = getattr(self.policy, "reset", None)
        if callable(policy_reset):
            with torch.inference_mode():
                policy_reset()

    def _build_lerobot_features(
        self,
        camera_infos: dict[str, tuple[int, int]],
        camera_feature_map: dict[str, str],
    ) -> dict[str, dict]:
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (len(self.state_joint_names),),
                "names": [f"{joint_name}.pos" for joint_name in self.state_joint_names],
            }
        }
        for feature_key, camera_key in camera_feature_map.items():
            camera_image_shape = camera_infos[camera_key]
            features[feature_key] = {
                "dtype": "image",
                "shape": (camera_image_shape[0], camera_image_shape[1], 3),
                "names": ["height", "width", "channels"],
            }
        return features

    def _map_camera_features(
        self,
        camera_infos: dict[str, tuple[int, int]],
        policy_image_features: dict[str, Any],
    ) -> dict[str, str]:
        """Map checkpoint image feature names to available env camera names."""
        policy_feature_keys = list(policy_image_features.keys())
        if not policy_feature_keys:
            policy_feature_keys = [
                f"observation.images.{camera_key}" for camera_key in camera_infos
            ]

        camera_keys = list(camera_infos.keys())
        if not camera_keys:
            raise ValueError("No camera observations are available for policy inference.")

        camera_feature_map: dict[str, str] = {}
        for index, feature_key in enumerate(policy_feature_keys):
            suffix = feature_key.split("observation.images.", 1)[-1]
            matched_camera = None
            for camera_key in camera_keys:
                if (
                    suffix == camera_key
                    or suffix.endswith(f".{camera_key}")
                    or suffix.endswith(f"_{camera_key}")
                    or camera_key in suffix
                ):
                    matched_camera = camera_key
                    break
            if matched_camera is None:
                matched_camera = camera_keys[min(index, len(camera_keys) - 1)]
            camera_feature_map[feature_key] = matched_camera

        print(
            "[rollout] policy image features: "
            + ", ".join(policy_feature_keys),
            flush=True,
        )
        print(
            "[rollout] camera feature map: "
            + ", ".join(
                f"{feature_key} <- {camera_key}"
                for feature_key, camera_key in camera_feature_map.items()
            ),
            flush=True,
        )
        return camera_feature_map

    def _build_raw_observation(self, observation_dict: dict) -> dict[str, Any]:
        raw_observation = {}
        for feature_key, camera_key in self.camera_feature_map.items():
            frame = observation_dict[camera_key].cpu().numpy().astype(np.uint8)[0]
            raw_key = feature_key.split("observation.images.", 1)[-1]
            raw_observation[raw_key] = frame
            raw_observation[feature_key] = frame
            raw_observation[camera_key] = frame
        raw_observation["task"] = observation_dict["task_description"]

        if self.task_type == "so101leader":
            joint_pos = convert_leisaac_action_to_lerobot(observation_dict["joint_pos"])
        elif self.task_type == "franka_panda":
            joint_pos = observation_dict["joint_pos"].cpu().numpy()
        else:
            raise ValueError(
                f"Task type {self.task_type} not supported for synchronous LeRobot inference yet."
            )

        for joint_index, joint_name in enumerate(self.state_joint_names):
            raw_observation[f"{joint_name}.pos"] = joint_pos[0, joint_index].item()

        return raw_observation

    def _config_horizon_summary(self) -> str:
        names = ["chunk_size", "n_action_steps", "action_chunk_size", "action_horizon"]
        values = []
        for name in names:
            if hasattr(self.policy.config, name):
                values.append(f"{name}={getattr(self.policy.config, name)}")
        return ", ".join(values) if values else "no known horizon fields found"

    def _prepare_observation(self, raw_observation: dict[str, Any]) -> dict[str, Any]:
        observation = raw_observation_to_observation(
            raw_observation,
            self.lerobot_features,
            self.policy.config.image_features,
        )
        if self.debug_policy_shapes:
            _print_mapping_shapes("[SyncPolicy] Prepared observation:", observation)

        observation = self.preprocessor(observation)
        if self.debug_policy_shapes:
            _print_mapping_shapes("[SyncPolicy] Preprocessed observation:", observation)
        return observation

    def _predict_lerobot_actions(self, observation: dict[str, Any]) -> torch.Tensor:
        with torch.inference_mode():
            action = self.policy.select_action(observation)
        if isinstance(action, torch.Tensor):
            action = action.clone()
        return self.postprocessor(action)

    def _convert_actions_to_leisaac(self, action_tensor: torch.Tensor) -> np.ndarray:
        if self.task_type == "so101leader":
            actions = convert_lerobot_action_to_leisaac(action_tensor)
        elif self.task_type == "franka_panda":
            actions = action_tensor.to("cpu").numpy()
        else:
            raise ValueError(
                f"Task type {self.task_type} not supported for synchronous LeRobot inference yet."
            )

        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected {self.action_dim} action values for task type {self.task_type}, got {actions.shape[-1]}."
            )
        return actions

    def get_action(self, observation_dict: dict) -> torch.Tensor:
        raw_observation = self._build_raw_observation(observation_dict)
        if self.debug_policy_shapes:
            _print_mapping_shapes("[SyncPolicy] Raw observation:", raw_observation)

        observation = self._prepare_observation(raw_observation)
        action_tensor = self._predict_lerobot_actions(observation)
        actions = self._convert_actions_to_leisaac(action_tensor)
        return torch.from_numpy(actions[:, None, :])


def preprocess_obs_dict(obs_dict: dict, language_instruction: str):
    obs_dict["task_description"] = language_instruction
    return obs_dict


def get_policy_type(policy_type_arg: str) -> str:
    if not policy_type_arg.startswith("lerobot-"):
        raise ValueError(
            f"policy_inference_sync.py only supports local LeRobot policies, got '{policy_type_arg}'. "
            "Use --policy_type=lerobot-."
        )
    return policy_type_arg.split("lerobot-", 1)[1]


def get_camera_infos(
    env: ManagerBasedRLEnv, policy_obs_dict: dict
) -> dict[str, tuple[int, int]]:
    camera_infos = {}
    for key, sensor in env.scene.sensors.items():
        if isinstance(sensor, Camera) and key in policy_obs_dict:
            camera_infos[key] = sensor.image_shape
    return camera_infos


def _apply_episode_poses(env: ManagerBasedRLEnv, poses: dict) -> None:
    device = env.device
    for name, (pos, quat) in poses.items():
        obj = env.scene[name]
        pose_tensor = torch.tensor(
            [[pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]],
            device=device,
            dtype=torch.float32,
        ).repeat(env.num_envs, 1)
        obj.write_root_pose_to_sim(pose_tensor)
        print(
            f"  [eval pose] {name}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})",
            flush=True,
        )


def _sync_sim_after_pose_write(env: ManagerBasedRLEnv) -> None:
    write_data = getattr(env.scene, "write_data_to_sim", None)
    if callable(write_data):
        write_data()

    forward = getattr(env.sim, "forward", None)
    if callable(forward):
        forward()
    else:
        env.sim.step(render=False)

    update = getattr(env.scene, "update", None)
    if callable(update):
        update(dt=0.0)


def _current_observations(env: ManagerBasedRLEnv, fallback_obs: dict) -> dict:
    getter = getattr(env, "get_observations", None)
    if callable(getter):
        obs = getter()
        return obs[0] if isinstance(obs, tuple) else obs

    obs_manager = getattr(env, "observation_manager", None)
    if obs_manager is not None and hasattr(obs_manager, "compute"):
        obs = obs_manager.compute()
        if isinstance(obs, dict) and "policy" in obs:
            return obs
        return {"policy": obs}

    return fallback_obs


def _reset_eval_episode(env: ManagerBasedRLEnv, episodes: list[dict], episode_index: int) -> dict:
    print("[rollout] resetting environment...", flush=True)
    obs_dict, _ = env.reset()
    print("[rollout] env.reset() returned", flush=True)
    if hasattr(env, "_dining_cleanup_last_status"):
        setattr(env, "_dining_cleanup_last_status", None)

    if episodes:
        pose_index = (episode_index - 1) % len(episodes)
        print(f"[rollout] applying object pose episode {pose_index + 1}/{len(episodes)}", flush=True)
        _apply_episode_poses(env, episodes[pose_index])
        _sync_sim_after_pose_write(env)
        obs_dict = _current_observations(env, obs_dict)
    return obs_dict


def _print_task_episode_status(env: ManagerBasedRLEnv, prefix: str) -> None:
    try:
        from simulator.tasks.dining_cleanup.dining_cleanup_env_cfg import print_dining_cleanup_status

        print_dining_cleanup_status(env, prefix=prefix)
    except KeyError:
        return
    except Exception as exc:
        print(f"{prefix} status report failed: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Wipe-coverage visualization mesh helpers
# (mirrors DiningCleanupStateMachine.setup() / _sync_wipe_vis())
# ---------------------------------------------------------------------------

def _init_wipe_vis_mesh(env) -> None:
    """Create the wipe-coverage visualisation mesh for every env in the stage.

    Reads geometry constants from dining_cleanup_env_cfg so the mesh matches
    exactly what the FSM creates during data generation.
    """
    try:
        from simulator.tasks.dining_cleanup.dining_cleanup_env_cfg import (
            LEFT_TABLE_X_RANGE,
            LEFT_TABLE_Y_RANGE,
            WIPE_COVERAGE_RESOLUTION,
            _create_wipe_vis_mesh,
        )
    except Exception as exc:
        print(f"[rollout][wipe_mesh] could not import mesh helpers: {exc}", flush=True)
        return

    import math

    stage = env.sim.stage
    x_bins = max(1, math.ceil((LEFT_TABLE_X_RANGE[1] - LEFT_TABLE_X_RANGE[0]) / WIPE_COVERAGE_RESOLUTION))
    y_bins = max(1, math.ceil((LEFT_TABLE_Y_RANGE[1] - LEFT_TABLE_Y_RANGE[0]) / WIPE_COVERAGE_RESOLUTION))

    # Cache bin counts on the env for the per-step sync to reuse.
    env._wipe_vis_x_bins = x_bins
    env._wipe_vis_y_bins = y_bins

    for idx in range(env.num_envs):
        origin = env.scene.env_origins[idx]
        ox, oy = float(origin[0]), float(origin[1])
        mesh_path = f"/World/envs/env_{idx}/Scene/wipe_vis_plane"
        if not stage.GetPrimAtPath(mesh_path).IsValid():
            _create_wipe_vis_mesh(
                stage,
                mesh_path,
                x_range=(ox + LEFT_TABLE_X_RANGE[0], ox + LEFT_TABLE_X_RANGE[1]),
                y_range=(oy + LEFT_TABLE_Y_RANGE[0], oy + LEFT_TABLE_Y_RANGE[1]),
                x_bins=x_bins,
                y_bins=y_bins,
                z=0.045,
            )

    print(
        f"[rollout][wipe_mesh] initialized mesh for {env.num_envs} env(s) "
        f"({x_bins}×{y_bins} bins)",
        flush=True,
    )


def _sync_wipe_vis_mesh(env) -> None:
    """Update wipe-coverage vertex colours for env 0 from the env's coverage state.

    Reads ``env._dining_cleanup_wipe_covered`` (a bool tensor written by the
    termination function each step) and maps it to an RGB heatmap identical to
    DiningCleanupStateMachine._sync_wipe_vis().
    """
    from pxr import Gf, UsdGeom, Vt

    state = getattr(env, "_dining_cleanup_wipe_covered", None)
    if state is None:
        return

    stage = env.sim.stage
    mesh_path = "/World/envs/env_0/Scene/wipe_vis_plane"
    prim = stage.GetPrimAtPath(mesh_path)
    if not prim.IsValid():
        return

    c = state[0].float().cpu()          # (x_bins, y_bins)
    colors = []
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            v = float(c[i, j])
            if c[i, j] <= 0.0:
                colors.append(Gf.Vec3f(0.36, 0.25, 0.20))
            else:
                colors.append(Gf.Vec3f(1.0, 1.0, 1.0))  
                
    UsdGeom.Mesh(prim).GetDisplayColorAttr().Set(Vt.Vec3fArray(colors))


def main():
    task_id = resolve_task(args_cli.task)
    args_cli.task = task_id
    env_cfg = parse_env_cfg(task_id, device=args_cli.device, num_envs=1)
    task_type = get_task_type(task_id)
    robot_name = getattr(env_cfg, "robot_name", None)
    policy_task_type = "franka_panda" if robot_name == "franka_panda" else task_type
    teleop_device = "keyboard" if policy_task_type == "franka_panda" else task_type
    env_cfg.use_teleop_device(teleop_device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.episode_length_s = args_cli.episode_length_s

    episodes = []
    if args_cli.object_poses:
        object_pose_cfg = getattr(env_cfg, "object_pose_cfg", None)
        if object_pose_cfg is None:
            raise ValueError(
                f"Task '{task_id}' env_cfg has no 'object_pose_cfg' attribute; "
                "cannot resolve --object_poses."
            )
        episodes = load_episode_poses(args_cli.object_poses, object_pose_cfg)
        if not episodes:
            raise ValueError(f"No status=='full' episodes found in {args_cli.object_poses}.")
        print(f"[rollout] loaded {len(episodes)} object pose episodes from {args_cli.object_poses}", flush=True)

    if args_cli.eval_rounds <= 0:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
    max_episode_count = args_cli.eval_rounds
    env_cfg.recorders = None

    env: ManagerBasedRLEnv = gym.make(task_id, cfg=env_cfg).unwrapped

    # Warm up the renderer before the first reset. Headless Isaac Sim with
    # camera observations otherwise hangs the first env.reset() while the
    # Vulkan / DLSS / shader pipeline compiles — the worker sees no output
    # for several minutes and the eval looks dead. A handful of app updates
    # forces shader compilation and material warm-up to happen here, where
    # we can attribute it.
    print("[rollout] warming up renderer (20 app updates)...", flush=True)
    for _ in range(20):
        simulation_app.update()
    obs_dict = _reset_eval_episode(env, episodes, episode_index=1)

    if args_cli.show_wipe_mesh:
        _init_wipe_vis_mesh(env)

    language_instruction = args_cli.policy_language_instruction
    if language_instruction is None:
        language_instruction = getattr(env_cfg, "task_description", None)

    policy_obs_dict = preprocess_obs_dict(obs_dict["policy"], language_instruction)
    camera_infos = get_camera_infos(env, policy_obs_dict)
    print(
        f"[rollout] camera_infos = {camera_infos}; loading policy...",
        flush=True,
    )

    policy = LeRobotSyncPolicy(
        policy_type=get_policy_type(args_cli.policy_type),
        pretrained_name_or_path=args_cli.policy_checkpoint_path,
        task_type=policy_task_type,
        camera_infos=camera_infos,
        actions_per_chunk=args_cli.policy_action_horizon,
        device=args_cli.device,
        debug_policy_shapes=args_cli.debug_policy_shapes,
    )

    rate_limiter = RateLimiter(args_cli.step_hz)
    controller = Controller()
    controller.reset()

    if args_cli.headless:
        print("[rollout] headless mode: skipping GUI viewport setup", flush=True)
    else:
        setup_dual_viewports()

    video_recorder = None
    if args_cli.record_video:
        video_cameras = [
            name.strip() for name in args_cli.video_cameras.split(",") if name.strip()
        ]
        if not video_cameras:
            raise ValueError("--video_cameras must include at least one camera name.")
        video_recorder = _EpisodeVideoRecorder(
            args_cli.video_dir,
            fps=args_cli.video_fps,
            step_hz=args_cli.step_hz,
            camera_names=video_cameras,
        )
        print(
            f"[video] recording rollout videos to {args_cli.video_dir} "
            f"from cameras={video_cameras} at {video_recorder.fps} fps",
            flush=True,
        )

    episode_step_total = max(1, int(round(args_cli.episode_length_s * args_cli.step_hz)))
    if max_episode_count > 0:
        print(
            f"[progress] total episodes={max_episode_count}, "
            f"episode budget={episode_step_total} steps "
            f"({args_cli.episode_length_s:g}s at {args_cli.step_hz}Hz)",
            flush=True,
        )
    else:
        print(
            f"[progress] open-ended eval, episode budget={episode_step_total} steps "
            f"({args_cli.episode_length_s:g}s at {args_cli.step_hz}Hz)",
            flush=True,
        )

    total_progress = _make_progress(
        total=max_episode_count if max_episode_count > 0 else None,
        desc="Evaluation",
        unit="episode",
        leave=True,
        disable=args_cli.disable_progress,
        mininterval=args_cli.progress_interval_s,
    )

    success_count, episode_count = 0, 1
    try:
        while max_episode_count <= 0 or episode_count <= max_episode_count:
            print(f"[Evaluation] Evaluating episode {episode_count}...", flush=True)
            if video_recorder is not None:
                video_recorder.start(episode_count)
            episode_progress = _make_progress(
                total=episode_step_total,
                desc=f"Episode {episode_count}",
                unit="step",
                leave=False,
                disable=args_cli.disable_progress,
                mininterval=args_cli.progress_interval_s,
            )
            success, time_out = False, False
            episode_steps = 0
            try:
                while simulation_app.is_running():
                    with torch.no_grad():
                        if controller.reset_state:
                            controller.reset()
                            policy.reset()
                            if video_recorder is not None:
                                video_recorder.finalize("reset")
                            total_progress.update(1)
                            episode_count += 1
                            if max_episode_count <= 0 or episode_count <= max_episode_count:
                                obs_dict = _reset_eval_episode(env, episodes, episode_count)
                            break

                        policy_obs_dict = preprocess_obs_dict(
                            obs_dict["policy"], language_instruction
                        )
                        actions = policy.get_action(policy_obs_dict).to(env.device)
                        for action_index in range(
                            min(args_cli.policy_action_horizon, actions.shape[0])
                        ):
                            action = actions[action_index, :, :]
                            if env.cfg.dynamic_reset_gripper_effort_limit:
                                dynamic_reset_gripper_effort_limit_sim(env, teleop_device)
                            obs_dict, _, reset_terminated, reset_time_outs, _ = env.step(action)
                            episode_steps += 1
                            episode_progress.update(1)
                            if args_cli.show_wipe_mesh:
                                _sync_wipe_vis_mesh(env)
                            if video_recorder is not None:
                                video_recorder.add(env)
                            if reset_terminated[0]:
                                success = True
                                break
                            if reset_time_outs[0]:
                                time_out = True
                                break
                            if rate_limiter:
                                rate_limiter.sleep(env)
                    if success:
                        print(f"[Evaluation] Episode {episode_count} is successful!", flush=True)
                        _print_task_episode_status(env, f"[Evaluation] Episode {episode_count}")
                        if video_recorder is not None:
                            video_recorder.finalize("success")
                        episode_count += 1
                        success_count += 1
                        total_progress.update(1)
                        total_progress.set_postfix(
                            success=f"{success_count}/{episode_count - 1}",
                            rate=f"{success_count / (episode_count - 1):.3f}",
                        )
                        policy.reset()
                        if max_episode_count <= 0 or episode_count <= max_episode_count:
                            obs_dict = _reset_eval_episode(env, episodes, episode_count)
                        break
                    if time_out:
                        print(f"[Evaluation] Episode {episode_count} timed out!", flush=True)
                        _print_task_episode_status(env, f"[Evaluation] Episode {episode_count}")
                        if video_recorder is not None:
                            video_recorder.finalize("failed")
                        episode_count += 1
                        total_progress.update(1)
                        total_progress.set_postfix(
                            success=f"{success_count}/{episode_count - 1}",
                            rate=f"{success_count / (episode_count - 1):.3f}",
                        )
                        policy.reset()
                        if max_episode_count <= 0 or episode_count <= max_episode_count:
                            obs_dict = _reset_eval_episode(env, episodes, episode_count)
                        break
            finally:
                if episode_steps:
                    episode_progress.set_postfix(steps=episode_steps)
                episode_progress.close()
            evaluated = max(episode_count - 1, 0)
            rate = success_count / evaluated if evaluated > 0 else 0.0
            print(
                f"[Evaluation] now success rate: {rate:.3f} "
                f" [{success_count}/{evaluated}]",
                flush=True,
            )
    finally:
        total_progress.close()

    if max_episode_count > 0:
        print(
            f"[Evaluation] Final success rate: {success_count / max_episode_count:.3f} "
            f" [{success_count}/{max_episode_count}]"
        )
    else:
        evaluated = max(episode_count - 1, 0)
        rate = success_count / evaluated if evaluated > 0 else 0.0
        print(f"[Evaluation] Final success rate: {rate:.3f} [{success_count}/{evaluated}]")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
