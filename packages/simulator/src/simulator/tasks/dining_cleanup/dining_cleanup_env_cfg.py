import math

import isaaclab.sim as sim_utils
import torch

from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.schemas import MassPropertiesCfg
from isaaclab.utils import configclass

from leisaac.utils.general_assets import parse_usd_and_create_subassets
from simulator import ASSETS_ROOT
from simulator.assets.scenes.dining_room import DINING_ROOM_CFG, DINING_ROOM_USD_PATH
from simulator.utils.object_poses_loader import ObjectPoseConfig

from simulator.tasks.template.single_arm_franka_cfg import (
    SingleArmFrankaObservationsCfg,
    SingleArmFrankaTaskEnvCfg,
    SingleArmFrankaTaskSceneCfg,
    SingleArmFrankaTerminationsCfg,
)

DINING_OBJECTS_ROOT = ASSETS_ROOT / "scenes" / "dining_room" / "objects"
BOWL_USD_PATH = DINING_OBJECTS_ROOT / "bowl" / "model_BalandaBowl_69323.usd"
SPOON_USD_PATH = DINING_OBJECTS_ROOT / "spoon" / "model_Kitchen_Spoon_B008H2JLP8_LargeWooden_69323.usd"
TRAY_USD_PATH = DINING_OBJECTS_ROOT / "tray" / "model_WhiteUtensilTray_69323.usd"
TISSUE_USD_PATH = DINING_OBJECTS_ROOT / "tissue" / "model_tissue_001_69323.usd"
VASE_USD_PATH = DINING_OBJECTS_ROOT / "vase" / "model_BlackVaseSmall_1_69323.usd"
CLOTH_USD_PATH = DINING_OBJECTS_ROOT / "cloth" / "model_tablecloth.usd"

BOWL_SCALE: tuple[float, float, float] = (0.57, 0.57, 0.57)
SPOON_SCALE: tuple[float, float, float] = (0.62, 0.62, 0.62)
TRAY_SCALE: tuple[float, float, float] = (0.79, 1.77, 1.0)
TISSUE_SCALE: tuple[float, float, float] = (1.0, 1.0, 1.0)
VASE_SCALE: tuple[float, float, float] = (1.0, 1.0, 1.0)
CLOTH_SCALE: tuple[float, float, float] = (0.10, 0.10, 1.0)
CLOTH_FOOTPRINT_SIZE: tuple[float, float] = (0.164, 0.098)

# UMI/object_poses entries.  Only bowl/spoon are randomized per replay episode;
# tray, tissue, vase, and cloth are part of the task scene with fixed initial
# placement so the cleanup target and obstacle layout stay consistent.
TAG_TO_OBJECT: dict[int, str] = {1: "bowl", 2: "spoon"}
ANCHOR_TAG_ID: int = 0
ANCHOR_WORLD_POSE: tuple[float, float, float] = (0.40, 0.10, 0.0)
OBJECT_Z: float = 0.05
OBJECT_ROLL: float = 0.0
OBJECT_PITCH: float = 0.0
PER_OBJECT_YAW_OFFSET: dict[str, float] = {
    "bowl": 0.0,
    "spoon": math.pi / 2.0,
}

# Dining-room table footprint in task/world XY is approximately
# x=[0.0, 0.70], y=[-0.65, 0.0].  For this advanced task we use the convention
# requested in the project proposal: +x is the Franka-view right side, -x is
# the Franka-view left side.
TABLE_X_RANGE: tuple[float, float] = (0.0, 0.70)
TABLE_Y_RANGE: tuple[float, float] = (-0.65, 0.0)
TABLE_SURFACE_Z: float = 0.05
LEFT_TABLE_X_RANGE: tuple[float, float] = (0.04, 0.22)
RIGHT_TABLE_X_RANGE: tuple[float, float] = (0.38, 0.66)
LEFT_TABLE_Y_RANGE: tuple[float, float] = (-0.50, -0.15)
WIPE_FINAL_XY: tuple[float, float] = (0.19, LEFT_TABLE_Y_RANGE[1])
WIPE_COVERAGE_THRESHOLD: float = 0.90
WIPE_COVERAGE_RESOLUTION: float = 0.01
WIPE_CONTACT_Z_RANGE: tuple[float, float] = (0.03, 0.13)
STATIC_OBJECT_XY_TOL: float = 0.035

TRAY_WORLD_POS: tuple[float, float, float] = (0.57, -0.36, TABLE_SURFACE_Z)
TISSUE_WORLD_POS: tuple[float, float, float] = (0.35, -0.18, 0.074)
VASE_WORLD_POS: tuple[float, float, float] = (0.35, -0.32, TABLE_SURFACE_Z)
CLOTH_WORLD_POS: tuple[float, float, float] = (0.35, -0.49, TABLE_SURFACE_Z)


@configclass
class DiningCleanupSceneCfg(SingleArmFrankaTaskSceneCfg):
    """Scene configuration for the dining cleanup task."""

    scene: AssetBaseCfg = DINING_ROOM_CFG.replace(prim_path="{ENV_REGEX_NS}/Scene")

    bowl: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/bowl",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.18, -0.30, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(BOWL_USD_PATH),
            scale=BOWL_SCALE,
            mass_props=MassPropertiesCfg(mass=0.10),
        ),
    )

    spoon: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/spoon",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.22, -0.42, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SPOON_USD_PATH),
            scale=SPOON_SCALE,
            mass_props=MassPropertiesCfg(mass=0.05),
        ),
    )

    tray: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/tray",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=TRAY_WORLD_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TRAY_USD_PATH),
            scale=TRAY_SCALE,
            mass_props=MassPropertiesCfg(mass=0.20),
        ),
    )

    tissue: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/tissue",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=TISSUE_WORLD_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TISSUE_USD_PATH),
            scale=TISSUE_SCALE,
            mass_props=MassPropertiesCfg(mass=0.05),
        ),
    )

    vase: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/vase",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=VASE_WORLD_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(VASE_USD_PATH),
            scale=VASE_SCALE,
            mass_props=MassPropertiesCfg(mass=0.20),
        ),
    )

    cloth: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Scene/cloth",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CLOTH_WORLD_POS,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(CLOTH_USD_PATH),
            scale=CLOTH_SCALE,
            mass_props=MassPropertiesCfg(mass=0.03),
        ),
    )


def dining_cleanup_success(
    env,
    bowl_cfg: SceneEntityCfg,
    spoon_cfg: SceneEntityCfg,
    tray_cfg: SceneEntityCfg,
    cloth_cfg: SceneEntityCfg,
    tissue_cfg: SceneEntityCfg,
    vase_cfg: SceneEntityCfg,
    tray_x_half_width: float,
    tray_y_half_width: float,
    z_range: tuple[float, float],
    cloth_final_xy: tuple[float, float],
    cloth_final_tol: float,
    wipe_x_range: tuple[float, float],
    wipe_y_range: tuple[float, float],
    cloth_xy_size: tuple[float, float],
    wipe_coverage_threshold: float,
    wipe_coverage_resolution: float,
    wipe_contact_z_range: tuple[float, float],
    tissue_initial_xy: tuple[float, float],
    vase_initial_xy: tuple[float, float],
    static_object_xy_tol: float,
) -> torch.Tensor:
    """Termination proxy for the cleanup task.

    Bowl/spoon must be arranged in the tray.  The wipe criterion accumulates
    cloth/table coverage over the episode on a coarse XY grid and requires a
    minimum target-area coverage ratio.
    """
    bowl: RigidObject = env.scene[bowl_cfg.name]
    spoon: RigidObject = env.scene[spoon_cfg.name]
    tray: RigidObject = env.scene[tray_cfg.name]
    cloth: RigidObject = env.scene[cloth_cfg.name]
    tissue: RigidObject = env.scene[tissue_cfg.name]
    vase: RigidObject = env.scene[vase_cfg.name]

    bowl_pos = bowl.data.root_pos_w - env.scene.env_origins
    spoon_pos = spoon.data.root_pos_w - env.scene.env_origins
    tray_pos = tray.data.root_pos_w - env.scene.env_origins
    cloth_pos = cloth.data.root_pos_w - env.scene.env_origins
    tissue_pos = tissue.data.root_pos_w - env.scene.env_origins
    vase_pos = vase.data.root_pos_w - env.scene.env_origins

    done = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    for obj_pos in (bowl_pos, spoon_pos):
        done = torch.logical_and(done, torch.abs(obj_pos[:, 0] - tray_pos[:, 0]) <= tray_x_half_width)
        done = torch.logical_and(done, torch.abs(obj_pos[:, 1] - tray_pos[:, 1]) <= tray_y_half_width)
        done = torch.logical_and(done, obj_pos[:, 2] >= tray_pos[:, 2] + z_range[0])
        done = torch.logical_and(done, obj_pos[:, 2] <= tray_pos[:, 2] + z_range[1])

    done = torch.logical_and(done, bowl_pos[:, 1] > tray_pos[:, 1])
    done = torch.logical_and(done, spoon_pos[:, 1] < tray_pos[:, 1])

    final_xy = torch.tensor(cloth_final_xy, dtype=cloth_pos.dtype, device=cloth_pos.device)
    cloth_dist = torch.norm(cloth_pos[:, :2] - final_xy, dim=1)
    done = torch.logical_and(done, cloth_dist <= cloth_final_tol)

    coverage_ratio = _update_wipe_coverage_ratio(
        env,
        cloth_pos,
        wipe_x_range=wipe_x_range,
        wipe_y_range=wipe_y_range,
        cloth_xy_size=cloth_xy_size,
        resolution=wipe_coverage_resolution,
        contact_z_range=wipe_contact_z_range,
    )
    done = torch.logical_and(done, coverage_ratio >= wipe_coverage_threshold)

    tissue_initial = torch.tensor(tissue_initial_xy, dtype=tissue_pos.dtype, device=tissue_pos.device)
    vase_initial = torch.tensor(vase_initial_xy, dtype=vase_pos.dtype, device=vase_pos.device)
    tissue_dist = torch.norm(tissue_pos[:, :2] - tissue_initial, dim=1)
    vase_dist = torch.norm(vase_pos[:, :2] - vase_initial, dim=1)
    done = torch.logical_and(done, tissue_dist <= static_object_xy_tol)
    done = torch.logical_and(done, vase_dist <= static_object_xy_tol)
    return done


def _update_wipe_coverage_ratio(
    env,
    cloth_pos: torch.Tensor,
    *,
    wipe_x_range: tuple[float, float],
    wipe_y_range: tuple[float, float],
    cloth_xy_size: tuple[float, float],
    resolution: float,
    contact_z_range: tuple[float, float],
) -> torch.Tensor:
    """Accumulate cloth/table contact coverage on a fixed XY grid."""
    x_min, x_max = wipe_x_range
    y_min, y_max = wipe_y_range
    x_bins = max(1, math.ceil((x_max - x_min) / resolution))
    y_bins = max(1, math.ceil((y_max - y_min) / resolution))
    state = getattr(env, "_dining_cleanup_wipe_covered", None)
    expected_shape = (env.num_envs, x_bins, y_bins)
    if state is None or state.shape != expected_shape or state.device != cloth_pos.device:
        state = torch.zeros(expected_shape, dtype=torch.bool, device=cloth_pos.device)
        setattr(env, "_dining_cleanup_wipe_covered", state)

    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is not None:
        reset_mask = episode_length_buf.to(device=cloth_pos.device) <= 1
        if reset_mask.any():
            state[reset_mask] = False

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
    half_x = 0.5 * cloth_xy_size[0]
    half_y = 0.5 * cloth_xy_size[1]
    in_contact = torch.logical_and(
        cloth_pos[:, 2] >= contact_z_range[0],
        cloth_pos[:, 2] <= contact_z_range[1],
    )
    covered_now = torch.logical_and(
        torch.abs(cloth_pos[:, 0, None, None] - grid_x[None, :, None]) <= half_x,
        torch.abs(cloth_pos[:, 1, None, None] - grid_y[None, None, :]) <= half_y,
    )
    state |= torch.logical_and(covered_now, in_contact[:, None, None])
    return state.float().mean(dim=(1, 2))


@configclass
class TerminationsCfg(SingleArmFrankaTerminationsCfg):
    """Termination configuration for the dining cleanup task."""

    success = DoneTerm(
        func=dining_cleanup_success,
        params={
            "bowl_cfg": SceneEntityCfg("bowl"),
            "spoon_cfg": SceneEntityCfg("spoon"),
            "tray_cfg": SceneEntityCfg("tray"),
            "cloth_cfg": SceneEntityCfg("cloth"),
            "tissue_cfg": SceneEntityCfg("tissue"),
            "vase_cfg": SceneEntityCfg("vase"),
            "tray_x_half_width": 0.12,
            "tray_y_half_width": 0.13,
            "z_range": (-0.05, 0.10),
            "cloth_final_xy": WIPE_FINAL_XY,
            "cloth_final_tol": 0.12,
            "wipe_x_range": LEFT_TABLE_X_RANGE,
            "wipe_y_range": LEFT_TABLE_Y_RANGE,
            "cloth_xy_size": CLOTH_FOOTPRINT_SIZE,
            "wipe_coverage_threshold": WIPE_COVERAGE_THRESHOLD,
            "wipe_coverage_resolution": WIPE_COVERAGE_RESOLUTION,
            "wipe_contact_z_range": WIPE_CONTACT_Z_RANGE,
            "tissue_initial_xy": TISSUE_WORLD_POS[:2],
            "vase_initial_xy": VASE_WORLD_POS[:2],
            "static_object_xy_tol": STATIC_OBJECT_XY_TOL,
        },
    )


@configclass
class DiningCleanupEnvCfg(SingleArmFrankaTaskEnvCfg):
    """Configuration for the advanced dining cleanup task."""

    scene: DiningCleanupSceneCfg = DiningCleanupSceneCfg(env_spacing=8.0)
    observations: SingleArmFrankaObservationsCfg = SingleArmFrankaObservationsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    task_description: str = (
        "clear the bowl and spoon into the tray, then wipe the left side of the table with the cloth."
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        self.viewer.eye = (0.8, 0.87, 0.67)
        self.viewer.lookat = (0.4, -1.3, -0.2)
        self.dynamic_reset_gripper_effort_limit = False

        self.scene.robot.init_state.pos = (0.35, -0.74, 0.01)
        self.scene.robot.init_state.rot = (0.707, 0.0, 0.0, 0.707)
        self.scene.robot.init_state.joint_pos = {
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

        parse_usd_and_create_subassets(DINING_ROOM_USD_PATH, self)

        self.object_pose_cfg = ObjectPoseConfig(
            tag_to_object=TAG_TO_OBJECT,
            anchor_tag_id=ANCHOR_TAG_ID,
            anchor_world_pose=ANCHOR_WORLD_POSE,
            object_z=OBJECT_Z,
            object_roll=OBJECT_ROLL,
            object_pitch=OBJECT_PITCH,
            per_object_yaw_offset=PER_OBJECT_YAW_OFFSET,
            use_fixed_yaw=False,
        )
