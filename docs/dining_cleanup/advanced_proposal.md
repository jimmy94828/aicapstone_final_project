# AI-Capstone Advanced Proposal (Group 13)

## Overview

The advanced task extends the entry-level cutlery arrangement task into a post-meal dining cleanup scenario. A Franka robot arm must clear a dining table by moving a bowl and a spoon into a tray, then grasp a cloth and wipe the dirty area of the table. The table also contains protected objects, currently a tissue box and a vase, that should remain in place throughout the episode.

Compared with the entry-level task, this task is more demanding because it combines sequential object manipulation, object category distinction, constrained placement, cloth-mediated table wiping, and obstacle preservation. The robot must not simply move every object on the table. It must identify which objects are task targets, which objects are tools, and which objects are protected scene objects.

The current implementation is registered as:

```text
HCIS-DiningCleanup-SingleArm-v0
```

## Motivation

Dining cleanup is a common household activity that is repetitive but still requires structured physical reasoning. A useful assistive robot should be able to remove tableware, preserve non-target objects, and clean the table surface after a meal. This is relevant for home assistance, elder care, mobility assistance, and service robotics.

The entry-level project focuses on pre-meal setup: placing a fork and knife around a plate. The advanced task instead focuses on post-meal cleanup. This shift changes the problem from a single placement objective into a multi-stage manipulation problem with explicit constraints on object identity, placement region, wiping coverage, and disturbance avoidance.

## Problem Formulation

We formulate dining cleanup as a finite-horizon manipulation task. At each timestep `t`, the environment state `s_t` contains the robot joint state, gripper state, object poses, camera observations, and task progress. The policy receives observations `o_t` from the Franka proprioceptive state and enabled cameras, and outputs an action `a_t` controlling the arm and gripper.

For the current Franka task, the action vector is:

```text
[panda_joint1, ..., panda_joint7, gripper]
```

The objective is to learn or script a policy `pi(a_t | o_t)` that completes the following ordered task:

1. Move the bowl into the tray.
2. Move the spoon into the tray.
3. Grasp the cloth.
4. Wipe the target dirty region on the left side of the table.
5. Keep the protected objects, tissue and vase, within a small displacement tolerance.

The task is successful only if all required subgoals are satisfied at the end of the episode:

```text
Success = placement_success
          AND wipe_success
          AND protected_object_stability
          AND no_object_drop
```

### Objects and Roles
Object USD files: https://drive.google.com/drive/folders/1FHOizW83ahsts2zrd36hD3e29t2Z2hLI?usp=drive_link
The current scene contains six task-relevant objects:

| Object | Role | Current placement policy |
|--------|------|--------------------------|
| bowl | target object to clear | randomized by `object_poses` |
| spoon | target object to clear | randomized by `object_poses` |
| tray | target placement region | fixed |
| cloth | wiping tool | fixed |
| tissue | protected object | fixed |
| vase | protected object | fixed |

Only the bowl and spoon are randomized per episode. The tray, cloth, tissue, and vase are fixed in the current environment to make the advanced task focus on manipulation sequencing, target-object cleanup, and wiping behavior.

### Workspace Definition

The dining table is represented in world XY coordinates:

```text
table x range = [0.00, 0.70]
table y range = [-0.65, 0.00]
table surface z = 0.05
```

The convention used by the current task is:

```text
+x = Franka-view right side
-x = Franka-view left side
```

The target dirty region to wipe is the left side of the table:

```text
wipe x range = [0.04, 0.22]
wipe y range = [-0.50, -0.15]
```

The fixed object positions are:

```text
tray   = (0.57, -0.36, 0.05)
tissue = (0.35, -0.18, 0.074)
vase   = (0.35, -0.32, 0.05)
cloth  = (0.35, -0.49, 0.05)
```

## Environment Settings and Dataset Generation

### Environment Settings

The scene is based on the dining room table environment. The Franka robot is placed in front of the table and operates over a constrained tabletop workspace. The environment is configured in:

```text
packages/simulator/src/simulator/tasks/dining_cleanup/dining_cleanup_env_cfg.py
```

The task uses `RigidObjectCfg` objects for the bowl, spoon, tray, tissue, vase, and cloth. The camera-enabled setup provides wrist and front camera observations when launched with `--enable_cameras`.

### Object Pose Generation

The current object pose generator creates randomized bowl and spoon initial poses. The generator is:

```text
scripts/generate_dining_cleanup_object_poses.py
```

The default generated file is:

```text
data/dining_clean/dining_cleanup_object_poses_500.json
```

The current generator samples bowl and spoon world positions in the dirty-area region:

```text
object world x range = [0.10, 0.24]
object world y range = [-0.50, -0.22]
```

The generator also enforces clearance constraints so that the bowl and spoon do not overlap each other or the fixed objects. The generated data follows the existing `object_poses` schema and is loaded through the task's `ObjectPoseConfig`. The anchor configuration is:

```text
anchor tag id = 0
anchor world pose = (0.40, 0.10, 0.0)
tag-to-object mapping = {1: bowl, 2: spoon}
```

### Demonstration Generation

Two demonstration sources are supported:

1. Scripted finite-state-machine demonstrations through `scripts/datagen/generate.py`.
2. Human keyboard teleoperation through `scripts/teleop.py`.

The scripted FSM is implemented in:

```text
packages/simulator/src/simulator/datagen/state_machine/dining_cleanup.py
```

It generates trajectories for bowl pickup and placement, spoon pickup and placement, cloth pickup, and table wiping. Teleoperation is used for manual inspection and small-scale human demonstrations.

## Planned Implementation Approach

The current implementation follows two stages: tableware clearing and table wiping.

### Stage 1: Clear the Tableware

The robot first clears the target tableware into the tray. The scripted order is:

1. Move above the bowl.
2. Approach the bowl with an edge-biased grasp target.
3. Close the gripper.
4. Lift the bowl.
5. Move above the tray.
6. Place the bowl in the tray on the `+y` side of the tray center.
7. Repeat the same sequence for the spoon, placing it on the `-y` side of the tray center.

The bowl uses an edge-grasp retreat from the bowl center toward the robot base. This is necessary because grasping the bowl center is unstable for the gripper geometry. The spoon uses a smaller retreat because it is a narrow elongated object.

### Stage 2: Wipe the Table

After the bowl and spoon are placed in the tray, the robot grasps the cloth and wipes the target dirty region on the left side of the table. The current scripted wipe pattern uses three lanes:

```text
wipe lane x positions = [0.08, 0.135, 0.19]
wipe y range = [-0.50, -0.15]
```

The cloth footprint is approximated as:

```text
cloth footprint = 0.069 m x 0.115 m
```

This lane design covers approximately 96.9 percent of the target wipe region under the planned footprint model, while keeping the cloth footprint away from the tissue and vase area. The success threshold is currently set to 90 percent coverage.

## Expected Outcome and Evaluation

The task is evaluated using placement success, wiping coverage, protected-object stability, and object-drop safety.

### 1. Tableware Placement Success

The bowl and spoon must both be inside the tray success zone. For each object, the position must satisfy:

```text
abs(object_x - tray_x) <= 0.12
abs(object_y - tray_y) <= 0.13
tray_z - 0.05 <= object_z <= tray_z + 0.10
```

In addition, the current task expects:

```text
bowl_y > tray_y
spoon_y < tray_y
```

A placement score can be reported as:

```text
Score_clear = number_of_correctly_placed_tableware / number_of_tableware
```

For the current task, `number_of_tableware = 2`.

### 2. Wiping Coverage

The target wipe region is discretized into a grid with 0.01 m resolution. A grid cell is considered cleaned if the cloth footprint covers that cell while the cloth is within the contact height range:

```text
wipe contact z range = [0.03, 0.13]
```

The coverage metric is:

```text
Coverage = cleaned_target_cells / total_target_cells
```

The current success threshold is:

```text
Coverage >= 0.90
```

### 3. Protected Object Stability

The tissue and vase should not be displaced by the robot, the carried objects, or the cloth. Stability is measured by the XY displacement from each object's initial position:

```text
Displacement_i = norm(final_xy_i - initial_xy_i)
```

The current tolerance is:

```text
Displacement_i <= 0.035 m
```

for both the tissue and the vase.

### 4. Object Drop Safety

The task should fail if any task object falls below the table surface. In the current datagen script, an object is treated as fallen if its root z position drops below:

```text
z < 0.0
```

### 5. Overall Success Criteria

An episode is successful if all of the following are true:

1. The bowl is placed inside the tray and on the `+y` side of the tray center.
2. The spoon is placed inside the tray and on the `-y` side of the tray center.
3. The wipe coverage over the target dirty region is at least 90 percent.
4. The tissue and vase remain within 0.035 m of their initial XY positions.
5. No task object falls off the table.

## Anticipated Challenges

### 1. Multi-stage Task Sequencing

The task requires a strict order of operations. The robot must clear the tableware before wiping, because the dirty region initially contains the bowl and spoon. Starting the wipe stage too early may push target objects, block the cloth trajectory, or disturb the protected objects.

### 2. Object-specific Grasping

The bowl and spoon require different grasp strategies. The bowl has a curved geometry and is more stable when approached with an edge-biased grasp. The spoon is elongated and thin, so gripper alignment and grasp height are more sensitive. A single generic grasp strategy is unlikely to be robust for both objects.

### 3. Preserving Non-target Objects

The robot must distinguish target objects from protected scene objects. The tissue and vase are not cleanup targets. They should remain fixed even though they are close enough to the robot's workspace to be disturbed by poor approach, transport, or wiping motions.

### 4. Wiping Coverage

Unlike a placement-only task, wiping is evaluated over an area rather than a point target. The cloth must cover enough of the dirty region while maintaining table contact. The trajectory must balance coverage, reachability, and obstacle clearance.

### 5. Rigid Cloth Approximation

The current cloth is represented as a rigid object with an approximate footprint. This makes the task tractable in simulation but differs from real deformable cloth behavior. A future version may replace this approximation with deformable cloth simulation or a more detailed contact model.

### 6. Demonstration Quality

The dataset must contain temporally smooth and physically plausible trajectories. Large jumps, unstable grasps, or inconsistent success labels can degrade imitation learning performance. This is especially important because the task has multiple stages and delayed success conditions.

## Current Scope and Future Extensions

The current scope intentionally fixes the tray, cloth, tissue, and vase positions. This keeps the task focused on bowl/spoon cleanup and left-table wiping. Future extensions may randomize tray placement, randomize protected-object locations, add more tableware items, or introduce adaptive wiping trajectories that plan around obstacle positions.
