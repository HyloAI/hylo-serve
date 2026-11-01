"""A real robot as a worldcurrent Environment, through LeRobot's Robot interface (S3).

Any LeRobot robot works (SO-100/101 and Koch followers, LeKiwi, OpenArm, ...; ROS 2 robots through LeRobot's
third-party plugins such as lerobot_robot_ros2_zenoh), and so does SimulatedArm for bring-up without hardware.

What the adapter adds on top of LeRobot's Robot:
- an EnvSpec: observation keys grouped into named state vectors (default: `joint_pos` and `gripper`), cameras
  renamed, an action box with explicit limits and semantics (joint_pos in the robot's units, continuous gripper),
  the control rate, and `real_robot=True` so the SafetyShield aborts on any modification;
- reset: connect if needed, move to the home pose by interpolation (`home_steps` small steps, each within the
  shield's limits), then hand over. The scene is the operator's job: the workcell waits for "reset_done";
- success is never claimed (None): the operator's verdict or a calibrated judge labels the episode;
- hold_action: the current measured position (the robot stays where it is), used by the workcell runtime when
  the policy is late; stop(): hold immediately (e-stop path, in addition to the robot's own hardware e-stop).

    env:
      type: lerobot.robot
      remote: {runtime: robot}            # the profile's runtime: the venv with lerobot and the robot SDK
      params:
        robot: {type: so101_follower, params: {port: /dev/ttyACM0, id: arm1, max_relative_target: 10}}
        cameras: {front: agentview, wrist: wrist}
        action_low: [-110, -100, -100, -100, -160, 0]
        action_high: [110, 100, 100, 100, 160, 100]
        home: [0, -90, 90, 0, 0, 10]
        control_hz: 30
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from worldcurrent.core.task import TaskInstance
from worldcurrent.core.types import (Action, ActionSemantics, BoxSpace, EnvCapabilities, EnvSpec, Observation,
                                     ObservationSpec, StepResult)

# LeRobot config type -> the robots subpackage that registers it
ROBOT_MODULES = {"so100_follower": "so_follower", "so101_follower": "so_follower"}


def make_robot(spec: dict[str, Any] | Any) -> Any:
    if not isinstance(spec, dict):
        return spec                                            # an already built robot (tests, notebooks)
    kind, params = spec["type"], dict(spec.get("params") or {})
    if kind == "sim_arm":
        from .sim_arm import SimulatedArm

        return SimulatedArm(**params)
    if ":" in kind:                                            # "module:factory" for custom robots
        mod, _, attr = kind.partition(":")
        return getattr(importlib.import_module(mod), attr)(**params)
    from lerobot.robots import RobotConfig, make_robot_from_config

    importlib.import_module(spec.get("module") or f"lerobot.robots.{ROBOT_MODULES.get(kind, kind)}")
    return make_robot_from_config(RobotConfig.get_choice_class(kind)(**params))


class LeRobotRobotEnvironment:
    def __init__(self, robot: dict[str, Any] | Any, cameras: dict[str, str] | None = None,
                 state_groups: dict[str, list[str]] | None = None, action_low: list[float] | None = None,
                 action_high: list[float] | None = None, control_mode: str = "joint_pos", gripper: str | None = "continuous",
                 control_hz: float = 30.0, home: list[float] | None = None, home_steps: int = 60,
                 real_robot: bool = True, name: str | None = None) -> None:
        self.robot = make_robot(robot)
        feats = self.robot.observation_features
        self.cam_map = dict(cameras) if cameras is not None else {k: k for k, v in feats.items() if isinstance(v, tuple)}
        scalars = [k for k, v in feats.items() if not isinstance(v, tuple)]
        if state_groups is None:
            grip = [k for k in scalars if k.startswith("gripper")]
            state_groups = {"joint_pos": [k for k in scalars if k not in grip], **({"gripper": grip} if grip else {})}
        self.state_groups = state_groups
        self.action_keys = list(self.robot.action_features)
        n = len(self.action_keys)
        if action_low is None or action_high is None:
            if real_robot:
                raise ValueError("a real robot needs explicit action_low / action_high (joint limits in its units)")
            action_low, action_high = [-180.0] * n, [180.0] * n
        if len(action_low) != n or len(action_high) != n:
            raise ValueError(f"action limits must have {n} entries ({self.action_keys})")
        self.low, self.high = np.asarray(action_low, float), np.asarray(action_high, float)
        self.control_mode, self.gripper, self.control_hz = control_mode, gripper, control_hz
        self.home = None if home is None else np.asarray(home, float)
        self.home_steps, self.real_robot = home_steps, real_robot
        self.name = name or f"lerobot.{getattr(self.robot, 'name', type(self.robot).__name__)}"
        self._last_obs: dict[str, Any] | None = None
        self._instruction = ""

    # -- Environment ---------------------------------------------------------------------------------------------
    def spec(self) -> EnvSpec:
        feats = self.robot.observation_features
        return EnvSpec(
            name=self.name, version="0.1",
            observation=ObservationSpec(
                state={g: (len(keys),) for g, keys in self.state_groups.items()},
                cameras={ours: tuple(feats[theirs]) for theirs, ours in self.cam_map.items()},
                units={g: "robot units (degrees / 0-100 gripper)" for g in self.state_groups}),
            action=BoxSpace(shape=(len(self.action_keys),), low=tuple(self.low), high=tuple(self.high),
                            semantics=ActionSemantics(control_mode=self.control_mode, frame="joint", gripper=self.gripper,
                                                      dim_labels=tuple(self.action_keys))),
            control_hz=self.control_hz,
            capabilities=EnvCapabilities(seedable=False, snapshot=False, privileged_state=False, real_robot=self.real_robot))

    def _obs(self) -> Observation:
        raw = self.robot.get_observation()
        self._last_obs = raw
        state = {g: np.array([float(raw[k]) for k in keys], dtype=np.float32) for g, keys in self.state_groups.items()}
        images = {ours: np.asarray(raw[theirs], dtype=np.uint8) for theirs, ours in self.cam_map.items() if theirs in raw}
        return Observation(state=state, images=images, instruction=self._instruction)

    def _measured_action(self) -> np.ndarray:
        raw = self._last_obs or self.robot.get_observation()
        return np.array([float(raw[k]) for k in self.action_keys])

    def _send(self, vec: np.ndarray) -> dict[str, Any]:
        vec = np.clip(np.asarray(vec, float).reshape(-1), self.low, self.high)   # the box, whatever the caller did
        return self.robot.send_action({k: float(v) for k, v in zip(self.action_keys, vec)})

    def reset(self, task: TaskInstance) -> Observation:
        if not self.robot.is_connected:
            self.robot.connect()
        self._instruction = task.spec.instruction
        obs = self._obs()
        if self.home is not None:                                # interpolate to the home pose in small steps
            start = self._measured_action()
            for t in np.linspace(0.0, 1.0, max(2, self.home_steps))[1:]:
                self._send(start + t * (self.home - start))
                obs = self._obs()
        return obs

    def step(self, action: Action) -> StepResult:
        sent = self._send(np.asarray(action.data, float))
        obs = self._obs()
        return StepResult(observation=obs, reward=None, terminated=False, truncated=False, success=None,
                          info={"sent": {k: round(float(v), 4) for k, v in sent.items()}})

    def hold_action(self, last: np.ndarray | None) -> np.ndarray:
        return self._measured_action()

    def stop(self) -> None:
        if self.robot.is_connected:
            self._send(self._measured_action())

    def close(self) -> None:
        if self.robot.is_connected:
            self.robot.disconnect()
