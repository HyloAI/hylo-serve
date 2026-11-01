"""SimulatedArm: a stand-in with LeRobot's Robot interface, for bringing up a workcell without hardware.

Five revolute joints and a gripper, in degrees and 0-100 like LeRobot's SO follower, with first-order motor
dynamics (each joint closes a fraction of the gap to its target per step), a per-step limit on the commanded
change (LeRobot's max_relative_target) and an overhead camera that draws the arm's projected joint positions.
It never claims success: on a real robot the operator (or a calibrated judge) decides.
"""

from __future__ import annotations

from typing import Any

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


class SimulatedArm:
    name = "sim_arm"

    def __init__(self, id: str = "sim", gain: float = 0.35, max_relative_target: float | None = 10.0,
                 camera: str | None = "front", image_size: int = 96, seed: int = 0) -> None:
        self.id, self.gain, self.max_relative_target = id, gain, max_relative_target
        self.camera, self.size = camera, image_size
        self.pos = np.zeros(len(JOINTS) + 1)
        self.pos[-1] = 50.0                                      # gripper half open
        self.target = self.pos.copy()
        self._connected = False
        self.rng = np.random.default_rng(seed)

    # -- LeRobot Robot interface -------------------------------------------------------------------------------
    @property
    def observation_features(self) -> dict[str, Any]:
        feats: dict[str, Any] = {f"{j}.pos": float for j in JOINTS} | {"gripper.pos": float}
        if self.camera:
            feats[self.camera] = (self.size, self.size, 3)
        return feats

    @property
    def action_features(self) -> dict[str, Any]:
        return {f"{j}.pos": float for j in JOINTS} | {"gripper.pos": float}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _image(self) -> np.ndarray:
        img = np.full((self.size, self.size, 3), 235, np.uint8)
        x, y, ang = self.size / 2, self.size * 0.9, -np.pi / 2
        for k, q in enumerate(self.pos[:4]):                     # planar projection of the first four joints
            ang += np.deg2rad(q) if k else np.deg2rad(q) * 0.5
            nx, ny = x + np.cos(ang) * self.size * 0.2, y + np.sin(ang) * self.size * 0.2
            for t in np.linspace(0, 1, 24):
                px, py = int(x + (nx - x) * t), int(y + (ny - y) * t)
                img[max(0, py - 1):py + 2, max(0, px - 1):px + 2] = (40, 70, 160)
            x, y = nx, ny
        g = int(4 + self.pos[-1] / 20)
        img[max(0, int(y) - g):int(y) + g, max(0, int(x) - 2):int(x) + 3] = (230, 110, 30)
        return img

    def get_observation(self) -> dict[str, Any]:
        if not self._connected:
            raise ConnectionError("SimulatedArm is not connected")
        self.pos += self.gain * (self.target - self.pos)            # motors move between observations
        obs: dict[str, Any] = {f"{j}.pos": float(v) for j, v in zip(JOINTS, self.pos[:-1])}
        obs["gripper.pos"] = float(self.pos[-1])
        if self.camera:
            obs[self.camera] = self._image()
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self._connected:
            raise ConnectionError("SimulatedArm is not connected")
        goal = np.array([float(action[f"{j}.pos"]) for j in JOINTS] + [float(action["gripper.pos"])])
        if self.max_relative_target is not None:                 # what LeRobot's ensure_safe_goal_position does
            goal = np.clip(goal, self.pos - self.max_relative_target, self.pos + self.max_relative_target)
        self.target = goal
        return {f"{j}.pos": float(v) for j, v in zip(JOINTS, goal[:-1])} | {"gripper.pos": float(goal[-1])}

    def disconnect(self) -> None:
        self._connected = False
