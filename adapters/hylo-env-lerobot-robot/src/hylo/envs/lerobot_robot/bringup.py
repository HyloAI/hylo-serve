"""Bring-up policy: small sinusoidal moves around the pose the arm is in, joint by joint.

Not a task policy. It checks the chain (policy server -> workcell -> adapter -> motors) with motions that stay
within a few degrees of the measured pose, so the shield's limits and the robot's own limits are never near.
The motion is a function of executed steps (counted in on_step), not of policy calls: under asynchronous
execution the runtime drops the actions that went stale during inference, and a chunk indexed by steps stays
continuous with the one before it.
"""

from __future__ import annotations

import numpy as np

from hylo.core.agent import AgentBase, AgentSpec
from hylo.core.types import ActionChunk, ActionSemantics, Observation


class BringupWave(AgentBase):
    def __init__(self, amplitude: float = 5.0, period_steps: int = 30, horizon: int = 15, joint: int | None = 0) -> None:
        self.amplitude, self.period, self.horizon, self.joint = amplitude, period_steps, horizon, joint
        self.steps, self.center = 0, None

    def spec(self) -> AgentSpec:
        return AgentSpec(name="bringup.wave", version="0.1", required_state=("joint_pos", "gripper"), action_dim=6,
                         action_semantics=ActionSemantics(control_mode="joint_pos", frame="joint", gripper="continuous"))

    def reset(self, ctx) -> None:
        super().reset(ctx)
        self.steps, self.center = 0, None

    def on_step(self, result) -> None:
        self.steps += 1

    def act(self, observation: Observation) -> ActionChunk:
        pose = np.concatenate([observation.state["joint_pos"], observation.state["gripper"]]).astype(float)
        if self.center is None:
            self.center = pose
        k = np.arange(self.steps, self.steps + self.horizon)          # the chunk starts at the observed step
        offs = np.zeros((self.horizon, len(pose)))
        j = self.joint if self.joint is not None else (self.steps // (2 * self.period)) % (len(pose) - 1)
        offs[:, j] = self.amplitude * np.sin(2 * np.pi * k / self.period)
        return ActionChunk(actions=self.center[None] + offs, execution_steps=self.horizon,
                           provenance={"model": "bringup.wave", "joint": int(j)})
