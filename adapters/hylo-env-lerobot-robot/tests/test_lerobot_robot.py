import numpy as np
import pytest

from hylo.core.agent import AgentBase, AgentSpec
from hylo.core.task import TaskInstance, TaskSpec
from hylo.core.types import ActionChunk, ActionSemantics
from hylo.envs.lerobot_robot.env import LeRobotRobotEnvironment
from hylo.serve.runtime import RealtimeConfig, RealtimeEpisodeRunner
from hylo.testing.conformance import check_environment

LIMITS = {"action_low": [-120, -120, -120, -120, -170, 0], "action_high": [120, 120, 120, 120, 170, 100]}
TASK = TaskInstance(spec=TaskSpec(id="arm/wave", version="1", instruction="wave", max_steps=40), seed=0)


def _env(**kw):
    return LeRobotRobotEnvironment(robot={"type": "sim_arm", "params": {"image_size": 32}}, cameras={"front": "agentview"},
                                   home=[0, -30, 60, 0, 0, 20], home_steps=10, control_hz=50, **LIMITS, **kw)


def test_spec_maps_lerobot_features_to_named_state_cameras_and_a_bounded_action_box():
    spec = _env().spec()
    assert spec.observation.state == {"joint_pos": (5,), "gripper": (1,)}
    assert spec.observation.cameras == {"agentview": (32, 32, 3)}
    assert spec.action.semantics.control_mode == "joint_pos" and spec.action.semantics.dim_labels[-1] == "gripper.pos"
    assert spec.capabilities.real_robot and not spec.capabilities.seedable
    with pytest.raises(ValueError, match="explicit action_low"):
        LeRobotRobotEnvironment(robot={"type": "sim_arm"})


def test_conformance_without_determinism():
    rep = check_environment(_env, TASK)                         # determinism is skipped: not seedable
    assert rep.ok, rep.summary()


class Wave(AgentBase):
    """Absolute joint targets around home; `late` makes it slower than the control loop."""

    def __init__(self, amp=15.0):
        self.amp = amp

    def spec(self):
        return AgentSpec(name="wave", version="0", required_state=("joint_pos", "gripper"), action_dim=6,
                         action_semantics=ActionSemantics(control_mode="joint_pos", frame="joint", gripper="continuous"))

    def act(self, obs):
        base = np.concatenate([obs.state["joint_pos"], obs.state["gripper"]])
        t = np.arange(10)[:, None]
        chunk = np.repeat(base[None], 10, 0) + np.r_[self.amp, 0, 0, 0, 0, 0] * np.sin(t / 3)
        return ActionChunk(actions=chunk, execution_steps=10)


def test_the_robot_holds_its_measured_position_while_the_policy_is_late():
    env = _env()
    runner = RealtimeEpisodeRunner(env, Wave(amp=5.0), config=RealtimeConfig(latency_s=0.2, refill_steps=0),
                                   safety={"max_delta": 20.0})
    res = runner.run(TASK)
    assert res.extra["status"] == "unlabeled" and res.outcome.success is None   # a robot never claims success
    assert res.holds >= 10 and res.outcome.steps == 40 and res.safety_interventions == 0


def test_real_robot_mode_aborts_instead_of_clipping():
    env = _env()
    runner = RealtimeEpisodeRunner(env, Wave(amp=500.0), config=RealtimeConfig(latency_s=0.0), safety={"max_delta": 20.0})
    res = runner.run(TASK)
    assert res.outcome.status == "safety_abort" and res.safety_interventions >= 1
    before = env._measured_action()
    env.stop()                                                   # the e-stop path holds where the arm is
    assert np.allclose(env.robot.target, before)
