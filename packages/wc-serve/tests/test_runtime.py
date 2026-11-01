import threading

import pytest

from worldcurrent.eval.experiment import run_experiment
from worldcurrent.core.config import ExperimentConfig
from worldcurrent.core.errors import SafetyAbort
from worldcurrent.core.registry import registry
from worldcurrent.core.task import TaskInstance
from worldcurrent.core.trace import TraceStore
from worldcurrent.serve.runtime import OperatorChannel, RealtimeConfig, RealtimeEpisodeRunner

CFG = ExperimentConfig(name="rt", benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"]}},
                       env={"type": "testing.point_reach"}, agent={"type": "testing.p_controller", "params": {"kp": 2.0}})


def _tasks():
    bench = registry.build("benchmarks", CFG.benchmark.model_dump(exclude={"split"}))
    return bench.instances(CFG.benchmark.split)


def _runner(tmp_path, agent_params=None, **cfg):
    env = registry.build("envs", {"type": "testing.point_reach"})
    agent = registry.build("agents", {"type": "testing.p_controller", "params": {"kp": 2.0, **(agent_params or {})}})
    store = TraceStore(tmp_path / "rt")
    return RealtimeEpisodeRunner(env, agent, config=RealtimeConfig(**cfg), safety=CFG.safety, trace=store), store


def test_zero_latency_reproduces_the_synchronous_kernel(tmp_path):
    sync = run_experiment(CFG, run_root=tmp_path, label="sync")
    runner, _ = _runner(tmp_path, latency_s=0.0, refill_steps=0)
    outcomes = {t.key: runner.run(t).outcome for t in _tasks()}
    assert {k: (o.success, o.steps) for k, o in outcomes.items()} == \
        {s["key"]: (s["status"] == "success", s["steps"]) for s in sync.log["samples"]}


def test_latency_is_absorbed_by_dropping_stale_actions_and_holding(tmp_path):
    runner, store = _runner(tmp_path, agent_params={"horizon": 8, "execution_steps": 8}, latency_s=0.25, refill_steps=2)
    res = runner.run(_tasks()[0])
    assert res.outcome.success and res.holds > 0                       # it still reaches the goal
    chunks = list(store.read(res.outcome.episode_id).of_type("agent.chunk"))
    dropped = [c["payload"]["provenance"]["dropped"] for c in chunks]
    assert dropped[0] == round(0.25 * runner.hz) and all(d == dropped[0] for d in dropped)
    assert all(c["payload"]["provenance"]["queued"] == 8 - dropped[0] for c in chunks)


def test_the_watchdog_aborts_when_no_actions_arrive(tmp_path):
    runner, _ = _runner(tmp_path, latency_s=100.0, max_hold_steps=5)
    res = runner.run(_tasks()[0])
    assert res.outcome.status == "error" and "watchdog" in res.outcome.termination_reason and res.holds == 6


def test_estop_is_latched_until_cleared(tmp_path):
    runner, _ = _runner(tmp_path, latency_s=0.0)
    runner.operator.post({"type": "estop", "by": "test"})
    with pytest.raises(SafetyAbort, match="latched"):
        runner.run(_tasks()[0])
    runner.operator.post({"type": "clear_fault"})
    assert runner.run(_tasks()[0]).outcome.success


class _Unjudged:
    """A robot-like env: no success signal, the operator decides."""

    def __init__(self):
        self.env = registry.build("envs", {"type": "testing.point_reach"})

    def spec(self):
        return self.env.spec()

    def reset(self, task):
        return self.env.reset(task)

    def step(self, action):
        r = self.env.step(action)
        return r.model_copy(update={"success": None, "terminated": False})

    def close(self):
        self.env.close()


def test_operator_verdict_labels_an_episode_the_env_cannot_judge(tmp_path):
    agent = registry.build("agents", {"type": "testing.p_controller", "params": {"kp": 2.0}})
    op = OperatorChannel()
    runner = RealtimeEpisodeRunner(_Unjudged(), agent, config=RealtimeConfig(latency_s=0.0, verdict_timeout_s=5.0,
                                                                           require_reset_confirm=True),
                                   trace=TraceStore(tmp_path / "op"), operator=op)
    task = _tasks()[0].model_copy(update={"spec": _tasks()[0].spec.model_copy(update={"max_steps": 30})})
    threading.Timer(0.3, lambda: op.post({"type": "reset_done"})).start()
    threading.Timer(0.6, lambda: op.post({"type": "verdict", "verdict": "partial", "by": "tester"})).start()
    res = runner.run(task)
    assert res.verdict["verdict"] == "partial" and res.outcome.status == "failure" and res.outcome.steps == 30


def test_wall_clock_runs_the_policy_in_a_worker_and_keeps_the_rate(tmp_path):
    runner, store = _runner(tmp_path, agent_params={"horizon": 4, "execution_steps": 4}, clock="wall", control_hz=200.0)
    res = runner.run(_tasks()[0])
    assert res.outcome.success and res.chunks > 0
    steps = [e["payload"]["step"] for e in store.read(res.outcome.episode_id).of_type("step.result")]
    assert steps == list(range(1, len(steps) + 1))
