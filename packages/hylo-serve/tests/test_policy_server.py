import json

import pytest
import websockets.sync.client as wsc

from hylo.eval.experiment import run_experiment
from hylo.core.config import ExperimentConfig
from hylo.core.remote import wire
from hylo.serve.policy import PolicyServer

AGENT = {"type": "testing.p_controller", "params": {"kp": 2.0}}
CFG = ExperimentConfig(name="ps", benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"]}},
                       env={"type": "testing.point_reach"}, agent=AGENT)
DOUBLE_OPENPI = {"observation/state": "state:pos=2,vel=2,goal=2,goal_visible=1", "prompt": "instruction"}


@pytest.fixture
def server():
    s = PolicyServer(agent=AGENT, port=0, openpi_inputs=DOUBLE_OPENPI, log=lambda m: None).start()
    yield s
    s.shutdown()


def test_a_served_agent_behaves_like_the_local_one(server, tmp_path):
    local = run_experiment(CFG, run_root=tmp_path, label="local")
    remote = run_experiment(CFG.with_overrides({"agent": {"type": "remote.ws", "params": {"url": f"ws://127.0.0.1:{server.port}/wc"}}}),
                            run_root=tmp_path, label="remote")
    assert remote.per_key() == local.per_key() and server.sessions == 1


def test_an_incompatible_environment_is_refused_before_any_action(server):
    with wsc.connect(f"ws://127.0.0.1:{server.port}/wc") as ws:
        assert json.loads(ws.recv())["hello"] == "hylo.agentserver/v0.1"
        env_spec = {"name": "other", "version": "1", "control_hz": 10,
                    "observation": {"state": {"joint_pos": [6]}}, "action": {"shape": [6], "low": [-1] * 6, "high": [1] * 6}}
        ws.send(json.dumps({"id": 1, "method": "reset", "args": {"episode_id": "e", "seed": 0, "env_spec": env_spec,
                                                                  "task": {"spec": {"id": "t", "version": "1", "instruction": "x"}, "seed": 0}}}))
        reply = json.loads(ws.recv())
    assert not reply["ok"] and "incompatible" in reply["error"]


def test_openpi_clients_can_run_our_agents(server, tmp_path):
    from hylo.policies.openpi.agent import OpenPIAgent

    client = OpenPIAgent(url=f"ws://127.0.0.1:{server.port}/openpi", preset=None, image_size=None, action_dim=2,
                         execution_steps=1, inputs={"observation/state": "state:pos,vel,goal,goal_visible", "prompt": "instruction"})
    assert client.metadata["server"].startswith("hylo.policy")
    via_openpi = run_experiment(CFG, run_root=tmp_path, label="openpi", agent=client)
    local = run_experiment(CFG, run_root=tmp_path, label="local")
    assert via_openpi.per_key() == local.per_key()
    client.close()


def test_one_session_at_a_time(server):
    first = wsc.connect(f"ws://127.0.0.1:{server.port}/openpi")
    wire.unpackb(first.recv())
    second = wsc.connect(f"ws://127.0.0.1:{server.port}/openpi")
    with pytest.raises(Exception):
        second.recv(timeout=5)
    assert second.close_code == 1013
    first.close()


def test_following_a_campaign_rebuilds_the_agent_between_sessions(tmp_path, monkeypatch):
    from hylo.evolve.campaign import EvolveConfig, run_campaign, rollback

    monkeypatch.chdir(tmp_path)
    (tmp_path / "exp.yaml").write_text(json.dumps(CFG.with_overrides({"agent.params.kp": 0.5}).model_dump(mode="json")))
    evo = EvolveConfig(name="c", experiment="exp.yaml", cycles=1, out_dir="camp",
                       experiment_overrides={"benchmark.params": {"tags": ["reach"], "protocol": "rsi"}},
                       proposer={"type": "rule.gain", "params": {"path": "agent.params.kp", "factors": [4.0]}})
    run_campaign(evo, log=lambda m: None)
    s = PolicyServer(campaign="camp", port=0, log=lambda m: None)
    assert s.served.version.startswith("cand-") and s.served.agent.kp == 2.0       # 0.5 x 4, promoted
    rollback(evo, reason="test", by="t")
    assert s.served.refresh() and s.served.version == "base" and s.served.agent.kp == 0.5
    s.shutdown()


def test_following_reads_only_the_active_version_file(tmp_path):
    from hylo.core.deployment import ActiveVersion, write_active

    write_active(tmp_path, ActiveVersion(campaign="c", version=None, agent={**AGENT, "params": {"kp": 0.5}}))
    s = PolicyServer(campaign=str(tmp_path), port=0, log=lambda m: None)
    assert s.served.version == "base" and s.served.agent.kp == 0.5 and not s.served.refresh()
    write_active(tmp_path, ActiveVersion(campaign="c", version="cand-x", agent=AGENT))
    assert s.served.refresh() and s.served.version == "cand-x" and s.served.agent.kp == 2.0
    s.shutdown()
