import json
import time

import pytest
from fastapi.testclient import TestClient

from worldcurrent.serve.policy import PolicyServer
from worldcurrent.serve.workcell import WorkcellConfig, create_app


@pytest.fixture
def policy():
    s = PolicyServer(agent={"type": "testing.p_controller", "params": {"kp": 2.0, "horizon": 5, "execution_steps": 5}},
                     port=0, log=lambda m: None).start()
    yield s
    s.shutdown()


def _cfg(tmp_path, policy, **runtime):
    return WorkcellConfig(name="double", env={"type": "testing.point_reach"},
                          benchmark={"type": "testing/reach-v0", "params": {"tags": ["reach"]}},
                          policy={"url": f"ws://127.0.0.1:{policy.port}/wc"},
                          runtime={"latency_s": 0.05, "refill_steps": 1, **runtime}, output_dir=str(tmp_path))


def _wait_idle(c, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = c.get("/status").json()
        if st["state"] in ("idle", "fault") and st.get("last"):
            return st
        time.sleep(0.05)
    raise AssertionError(f"workcell never finished: {st}")


def test_an_episode_runs_with_a_remote_policy_and_is_recorded(tmp_path, policy):
    c = TestClient(create_app(_cfg(tmp_path, policy)))
    tasks = c.get("/status").json()["tasks"]
    with c.websocket_connect("/events") as ws:
        assert c.post("/episodes", json={"task_id": tasks[0][0], "seed": 10000}).status_code == 200
        kinds = set()
        while True:
            m = ws.receive_json()
            kinds.add(m["type"])
            if m["type"] in ("done", "failed"):
                break
    assert m["type"] == "done" and m["status"] == "success", m
    assert {"episode.start", "observation", "chunk", "action", "episode.end"} <= kinds
    log = json.loads((tmp_path / "workcell-double" / "evallog.json").read_text())
    s = log["samples"][0]
    assert s["status"] == "success" and s["labels"]["success"]["by"] == "env" and log["benchmark"]["split"] == "deployment"


def test_estop_latches_and_blocks_new_episodes_until_cleared(tmp_path, policy):
    c = TestClient(create_app(_cfg(tmp_path, policy)))
    task = c.get("/status").json()["tasks"][0][0]
    assert c.post("/operator", json={"type": "estop", "by": "t"}).json()["state"] == "fault"
    r = c.post("/episodes", json={"task_id": task, "seed": 10000})
    assert r.status_code == 409 and "latched" in r.json()["detail"]
    assert c.post("/operator", json={"type": "launch"}).status_code == 400          # wc.workcell/v0.1 events only
    assert c.post("/episodes", json={"task": task}).status_code == 400
    c.post("/operator", json={"type": "clear_fault"})
    assert c.post("/episodes", json={"task_id": task, "seed": 10000}).status_code == 200
    assert _wait_idle(c)["last"]["status"] == "success"


def test_a_dead_policy_endpoint_fails_cleanly(tmp_path, policy):
    cfg = WorkcellConfig.model_validate({**_cfg(tmp_path, policy).model_dump(mode="json"),
                                         "policy": {"url": "ws://127.0.0.1:9/wc", "params": {"connect_timeout_s": 1}}})
    c = TestClient(create_app(cfg))
    c.post("/episodes", json={"task_id": c.get("/status").json()["tasks"][0][0]})
    st = _wait_idle(c)
    assert "error" in st["last"] and st["state"] == "idle"
