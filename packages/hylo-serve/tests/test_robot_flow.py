"""The real-robot path end to end, on the simulated arm: policy server -> workcell (wall clock) -> LeRobot-style
robot adapter, with the operator confirming the reset and judging the episode."""

import json
import time

from fastapi.testclient import TestClient

from hylo.serve.policy import PolicyServer
from hylo.serve.workcell import WorkcellConfig, create_app


def test_operator_confirms_reset_and_labels_a_robot_episode(tmp_path):
    policy = PolicyServer(agent={"type": "bringup.wave", "params": {"amplitude": 6, "horizon": 10}}, port=0,
                          log=lambda m: None).start()
    cfg = WorkcellConfig.model_validate({
        "name": "arm", "output_dir": str(tmp_path), "max_steps": 60,
        "env": {"type": "lerobot.robot", "params": {
            "robot": {"type": "sim_arm", "params": {"image_size": 32}}, "cameras": {"front": "agentview"},
            "action_low": [-120, -120, -120, -120, -170, 0], "action_high": [120, 120, 120, 120, 170, 100],
            "home": [0, -30, 60, 0, 0, 20], "home_steps": 5, "control_hz": 120}},
        "policy": {"url": f"ws://127.0.0.1:{policy.port}/wc"},
        "runtime": {"clock": "wall", "refill_steps": 4, "max_hold_steps": 200, "require_reset_confirm": True,
                    "verdict_timeout_s": 20},
        "safety": {"max_delta": 15.0}})
    c = TestClient(create_app(cfg))
    assert c.post("/episodes", json={"instruction": "wave hello"}).status_code == 200
    deadline = time.time() + 30
    while c.get("/status").json()["state"] != "awaiting_reset":
        assert time.time() < deadline
        time.sleep(0.05)
    c.post("/operator", json={"type": "reset_done", "by": "tester"})
    while c.get("/status").json()["state"] not in ("running",) and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(1.0)                                              # 60 steps at 120 Hz, then it waits for a verdict
    c.post("/operator", json={"type": "verdict", "verdict": "success", "by": "tester"})
    while (st := c.get("/status").json())["state"] != "idle" or not st.get("last"):
        assert time.time() < deadline, st
        time.sleep(0.05)
    assert st["last"]["status"] == "success" and st["last"]["steps"] == 60 and st["last"]["chunks"] > 0
    s = json.loads((tmp_path / "workcell-arm" / "evallog.json").read_text())["samples"][0]
    assert s["labels"]["success"] == {"value": 1.0, "by": "operator:tester", "verdict": "success"}
    assert s["task_id"].startswith("workcell/wave_hello")
    policy.shutdown()
