"""Serving a campaign's active version: a promotion and a rollback reach the robot between sessions, never during one.

1. Runs a small improvement campaign on MuJoCo planar reach with manual approval (`hylo evolve run`): the IK
   controller starts with a weak gain (kp 0.02); a candidate with a larger gain passes the gate and waits for
   approval. The campaign's active.json names the base version.
2. Starts `hylo serve policy --campaign DIR`, which serves whatever active.json names.
3. Runs the held-out split through the policy server (`hylo run` with a `remote.ws` agent; each run is one
   session), four times: base; a long run during which the candidate is approved (`hylo evolve approve`) - the
   session in flight keeps the version it started with; after the approval (the new version); after
   `hylo evolve rollback` (the base again, episode for episode).

Every phase is an ordinary run (split heldout of the client config), readable by `hylo trace`.

    uv run python examples/04_campaign_hot_switch.py [--quick] [--out DIR]

Runtime: about 35 s with --quick, about 45 s in full (more held-out seeds).
Requirements: hylo-evolve, hylo-env-mujoco and hylo-serve (the workspace venv); CPU only.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from _common import Procs, hylo_trace, http, parse_args, show, start_policy, write_result

SLUG = "campaign-hot-switch"

EXPERIMENT = {"schema_version": "hylo.experiment/v0.1", "name": "mujoco-reach",
              "benchmark": {"type": "mujoco/reach-v0", "split": "validation"}, "env": {"type": "mujoco.planar_reach"},
              "agent": {"type": "baseline.planar_ik", "params": {"kp": 0.3}}, "output_dir": "runs"}


def hylo(*args: str, cwd: Path) -> str:
    res = subprocess.run([sys.executable, "-m", "hylo.core.cli", *args], cwd=cwd, capture_output=True, text=True, timeout=600)
    if res.returncode != 0:
        raise RuntimeError(f"hylo {' '.join(args)} failed:\n{(res.stderr or res.stdout)[-1500:]}")
    return res.stdout


def run_dirs(root: Path) -> set[Path]:
    return {p.parent for p in root.glob("*/manifest.json")}


def summarize(run_dir: Path) -> dict:
    log = json.loads((run_dir / "evallog.json").read_text())
    samples = sorted(log["samples"], key=lambda s: (s["task_id"], s["seed"]))
    return {"run": run_dir.name, "episodes": len(samples),
            "success": sum(s["status"] == "success" for s in samples) / len(samples),
            "mean_steps": sum(s["steps"] for s in samples) / len(samples),
            "steps_by_episode": [s["steps"] for s in samples]}


def main() -> int:
    a = parse_args(SLUG, __doc__)
    out = a.out
    heldout = 10 if a.quick else 25                                # seeds per task (2 tasks)
    camp = out / "campaign"
    (out / "configs").mkdir(exist_ok=True)
    (out / "configs" / "mujoco_reach.yaml").write_text(yaml.safe_dump(EXPERIMENT, sort_keys=False))
    (out / "configs" / "campaign.yaml").write_text(yaml.safe_dump({
        "schema_version": "hylo.evolve/v0.1", "name": "reach-gain-manual", "experiment": "mujoco_reach.yaml",
        "experiment_overrides": {"agent.params.kp": 0.02, "workers": 16, "benchmark.params": {"protocol": "rsi", "gate_looks": 2}},
        "proposer": {"type": "rule.gain", "params": {"path": "agent.params.kp", "factors": [2.0]}},
        "protocol": {"alpha": 0.05, "gate_looks": 2, "stage1_pairs": 20}, "cycles": 1, "approval": "manual",
        "out_dir": str(camp)}, sort_keys=False))
    if camp.exists():
        import shutil

        shutil.rmtree(camp)
    t0 = time.monotonic()
    hylo("evolve", "run", str(out / "configs" / "campaign.yaml"), cwd=out)
    pending = json.loads((camp / "pending_approval.json").read_text())
    campaign_s = time.monotonic() - t0
    print(f"campaign: {pending['candidate_id']} passed the gate and waits for approval ({campaign_s:.0f} s)")

    runs = out / "runs"
    procs = Procs(out / "logs")
    phases = []
    try:
        port = start_policy(procs, campaign=camp)
        client = out / "configs" / "client.yaml"

        def client_cfg(n: int) -> None:
            client.write_text(yaml.safe_dump({
                "schema_version": "hylo.experiment/v0.1", "name": "served-reach",
                "benchmark": {"type": "mujoco/reach-v0", "split": "heldout", "params": {"heldout": n}},
                "env": {"type": "mujoco.planar_reach"}, "agent": {"type": "remote.ws", "params": {"url": f"ws://127.0.0.1:{port}/hylo"}},
                "workers": 1, "output_dir": str(runs)}, sort_keys=False))

        def phase(name: str, n: int, during=None) -> dict:
            client_cfg(n)
            before = run_dirs(runs)
            p = subprocess.Popen([sys.executable, "-m", "hylo.core.cli", "run", str(client)], cwd=out,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            extra = during(p) if during else {}
            _, err = p.communicate(timeout=600)
            if p.returncode != 0:
                raise RuntimeError(f"hylo run failed:\n{err[-1500:]}")
            (new,) = run_dirs(runs) - before
            s = summarize(new)
            meta = http("GET", f"http://127.0.0.1:{port}/meta")[1]
            active = json.loads((camp / "active.json").read_text())
            row = {"phase": name, "served_version": meta["version"], "active_json": active["version"] or "base",
                   "kp": active["agent"]["params"]["kp"] if (active["version"] or "base") == meta["version"] else None,
                   **s, **extra}
            phases.append(row)
            print(f"{name:<24} served {row['served_version']:<16} success {s['success']:.2f}  mean steps {s['mean_steps']:.1f}")
            return row

        base = phase("base", heldout)

        def approve_mid_session(p: subprocess.Popen) -> dict:
            log = out / "logs" / "policy.log"
            sessions = log.read_text().count("session from")
            while log.read_text().count("session from") == sessions:     # the run's session has started
                time.sleep(0.01)
            t = time.monotonic()
            hylo("evolve", "approve", str(camp), cwd=out)
            return {"approved_mid_session": p.poll() is None, "approve_s": time.monotonic() - t}

        during = phase("approve during session", 4 * heldout, approve_mid_session)
        promoted = phase("after approval", heldout)
        hylo("evolve", "rollback", str(camp), "--reason", "example: simulated field regression", cwd=out)
        rolled = phase("after rollback", heldout)
        switches = [line for line in procs.log("policy").splitlines() if "serving version" in line]
    finally:
        procs.stop_all()

    print()
    show(phases, ["phase", "served_version", "active_json", "kp", "episodes", "success", "mean_steps"])
    identical = base["steps_by_episode"] == rolled["steps_by_episode"]
    print(f"\nduring the session that saw the approval: served {during['served_version']} while active.json said "
          f"{during['active_json']} (approved mid-session: {during['approved_mid_session']})")
    print(f"after rollback, every episode matches the base run step for step: {identical}")
    print("policy server: " + "; ".join(s.split("] ", 1)[-1] for s in switches))
    trace = hylo_trace(runs / rolled["run"])
    print("\n`hylo trace` on the last phase:\n" + "\n".join(trace.splitlines()[:6]))
    for r in phases:
        r.pop("steps_by_episode")
    write_result(out, {"example": SLUG, "quick": a.quick, "candidate": pending["candidate_id"], "campaign_s": campaign_s,
                       "phases": phases, "rollback_restores_base_exactly": identical, "policy_log": switches,
                       "hylo_trace": trace.splitlines()[:30]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
