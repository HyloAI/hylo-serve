"""Policy server + workcell on the simulated arm, end to end, with the control loop's measured timing.

Starts `hylo serve policy` (the bring-up wave) and `hylo serve workcell` (the deployments repository's
workcells/sim_arm.yaml: wall clock at 30 Hz, SafetyShield, reset confirmation, operator verdict) as two
subprocesses on free ports, runs N episodes through the workcell API with this script as the operator, then
reads the recorded traces: intervals between executed actions (period jitter, deadline misses), the policy round
trip the runtime measured (p50 / p99), holds and the actions the aligned merge dropped. The episodes are an
ordinary run: `hylo trace` reads it.

    uv run python examples/01_sim_arm_end_to_end.py [--quick] [--out DIR]

Runtime: about 25 s with --quick (2 episodes x 150 steps), about 70 s in full (5 x 300 steps).
Requirements: hylo-serve and hylo-env-lerobot-robot (the workspace venv); CPU only, no robot.
"""

from __future__ import annotations

import time

import numpy as np

from _common import (Procs, hylo_trace, loop_timing, parse_args, pct, read_events, run_operated_episode, show,
                     start_policy, start_workcell, workcell_config, write_result)

SLUG = "sim-arm-end-to-end"


def main() -> int:
    a = parse_args(SLUG, __doc__)
    n, steps = (2, 150) if a.quick else (5, 300)
    procs = Procs(a.out / "logs")
    t0 = time.monotonic()
    try:
        pport = start_policy(procs, {"type": "bringup.wave", "params": {"amplitude": 8, "horizon": 15}})
        cfg = workcell_config(a.out, f"ws://127.0.0.1:{pport}/hylo")
        base = start_workcell(procs, cfg)
        startup_s = time.monotonic() - t0
        episodes = []
        for k in range(n):
            last = run_operated_episode(base, instruction="wave the shoulder", max_steps=steps, seed=k)
            episodes.append(last)
            print(f"episode {k + 1}/{n}: {last['status']} steps={last['steps']} chunks={last['chunks']} holds={last['holds']}")
    finally:
        procs.stop_all()
    run_dir = a.out / "runs" / "workcell-sim-arm"
    rows, all_lat, timings = [], [], []
    for e in episodes:
        tm = loop_timing(read_events(run_dir, e["episode_id"]), hz=30.0)
        timings.append(tm)
        all_lat += tm["latencies_ms"]
        rows.append({"episode": e["episode_id"][-8:], "status": e["status"], "steps": tm["steps"], "hz": tm["achieved_hz"],
                     "jitter_std_ms": tm["jitter_std_ms"], "jitter_p99_ms": tm["jitter_p99_abs_ms"],
                     "max_interval_ms": tm["interval_max_ms"], "misses": tm["deadline_misses"], "chunks": tm["chunks"],
                     "lat_p50_ms": pct(tm["latencies_ms"], 50), "lat_p99_ms": pct(tm["latencies_ms"], 99),
                     "holds": tm["holds"], "dropped/chunk": tm["dropped_per_chunk"]})
    print()
    show(rows)
    summary = {
        "episodes": len(episodes), "steps_total": sum(t["steps"] for t in timings), "control_hz": 30.0,
        "achieved_hz_mean": float(np.mean([t["achieved_hz"] for t in timings])),
        "jitter_std_ms_mean": float(np.mean([t["jitter_std_ms"] for t in timings])),
        "jitter_p99_abs_ms_max": float(max(t["jitter_p99_abs_ms"] for t in timings)),
        "interval_max_ms": float(max(t["interval_max_ms"] for t in timings)),
        "deadline_misses": int(sum(t["deadline_misses"] for t in timings)),
        "policy_latency_p50_ms": pct(all_lat, 50), "policy_latency_p99_ms": pct(all_lat, 99),
        "holds": int(sum(t["holds"] for t in timings)), "chunks": int(sum(t["chunks"] for t in timings)),
        "verdicts": [e["verdict"]["verdict"] if e.get("verdict") else None for e in episodes],
        "startup_s": startup_s,
    }
    trace = hylo_trace(run_dir)
    print("\n`hylo trace` on the workcell run:\n" + "\n".join(trace.splitlines()[:12]))
    print(f"\n{summary['episodes']} episodes, {summary['steps_total']} steps at {summary['achieved_hz_mean']:.2f} Hz "
          f"(target 30): jitter std {summary['jitter_std_ms_mean']:.2f} ms, {summary['deadline_misses']} deadline misses, "
          f"policy round trip p50 {summary['policy_latency_p50_ms']:.1f} ms / p99 {summary['policy_latency_p99_ms']:.1f} ms")
    write_result(a.out, {"example": SLUG, "quick": a.quick, "summary": summary, "episodes": rows,
                         "run_dir": str(run_dir), "hylo_trace": trace.splitlines()[:40]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
