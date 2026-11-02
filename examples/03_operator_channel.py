"""Operator channel: the e-stop stops actions within one control period and stays latched until cleared; an
episode does not move before the operator confirms the reset; the operator's verdict labels the episode.

Starts `hylo serve policy` (the bring-up wave) and `hylo serve workcell` (the sim_arm workcell: 30 Hz wall clock,
reset confirmation, operator verdict) and drives them through the workcell API (hylo.workcell/v0.1), as the
playground's Operator page does:

1. e-stop trials: start an episode, confirm the reset, press the e-stop mid-episode; measure the time from the
   request to the last executed action and to the abort in the trace, and the actions executed after it;
2. latch: while the fault is latched every new episode is refused (409), however long one waits; clear_fault
   releases it;
3. reset confirmation: a new episode waits in `awaiting_reset` without moving the arm until `reset_done`;
4. verdict: the episode ends `unlabeled` (a robot cannot judge itself) until the operator says success /
   partial / failure; the label lands in the run's evallog.json with who gave it.

    uv run python examples/03_operator_channel.py [--quick] [--out DIR]

Runtime: about 25 s with --quick (3 e-stop trials), about 60 s in full (10 trials).
Requirements: hylo-serve and hylo-env-lerobot-robot (the workspace venv); CPU only.
"""

from __future__ import annotations

import json
import time

from _common import (Procs, hylo_trace, http, parse_args, pct, read_events, show, start_policy, start_workcell,
                     wait_state, workcell_config, write_result)

SLUG = "operator-channel"
HZ = 30.0


def started(base: str, max_steps: int = 900, seed: int = 0) -> None:
    code, body = http("POST", base + "/episodes", {"instruction": "wave the shoulder", "max_steps": max_steps, "seed": seed})
    if code != 200:
        raise RuntimeError(f"POST /episodes -> {code}: {body}")


def main() -> int:
    a = parse_args(SLUG, __doc__)
    trials = 3 if a.quick else 10
    procs = Procs(a.out / "logs")
    estops, latch, reset, verdict = [], {}, {}, {}
    try:
        pport = start_policy(procs, {"type": "bringup.wave", "params": {"amplitude": 8, "horizon": 15}})
        base = start_workcell(procs, workcell_config(a.out, f"ws://127.0.0.1:{pport}/hylo"))
        run_dir = a.out / "runs" / "workcell-sim-arm"

        # 1. e-stop mid-episode
        for k in range(trials):
            started(base, seed=k)
            wait_state(base, ("awaiting_reset",))
            http("POST", base + "/operator", {"type": "reset_done", "by": "operator"})
            time.sleep(0.6 + 0.137 * k)                                # mid-episode, at a different phase each time
            t_press = time.time()
            code, ack = http("POST", base + "/operator", {"type": "estop", "by": "operator"})
            t_ack = time.time()
            last = wait_state(base, ("fault", "idle"), need_last=True)["last"]
            while last.get("episode_id") is None or (estops and last["episode_id"] == estops[-1]["episode_id"]):
                last = wait_state(base, ("fault", "idle"), need_last=True)["last"]
            ev = read_events(run_dir, last["episode_id"])
            acts = [e["wall_time"] for e in ev if e["type"] == "action"]
            err = next(e["wall_time"] for e in ev if e["type"] == "error")
            estops.append({"trial": k + 1, "episode_id": last["episode_id"], "status": last["status"],
                           "steps": last["steps"], "ack_state": ack["state"], "ack_ms": (t_ack - t_press) * 1000,
                           "last_action_after_ms": (acts[-1] - t_press) * 1000,
                           "abort_after_ms": (err - t_press) * 1000,
                           "actions_after": sum(1 for t in acts if t > t_press), "reason": last["reason"],
                           "estop_in_trace": any(e["type"] == "operator.estop" for e in ev)})
            # 2. the fault is latched: refused now, and still refused later
            code_now, body = http("POST", base + "/episodes", {"instruction": "wave the shoulder"})
            if k == 0:
                time.sleep(3.0)
                code_later, _ = http("POST", base + "/episodes", {"instruction": "wave the shoulder"})
                latch = {"refused_now": code_now, "refused_after_3s": code_later, "detail": body.get("detail"),
                         "status_state": http("GET", base + "/status")[1]["state"]}
            assert code_now == 409, (code_now, body)
            _, ack = http("POST", base + "/operator", {"type": "clear_fault", "by": "operator"})
            estops[-1]["cleared_state"] = ack["state"]
            print(f"e-stop {k + 1}/{trials}: {last['status']} after {last['steps']} steps; last action "
                  f"{estops[-1]['last_action_after_ms']:.1f} ms after the press, then latched until cleared")

        # 3. reset confirmation: nothing moves before reset_done
        started(base, max_steps=60)
        wait_state(base, ("awaiting_reset",))
        time.sleep(2.0)
        st = http("GET", base + "/status")[1]
        eid = st["episode"]
        t_reset = time.time()
        http("POST", base + "/operator", {"type": "reset_done", "by": "operator"})
        # 4. verdict: the episode waits for the operator's outcome
        st2 = wait_state(base, ("awaiting_verdict",), timeout=30)
        time.sleep(0.5)
        http("POST", base + "/operator", {"type": "verdict", "verdict": "partial", "by": "operator", "note": "half wave"})
        last = wait_state(base, ("idle",), need_last=True)["last"]
        while last["episode_id"] != eid:
            last = wait_state(base, ("idle",), need_last=True)["last"]
        ev = read_events(run_dir, eid)
        acts = [e["wall_time"] for e in ev if e["type"] == "action"]
        reset = {"state_while_waiting": st["state"], "waited_s": 2.0,
                 "actions_before_reset_done": sum(1 for t in acts if t < t_reset),
                 "first_action_after_reset_ms": (acts[0] - t_reset) * 1000 if acts else None}
        sample = [s for s in json.loads((run_dir / "evallog.json").read_text())["samples"] if s["episode_id"] == eid][0]
        verdict = {"state_before": st2["state"], "status": last["status"], "label": sample["labels"]["success"],
                   "trace_events": sorted({e["type"] for e in ev if e["type"].startswith("operator.")})}
    finally:
        procs.stop_all()

    print()
    show(estops, ["trial", "status", "steps", "ack_ms", "last_action_after_ms", "abort_after_ms", "actions_after",
                  "estop_in_trace", "ack_state", "cleared_state"])
    lat = [e["last_action_after_ms"] for e in estops]
    summary = {"trials": trials, "period_ms": 1000 / HZ,
               "last_action_after_estop_ms_p50": pct(lat, 50), "last_action_after_estop_ms_max": max(lat),
               "abort_after_estop_ms_max": max(e["abort_after_ms"] for e in estops),
               "actions_after_estop_max": max(e["actions_after"] for e in estops),
               "latched": latch, "reset_confirmation": reset, "verdict": verdict}
    print(f"\nlatch: new episode -> {latch['refused_now']} at once and {latch['refused_after_3s']} after 3 s "
          f"({latch['detail']}); state {latch['status_state']!r} until clear_fault")
    print(f"reset: {reset['actions_before_reset_done']} actions in {reset['waited_s']:.0f} s of awaiting_reset; first action "
          f"{reset['first_action_after_reset_ms']:.0f} ms after reset_done")
    print(f"verdict: {verdict['state_before']} -> {verdict['status']}, label {verdict['label']}")
    print(f"operator events in the trace: {verdict['trace_events']}")
    trace = hylo_trace(run_dir)
    print("\n`hylo trace` on the workcell run:\n" + "\n".join(trace.splitlines()[:6]))
    write_result(a.out, {"example": SLUG, "quick": a.quick, "summary": summary, "estops": estops,
                         "run_dir": str(run_dir), "hylo_trace": trace.splitlines()[:40]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
