"""Watchdog and hold: inject policy latency, a stall and a dropped connection between the workcell and its policy.

A small fault-injecting WebSocket proxy (in this script) sits between `hylo serve workcell` (the sim_arm
workcell, wall clock at 30 Hz, watchdog after 60 consecutive holds) and `hylo serve policy` (the bring-up wave).
It forwards the Hylo agent protocol unchanged, except that it delays the replies to `act` or closes the
connection, on a schedule. Five episodes:

- baseline: no fault;
- latency: every policy reply 100 ms late (3 control periods): the aligned merge drops the stale actions;
- short stall: one reply 1.0 s late: the arm holds at its measured pose, then the episode resumes and completes;
- long stall: one reply 3.0 s late: after 60 holds (2 s) the watchdog aborts the episode, the arm is stopped;
- dropped connection: the proxy closes the socket: the episode ends with an error at the next policy call.

Every episode is recorded in the workcell's run; the measured effects come from the traces (holds, longest hold
streak, the arm's joint speed while moving and while holding, time from the fault to the first hold and to the
abort).

    uv run python examples/02_watchdog_and_hold.py [--quick] [--out DIR]

Runtime: about 35 s with --quick (episodes of 150 steps), about 60 s in full (300 steps).
Requirements: hylo-serve, hylo-env-lerobot-robot, websockets (the workspace venv); CPU only.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

import numpy as np

from _common import (Procs, hylo_trace, loop_timing, parse_args, pct, read_events, run_operated_episode, show,
                     start_policy, start_workcell, workcell_config, write_result)

SLUG = "watchdog-and-hold"
HZ = 30.0


class FaultProxy:
    """WebSocket relay for the Hylo agent protocol that delays `act` replies or drops the connection."""

    def __init__(self, upstream: str) -> None:
        self.upstream, self.port = upstream, 0
        self.configure()
        self._server = None

    def configure(self, delay_s: float = 0.0, stall_at: int | None = None, stall_s: float = 0.0,
                  drop_at: int | None = None) -> None:
        self.delay_s, self.stall_at, self.stall_s, self.drop_at = delay_s, stall_at, stall_s, drop_at
        self.acts, self.fault_t = 0, None

    def _handle(self, conn) -> None:
        from websockets.sync.client import connect

        with connect(self.upstream + conn.request.path, max_size=None, compression=None) as up:
            conn.send(up.recv())                                         # the server's hello
            for frame in conn:
                is_act = json.loads(frame).get("method") == "act"
                if is_act:
                    self.acts += 1
                    if self.acts == self.drop_at:
                        self.fault_t = time.time()
                        conn.close()
                        return
                up.send(frame)
                reply = up.recv()
                if is_act:
                    late = self.delay_s
                    if self.acts == self.stall_at:
                        self.fault_t, late = time.time(), late + self.stall_s
                    if late:
                        time.sleep(late)
                conn.send(reply)

    def start(self) -> "FaultProxy":
        import websockets.sync.server as wss

        self._server = wss.serve(self._handle, "127.0.0.1", 0, max_size=None, compression=None)
        self.port = self._server.socket.getsockname()[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()


def hold_effects(events: list[dict[str, Any]], fault_t: float | None) -> dict[str, Any]:
    """The arm while holding: joint speed during hold steps vs. while moving; time from the fault to the first
    hold and to the end of the episode; the longest hold streak."""
    pos, held, best, step = {}, set(), 0, 0
    for e in events:
        if e["type"] == "x.workcell.hold":
            held.add(e["payload"]["step"])
            best = max(best, e["payload"]["consecutive"])
        elif e["type"] == "step.result":
            step = e["payload"]["step"]
        elif e["type"] == "observation" and e["payload"].get("state"):
            pos[step] = np.asarray(e["payload"]["state"]["joint_pos"], float)
    steps = sorted(pos)
    speed = {s: float(np.abs(pos[s] - pos[s - 1]).max()) for s in steps if s - 1 in pos}
    # a hold at step s is executed between observation s and s+1
    hold_speed = [speed[s + 1] for s in held if s + 1 in speed and s > 2]
    move_speed = [v for s, v in speed.items() if (s - 1) not in held and s > 2]
    out: dict[str, Any] = {"longest_hold_streak": best,
                           "hold_settle_max_deg": max(hold_speed[:3]) if len(hold_speed) > 3 else None,
                           "hold_speed_max_deg": max(hold_speed[3:]) if len(hold_speed) > 3 else None,
                           "moving_speed_p50_deg": pct(move_speed, 50)}
    if fault_t is not None:
        holds_after = [e["wall_time"] for e in events if e["type"] == "x.workcell.hold" and e["wall_time"] >= fault_t]
        err = next((e["wall_time"] for e in events if e["type"] == "error"), None)
        out["first_hold_after_fault_ms"] = (holds_after[0] - fault_t) * 1000 if holds_after else None
        out["abort_after_fault_ms"] = (err - fault_t) * 1000 if err is not None else None
    return out


def main() -> int:
    a = parse_args(SLUG, __doc__)
    steps = 150 if a.quick else 300
    at = 6                                                         # the fault hits at the 6th policy call (~step 55)
    scenarios = [("baseline", {}), ("latency 100 ms", {"delay_s": 0.1}),
                 ("stall 1.0 s", {"stall_at": at, "stall_s": 1.0}), ("stall 3.0 s", {"stall_at": at, "stall_s": 3.0}),
                 ("dropped connection", {"drop_at": at})]
    procs, proxy = Procs(a.out / "logs"), None
    results = []
    try:
        pport = start_policy(procs, {"type": "bringup.wave", "params": {"amplitude": 8, "horizon": 15}})
        proxy = FaultProxy(f"ws://127.0.0.1:{pport}").start()
        cfg = workcell_config(a.out, f"ws://127.0.0.1:{proxy.port}/hylo")
        base = start_workcell(procs, cfg)
        for k, (name, fault) in enumerate(scenarios):
            proxy.configure(**fault)
            last = run_operated_episode(base, instruction="wave the shoulder", max_steps=steps, seed=k)
            results.append((name, fault, last, proxy.fault_t))
            print(f"{name:<20} {last['status']:<8} steps={last['steps']} holds={last['holds']} reason={last['reason'][:70]}")
    finally:
        if proxy is not None:
            proxy.stop()
        procs.stop_all()
    run_dir = a.out / "runs" / "workcell-sim-arm"
    rows = []
    for name, fault, last, fault_t in results:
        ev = read_events(run_dir, last["episode_id"])
        tm = loop_timing(ev, HZ)
        eff = hold_effects(ev, fault_t)
        rows.append({"scenario": name, "status": last["status"], "steps": last["steps"], "holds": tm["holds"],
                     "max_streak": eff["longest_hold_streak"], "dropped/chunk": tm["dropped_per_chunk"],
                     "lat_p50_ms": pct(tm["latencies_ms"], 50), "settle_deg": eff["hold_settle_max_deg"],
                     "hold_speed_deg": eff["hold_speed_max_deg"],
                     "move_speed_deg": eff["moving_speed_p50_deg"],
                     "1st_hold_ms": eff.get("first_hold_after_fault_ms"), "abort_ms": eff.get("abort_after_fault_ms"),
                     "reason": last["reason"][:60], "policy_error": last.get("policy_error"),
                     "episode_id": last["episode_id"]})
    print()
    show(rows, ["scenario", "status", "steps", "holds", "max_streak", "dropped/chunk", "lat_p50_ms", "move_speed_deg",
                "settle_deg", "hold_speed_deg", "1st_hold_ms", "abort_ms"])
    print("\nreasons:")
    for r in rows:
        print(f"  {r['scenario']:<20} {r['reason']}" + (f"  (then: {r['policy_error'][:60]})" if r["policy_error"] else ""))
    trace = hylo_trace(run_dir)
    print("\n`hylo trace` on the workcell run:\n" + "\n".join(trace.splitlines()[:8]))
    write_result(a.out, {"example": SLUG, "quick": a.quick, "control_hz": HZ, "max_hold_steps": 60, "fault_at_call": at,
                         "scenarios": rows, "run_dir": str(run_dir), "hylo_trace": trace.splitlines()[:40]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
