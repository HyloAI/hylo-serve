"""Shared helpers for the serve examples: free ports, subprocesses that are always cleaned up, the workcell API
over HTTP, timing statistics from a recorded trace, and the result table.

Not an example itself (the leading underscore). Everything here uses public interfaces: the `hylo` command,
the workcell HTTP API (hylo.workcell/v0.1) and the trace store (hylo.core.trace).
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]            # the serve repository
WORKSPACE = ROOT.parent


def deployments_dir() -> Path:
    """The deployments repository (a lab's own, or the template): $HYLO_DEPLOYMENTS, else the side-by-side checkout."""
    return Path(os.environ.get("HYLO_DEPLOYMENTS", WORKSPACE / "deployments-template"))


def parse_args(slug: str, doc: str, extra: Any = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=doc.strip().splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="a shorter run (under a minute)")
    ap.add_argument("--out", default=str(ROOT / "runs" / "examples" / slug), help="output directory")
    if extra is not None:
        extra(ap)
    a = ap.parse_args()
    a.out = Path(a.out).resolve()
    a.out.mkdir(parents=True, exist_ok=True)
    return a


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Procs:
    """Subprocesses started by an example; `stop_all` (in a finally) terminates every one of them."""

    def __init__(self, logdir: Path) -> None:
        self.logdir, self.procs = logdir, {}
        logdir.mkdir(parents=True, exist_ok=True)

    def start(self, name: str, args: Sequence[str], env: dict[str, str] | None = None) -> subprocess.Popen:
        log = open(self.logdir / f"{name}.log", "w")
        p = subprocess.Popen([sys.executable, "-m", "hylo.core.cli", *args], stdout=log, stderr=subprocess.STDOUT,
                             env={**os.environ, **(env or {})}, start_new_session=True)
        self.procs[name] = p
        return p

    def stop(self, name: str, sig: int = signal.SIGTERM) -> None:
        p = self.procs.pop(name, None)
        if p is None or p.poll() is not None:
            return
        os.killpg(p.pid, sig)
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=10)

    def stop_all(self) -> None:
        for name in list(self.procs):
            self.stop(name)

    def log(self, name: str) -> str:
        return (self.logdir / f"{name}.log").read_text(errors="replace")[-2000:]


def http(method: str, url: str, body: dict[str, Any] | None = None, timeout: float = 10.0) -> tuple[int, Any]:
    req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode(), method=method,
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except ValueError:
            return exc.code, None


def wait_http(url: str, timeout: float = 60.0, procs: Procs | None = None, name: str | None = None) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except (urllib.error.URLError, ConnectionError, OSError):
            if procs is not None and name in procs.procs and procs.procs[name].poll() is not None:
                raise RuntimeError(f"{name} exited:\n{procs.log(name)}")
            time.sleep(0.1)
    raise TimeoutError(f"{url} did not come up in {timeout:.0f}s")


def start_policy(procs: Procs, agent: dict[str, Any] | None = None, *, campaign: str | Path | None = None,
                 name: str = "policy", env: dict[str, str] | None = None) -> int:
    port = free_port()
    src = ["--campaign", str(campaign)] if campaign is not None else ["--agent", json.dumps(agent)]
    procs.start(name, ["serve", "policy", *src, "--port", str(port)], env=env)
    wait_http(f"http://127.0.0.1:{port}/healthz", procs=procs, name=name)
    return port


def workcell_config(out: Path, policy_url: str, *, runtime: dict[str, Any] | None = None, **top: Any) -> Path:
    """The deployments repository's sim_arm workcell, pointed at this example's policy server and output dir."""
    import yaml

    cfg = yaml.safe_load((deployments_dir() / "workcells" / "sim_arm.yaml").read_text())
    cfg["policy"] = {**cfg.get("policy", {}), "url": policy_url}
    cfg["runtime"] = {**cfg.get("runtime", {}), **(runtime or {})}
    cfg["output_dir"] = str(out / "runs")
    cfg.update(top)
    path = out / "workcell.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def start_workcell(procs: Procs, config: Path, name: str = "workcell") -> str:
    port = free_port()
    procs.start(name, ["serve", "workcell", "--config", str(config), "--port", str(port)])
    base = f"http://127.0.0.1:{port}"
    wait_http(base + "/healthz", procs=procs, name=name)
    return base


def wait_state(base: str, states: Sequence[str], timeout: float = 60.0, need_last: bool = False) -> dict[str, Any]:
    deadline, st = time.monotonic() + timeout, {}
    while time.monotonic() < deadline:
        _, st = http("GET", base + "/status")
        if st["state"] in states and (not need_last or st.get("last")):
            return st
        time.sleep(0.02)
    raise TimeoutError(f"workcell stayed in {st.get('state')!r}, expected {list(states)}: {st}")


def run_operated_episode(base: str, *, instruction: str, max_steps: int, verdict: str | None = "success",
                         seed: int = 0, timeout: float = 120.0) -> dict[str, Any]:
    """Start an episode and act as the operator: confirm the reset, then give the verdict when asked."""
    before = (http("GET", base + "/status")[1].get("last") or {}).get("episode_id")
    code, body = http("POST", base + "/episodes", {"instruction": instruction, "max_steps": max_steps, "seed": seed})
    if code != 200:
        raise RuntimeError(f"POST /episodes -> {code}: {body}")
    deadline, sent = time.monotonic() + timeout, set()
    while time.monotonic() < deadline:
        _, st = http("GET", base + "/status")
        if st["state"] == "awaiting_reset" and "reset" not in sent:
            http("POST", base + "/operator", {"type": "reset_done", "by": "example"})
            sent.add("reset")
        elif st["state"] == "awaiting_verdict" and verdict and "verdict" not in sent:
            http("POST", base + "/operator", {"type": "verdict", "verdict": verdict, "by": "example"})
            sent.add("verdict")
        elif st["state"] in ("idle", "fault") and (st.get("last") or {}).get("episode_id", before) != before:
            return st["last"]
        elif st["state"] in ("idle", "fault") and (st.get("last") or {}).get("error"):
            raise RuntimeError(f"episode failed: {st['last']['error']}")
        time.sleep(0.02)
    raise TimeoutError(f"the episode did not finish in {timeout:.0f}s: {st}")


# ------------------------------------------------------------------ trace analysis

def read_events(run_dir: Path, episode_id: str) -> list[dict[str, Any]]:
    from hylo.core.trace import TraceStore

    return TraceStore(run_dir).read(episode_id).events


def pct(xs: Sequence[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(xs, float), q)) if len(xs) else None


def loop_timing(events: list[dict[str, Any]], hz: float) -> dict[str, Any]:
    """Control-loop timing from the trace: intervals between executed actions (wall clock), the policy round
    trips the runtime measured, holds and the actions the aligned merge dropped."""
    period = 1.0 / hz
    t = np.array([e["wall_time"] for e in events if e["type"] == "action"], float)
    dt = np.diff(t) if len(t) > 1 else np.zeros(0)
    chunks = [e["payload"] for e in events if e["type"] == "agent.chunk"]
    lat = [float(c.get("latency_s") or 0.0) for c in chunks]
    prov = [c.get("provenance") or {} for c in chunks]
    return {
        "steps": int(len(t)), "period_ms": period * 1000,
        "interval_mean_ms": float(dt.mean() * 1000) if len(dt) else None,
        "jitter_std_ms": float(dt.std() * 1000) if len(dt) else None,
        "jitter_p99_abs_ms": pct(np.abs(dt - period) * 1000, 99) if len(dt) else None,
        "interval_max_ms": float(dt.max() * 1000) if len(dt) else None,
        "deadline_misses": int((dt > 1.5 * period).sum()),       # a tick later than half a period
        "achieved_hz": float(1.0 / dt.mean()) if len(dt) else None,
        "chunks": len(chunks),
        "latencies_ms": [x * 1000 for x in lat],
        "dropped_per_chunk": float(np.mean([p.get("dropped", 0) for p in prov])) if prov else None,
        "holds": sum(1 for e in events if e["type"] == "x.workcell.hold"),
    }


def show(rows: list[dict[str, Any]], cols: Sequence[str] | None = None) -> None:
    cols = list(cols or rows[0].keys())

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"
        return "-" if v is None else str(v)

    cells = [[fmt(r.get(c)) for c in cols] for r in rows]
    w = [max(len(c), *(len(x[i]) for x in cells)) for i, c in enumerate(cols)]
    print("  ".join(c.ljust(w[i]) for i, c in enumerate(cols)))
    print("  ".join("-" * w[i] for i in range(len(cols))))
    for x in cells:
        print("  ".join(v.ljust(w[i]) for i, v in enumerate(x)))


def write_result(out: Path, result: dict[str, Any]) -> Path:
    path = out / "result.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nresult: {path}")
    return path


def hylo_trace(run_dir: Path, episode_id: str | None = None) -> str:
    """`hylo trace` on a recorded run: proof that a workcell episode is an ordinary run."""
    args = [sys.executable, "-m", "hylo.core.cli", "trace", str(run_dir), *([episode_id] if episode_id else [])]
    res = subprocess.run(args, capture_output=True, text=True, timeout=120)
    if res.returncode != 0:
        raise RuntimeError(f"hylo trace failed: {res.stderr[-800:]}")
    return res.stdout
