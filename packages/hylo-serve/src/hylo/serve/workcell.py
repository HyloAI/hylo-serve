"""Workcell service (S2): runs next to the robot (or simulator) and executes episodes with a remote policy.

It owns the environment, the SafetyShield, the real-time runner and the trace, so safety never depends on the
network: if the policy server stalls the robot holds and the watchdog aborts; an e-stop is latched here until an
operator clears it. The policy is reached over the network (`policy.protocol`: "hylo" = the Hylo agent protocol,
"openpi" = openpi's). Episodes are recorded as an ordinary run (split `deployment` by default, which no learner
reads unless asked) so replay, judges and the catalog work on them.

    hylo serve workcell --config workcell.yaml --port 8100

    name: sim-libero
    env: {type: libero, remote: {...}}                     # any env plugin: a simulator or a robot adapter
    benchmark: {type: libero, params: {suite: libero_10}}  # optional: named tasks and init states
    policy: {url: "ws://gpu:8000/hylo", protocol: hylo}
    runtime: {clock: wall, refill_steps: 10, max_hold_steps: 40, require_reset_confirm: true, verdict_timeout_s: 60}
    safety: {max_delta: 0.5}

HTTP API: hylo.workcell/v0.1 (core.remote.workcell): GET /healthz, GET /status, POST /episodes, POST /operator,
WS /events (live stream).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import Field, ValidationError

from hylo.core.config import ComponentRef, EnvRef
from hylo.core.contract import Contract, JsonAny
from hylo.core.events import Event, EventBus
from hylo.core.registry import registry
from hylo.core.remote.workcell import OperatorAck, OperatorEvent, PolicyEndpoint, StartEpisode
from hylo.core.task import TaskInstance, TaskSpec
from hylo.core.trace import TraceStore

from .runtime import OperatorChannel, RealtimeConfig, RealtimeEpisodeRunner


class WorkcellConfig(Contract):
    name: str
    env: EnvRef                               # local plugin, or a subprocess in another venv (remote)
    benchmark: ComponentRef | None = None
    policy: PolicyEndpoint | None = None
    runtime: RealtimeConfig = Field(default_factory=RealtimeConfig)
    safety: dict[str, JsonAny] = Field(default_factory=dict)
    image_every: int = 10                     # frames kept in the trace (every frame is streamed)
    split: str = "deployment"
    max_steps: int = 600                      # free-form instructions (no benchmark task)
    output_dir: str = "runs/workcell"

    @classmethod
    def load(cls, path: str | Path) -> "WorkcellConfig":
        import yaml

        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


def _jpeg(img: Any, quality: int = 75) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _num(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return np.round(v.astype(float), 4).tolist()
    if isinstance(v, dict):
        return {k: _num(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_num(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


class _Instruction:
    """Agent proxy: the agent hears the operator's instruction; the task and its success check are unchanged."""

    def __init__(self, inner: Any, instruction: str | None) -> None:
        self.inner, self.instruction = inner, instruction

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def act(self, observation):
        if self.instruction:
            observation = observation.model_copy(update={"instruction": self.instruction})
        return self.inner.act(observation)


class Workcell:
    def __init__(self, cfg: WorkcellConfig) -> None:
        self.cfg = cfg
        self.operator = OperatorChannel()
        self.env: Any = None
        self.status: dict[str, Any] = {"state": "idle", "episode": None, "last": None}
        self.run_dir = Path(cfg.output_dir) / f"workcell-{cfg.name}"
        self.store = TraceStore(self.run_dir)
        self._listeners: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # -- live stream -------------------------------------------------------------------------------------------
    def subscribe(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._listeners.append((loop, q))
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._listeners = [(lp, x) for lp, x in self._listeners if x is not q]

    def _broadcast(self, msg: dict[str, Any]) -> None:
        for loop, q in list(self._listeners):
            loop.call_soon_threadsafe(lambda q=q: q.full() or q.put_nowait(msg))

    def _on_event(self, e: Event, counter: dict[str, int]) -> None:
        p = dict(e.payload)
        if e.type == "observation":
            imgs = p.get("images") or {}
            self._broadcast({"type": "observation", "step": counter["step"], "instruction": p.get("instruction"),
                             "images": {c: _jpeg(v) for c, v in imgs.items()},
                             "state": _num({k: np.asarray(v).reshape(-1) for k, v in (p.get("state") or {}).items()})})
            if counter["obs"] % max(1, self.cfg.image_every):
                e = e.model_copy(update={"payload": {**p, "images": {}}})
            counter["obs"] += 1
        elif e.type == "action":
            counter["step"] += 1
            self._broadcast({"type": "action", "step": counter["step"], "data": _num(np.asarray(p["data"]).reshape(-1))})
        elif e.type == "agent.chunk":
            self._broadcast({"type": "chunk", "step": counter["step"], "latency_s": round(float(p.get("latency_s") or 0), 3),
                             "provenance": _num({k: v for k, v in (p.get("provenance") or {}).items() if k != "seeds"})})
        elif e.type == "safety.review" and p.get("modified"):
            d = np.abs(np.asarray(p["approved"], float) - np.asarray(p["proposed"], float))
            self._broadcast({"type": "safety", "step": counter["step"], "reasons": p.get("reasons"), "max_change": round(float(d.max()), 4)})
        elif e.type in ("episode.start", "episode.end", "error", "x.workcell.hold") or e.type.startswith("operator."):
            if e.type == "operator.awaiting_reset":
                self.status["state"] = "awaiting_reset"
            elif e.type == "episode.start" or e.type == "operator.reset_done":
                self.status["state"], self.status["episode"] = "running", e.episode_id
            elif e.type == "operator.awaiting_verdict":
                self.status["state"] = "awaiting_verdict"              # the operator decides the outcome
            self._broadcast({"type": e.type, "step": counter["step"], "payload": _num({k: v for k, v in p.items() if k != "runtime"})})
        self.store.write(e)

    # -- episodes --------------------------------------------------------------------------------------------
    def _build_env(self) -> Any:
        spec = self.cfg.env
        if spec.remote is not None:                   # the env lives in its own venv (simulators, robot SDKs)
            from hylo.core.remote.client import RemoteEnvironment

            return RemoteEnvironment(registry.reference("envs", spec.type), spec.params, python=spec.remote.python,
                                     env=spec.remote.env, runtime=spec.remote.runtime)
        return registry.build("envs", spec.model_dump(exclude={"remote"}))

    def _agent(self, policy: PolicyEndpoint) -> Any:
        if policy.protocol == "openpi":
            return registry.build("agents", {"type": "openpi", "params": {"url": policy.url, "api_key": policy.token, **policy.params}})
        return registry.build("agents", {"type": "remote.ws", "params": {"url": policy.url, "token": policy.token, **policy.params}})

    def _task(self, req: dict[str, Any]) -> TaskInstance:
        if self.cfg.benchmark is not None and req.get("task_id"):
            bench = registry.build("benchmarks", self.cfg.benchmark.model_dump())
            split = req.get("split") or "validation"
            inst = [i for i in bench.instances(split) if i.spec.id == req["task_id"]]
            if not inst:
                raise ValueError(f"task {req['task_id']} is not in split {split}")
            seed = int(req.get("seed", inst[0].seed))
            t = next((i for i in inst if i.seed == seed), inst[0].model_copy(update={"seed": seed}))
            return t.model_copy(update={"split": self.cfg.split})
        instruction = (req.get("instruction") or "").strip()
        if not instruction:
            raise ValueError("give a task_id (with a benchmark) or an instruction")
        slug = "_".join(instruction.lower().split())[:80]
        return TaskInstance(spec=TaskSpec(id=f"workcell/{slug}", version="1", instruction=instruction,
                                          max_steps=int(req.get("max_steps") or self.cfg.max_steps)),
                            seed=int(req.get("seed", 0)), split=self.cfg.split)

    def start(self, req: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("an episode is running")
            if self.operator.fault:
                raise RuntimeError(f"latched fault: {self.operator.fault}; clear it first")
            policy = PolicyEndpoint.model_validate(req["policy"]) if req.get("policy") else self.cfg.policy
            if policy is None:
                raise ValueError("no policy endpoint (in the request or the workcell config)")
            task = self._task(req)
            override = (req.get("instruction") or "").strip() if req.get("task_id") else None
            self.status = {"state": "starting", "episode": None, "task": task.spec.id, "policy": policy.url,
                           "last": self.status.get("last")}
            self._thread = threading.Thread(target=self._run, args=(task, policy, override), daemon=True, name="hylo-episode")
            self._thread.start()
            return {"task_id": task.spec.id, "policy": policy.url}

    def _run(self, task: TaskInstance, policy: PolicyEndpoint, override: str | None) -> None:
        counter = {"obs": 0, "step": 0}
        bus = EventBus()
        bus.subscribe(lambda e: self._on_event(e, counter), critical=True)
        agent = None
        try:
            if self.env is None:
                self._broadcast({"type": "status", "text": "starting the environment"})
                self.env = self._build_env()
            agent = _Instruction(self._agent(policy), override)
            runtime = self.cfg.runtime.model_copy(update={"image_every": 1})
            runner = RealtimeEpisodeRunner(self.env, agent, config=runtime, safety=self.cfg.safety, bus=bus,
                                           operator=self.operator)
            self.status["state"] = "running"
            res = runner.run(task)
            self.store.close_episode(res.outcome.episode_id)
            last = {"episode_id": res.outcome.episode_id, "status": res.extra["status"], "steps": res.outcome.steps,
                    "holds": res.holds, "chunks": res.chunks, "safety_interventions": res.safety_interventions,
                    "verdict": res.verdict, "reason": res.outcome.termination_reason}
            self._record(task, res, policy, override)
            self.status = {"state": "fault" if self.operator.fault else "idle", "episode": None, "last": last}
            self._broadcast({"type": "done", **_num(last), "run_id": self.run_dir.name})
        except Exception as exc:  # noqa: BLE001 - reported to the operator, the service stays up
            self.status = {"state": "fault" if self.operator.fault else "idle", "episode": None,
                           "last": {"error": f"{type(exc).__name__}: {exc}"[:400]}}
            self._broadcast({"type": "failed", "error": f"{type(exc).__name__}: {exc}"[:400]})
        finally:
            close = getattr(agent, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001
                    pass

    def _record(self, task: TaskInstance, res: Any, policy: PolicyEndpoint, override: str | None) -> None:
        """Keep evallog.json / manifest.json current so the episode is a normal run for replay and the catalog."""
        man = self.run_dir / "manifest.json"
        if not man.exists():
            man.write_text(json.dumps({"run_id": self.run_dir.name, "schema_version": "hylo.workcell/v0.1", "created_at": time.time(),
                                       "benchmark": {"id": f"workcell/{self.cfg.name}", "split": self.cfg.split},
                                       "config": {"name": f"workcell-{self.cfg.name}", "env": self.cfg.env.model_dump(mode="json")}},
                                      indent=2, default=str))
        logf = self.run_dir / "evallog.json"
        log = json.loads(logf.read_text()) if logf.exists() else {"version": 1, "run_id": self.run_dir.name, "samples": [],
                                                                   "benchmark": {"id": f"workcell/{self.cfg.name}", "split": self.cfg.split},
                                                                   "agent": {"name": "remote policy"}, "results": {}}
        status = res.extra["status"]
        value = {"success": 1.0, "failure": 0.0}.get(status)
        by = f"operator:{res.verdict.get('by', 'operator')}" if res.verdict else ("env" if value is not None else None)
        log["samples"].append({"key": task.key, "task_id": task.spec.id, "seed": task.seed, "episode_id": res.outcome.episode_id,
                               "status": status, "steps": res.outcome.steps, "trace": f"episodes/{res.outcome.episode_id}.jsonl",
                               "scores": {"success": {"value": value}},
                               "labels": {"success": {"value": value, "by": by,
                                                      "verdict": res.verdict.get("verdict") if res.verdict else None}},
                               "workcell": {"policy": policy.url, "protocol": policy.protocol, "instruction": override,
                                            "holds": res.holds, "chunks": res.chunks}})
        vals = [s["scores"]["success"]["value"] for s in log["samples"] if s["scores"]["success"]["value"] is not None]
        log["results"] = {"n_episodes": len(log["samples"]), "success_mean": (sum(vals) / len(vals)) if vals else None,
                          "safety_interventions_mean": None}
        logf.write_text(json.dumps(log, indent=2, default=str))

    def close(self) -> None:
        if self.env is not None:
            getattr(self.env, "close", lambda: None)()


def create_app(cfg: WorkcellConfig, *, token: str | None = None) -> FastAPI:
    cell = Workcell(cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            cell.close()

    app = FastAPI(title=f"Hylo workcell {cfg.name}", lifespan=lifespan)
    app.state.workcell = cell

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if token and request.url.path != "/healthz" and request.headers.get("authorization") != f"Bearer {token}":
            return JSONResponse({"detail": "missing or wrong token"}, status_code=401)
        return await call_next(request)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"ok": "yes"}

    @app.get("/status")
    def status() -> dict[str, Any]:
        tasks = None
        if cfg.benchmark is not None:
            try:
                bench = registry.build("benchmarks", cfg.benchmark.model_dump())
                tasks = sorted({i.spec.id: i.spec.instruction for i in bench.instances("validation")}.items())
            except Exception:  # noqa: BLE001
                tasks = None
        return {"name": cfg.name, **_num(cell.status), "fault": cell.operator.fault, "run_id": cell.run_dir.name,
                "policy": cfg.policy.model_dump() if cfg.policy else None, "split": cfg.split,
                "runtime": cfg.runtime.model_dump(mode="json"), "tasks": tasks}

    @app.post("/episodes")
    def episodes(body: dict[str, Any]) -> dict[str, Any]:
        try:
            req = StartEpisode.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(400, str(exc))
        try:
            return cell.start(req.model_dump(mode="json", exclude_none=True))
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(409, str(exc))

    @app.post("/operator")
    def operator(body: dict[str, Any]) -> dict[str, Any]:
        try:
            event = OperatorEvent.model_validate(body)
            cell.operator.post(event.model_dump(mode="json", exclude_none=True))
        except (ValidationError, ValueError) as exc:
            raise HTTPException(400, str(exc))
        if event.type == "clear_fault" and cell.status.get("state") == "fault":
            cell.status["state"] = "idle"
        if event.type == "estop":
            cell.status["state"] = "fault"
        return OperatorAck(fault=cell.operator.fault, state=cell.status.get("state")).model_dump()

    @app.websocket("/events")
    async def events(ws: WebSocket) -> None:
        if token and ws.query_params.get("token") != token:
            await ws.close(code=4401)
            return
        await ws.accept()
        q = cell.subscribe(asyncio.get_running_loop())
        try:
            while True:
                await ws.send_json(await q.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            cell.unsubscribe(q)

    return app


def serve_workcell(config: str, *, host: str = "127.0.0.1", port: int = 8100, token: str | None = None) -> int:
    import uvicorn

    cfg = WorkcellConfig.load(config)
    print(f"[workcell] {cfg.name} on http://{host}:{port} (events: ws://{host}:{port}/events)", flush=True)
    uvicorn.run(create_app(cfg, token=token), host=host, port=port, log_level="warning")
    return 0
