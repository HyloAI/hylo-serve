"""Policy server (S1): any worldcurrent agent as a network service, for robots and for other harnesses.

One WebSocket server, two protocols, chosen by path:
- `/wc`      the worldcurrent agent protocol (wc.agentserver/v0.1, JSON): spec / reset / act / on_step /
             on_episode_end. `reset` carries the client's EnvSpec and the session is refused there, before any
             motion, if the agent is not compatible with it (core.compat). Client: core.remote.ws.WSAgent.
- `/openpi`  openpi's protocol (msgpack-numpy): the server sends its metadata, then answers every observation dict
             with {"actions": (H, A), "server_timing": {...}}, so openpi clients (DROID, RoboArena, openpi's own
             examples) can run our agents. openpi has no reset message: the agent is reset on connect, when the
             prompt changes, or when the client sends {"reset": true}.
- `/healthz` liveness, `/meta` the served agent's spec as JSON.

One session at a time (a robot talks to one policy; a second client is refused with code 1013, like LeRobot's
admission). The agent is built once and reused across sessions. With `campaign`, the server follows that
campaign's active version through its active.json (core.deployment): between sessions, never during one, it
rebuilds the agent when a promotion or rollback changed it.

    wc serve policy --agent '{"type": "lerobot.steered", ...}'      # or --config EXP.yaml / --campaign DIR
"""

from __future__ import annotations

import hashlib
import hmac
import http
import json
import sys
import threading
import time
import uuid
from typing import Any, Callable

import numpy as np

from worldcurrent.core.agent import AgentSpec, EpisodeContext
from worldcurrent.core.compat import check_compatibility
from worldcurrent.core.config import ExperimentConfig
from worldcurrent.core.deployment import read_active
from worldcurrent.core.registry import registry
from worldcurrent.core.remote import wire
from worldcurrent.core.remote.handlers import AgentHandler
from worldcurrent.core.task import TaskInstance, TaskSpec
from worldcurrent.core.types import EnvSpec, Observation

# openpi observation key -> our observation field: "camera:<name>", "state:<key>=<n>,<key>=<n>" (split a flat
# vector into named state entries), or "instruction". The inverse of the openpi client's presets.
OPENPI_PRESETS: dict[str, dict[str, str]] = {
    "libero": {"observation/image": "camera:agentview", "observation/wrist_image": "camera:wrist",
               "observation/state": "state:eef_pos=3,eef_axis_angle=3,gripper_qpos=2", "prompt": "instruction"},
}


def observation_from_openpi(obs: dict[str, Any], mapping: dict[str, str]) -> Observation:
    state, images, instruction = {}, {}, None
    for key, src in mapping.items():
        if key not in obs:
            continue
        kind, _, spec = src.partition(":")
        v = obs[key]
        if kind == "camera":
            images[spec] = np.asarray(v, dtype=np.uint8)
        elif kind == "state":
            vec, i = np.asarray(v, dtype=np.float32).reshape(-1), 0
            for part in spec.split(","):
                name, _, n = part.partition("=")
                n = int(n or len(vec) - i)
                state[name] = vec[i:i + n]
                i += n
        elif src == "instruction":
            instruction = v.decode() if isinstance(v, bytes) else str(v)
    return Observation(state=state, images=images, instruction=instruction)


class _Agent:
    """The served agent, built from a spec, an experiment config or a campaign's active version."""

    def __init__(self, agent: dict[str, Any] | None = None, config: str | None = None, campaign: str | None = None) -> None:
        if sum(x is not None for x in (agent, config, campaign)) != 1:
            raise ValueError("give exactly one of agent, config, campaign")
        self.agent_spec, self.config, self.campaign = agent, config, campaign
        self.agent: Any = None
        self.version: str | None = None
        self.refresh()

    def _resolve(self) -> tuple[dict[str, Any], str]:
        if self.agent_spec is not None:
            return self.agent_spec, "static"
        if self.config is not None:
            return ExperimentConfig.load(self.config).agent.model_dump(mode="json"), "static"
        active = read_active(self.campaign)               # written by the campaign on every change (core.deployment)
        return dict(active.agent), active.version or "base"

    def refresh(self) -> bool:
        """(Re)build the agent if its definition changed. Returns True when it did."""
        spec, version = self._resolve()
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()[:16]
        if self.agent is not None and digest == getattr(self, "_digest", None):
            return False
        self.close()
        self.agent, self._digest, self.version = registry.build("agents", spec), digest, version
        return True

    def close(self) -> None:
        if self.agent is not None:
            fn = getattr(self.agent, "close", None)
            if callable(fn):
                fn()
            self.agent = None


class _CheckedHandler(AgentHandler):
    """The /wc protocol, refusing an incompatible environment at reset (before the first action)."""

    def handle(self, msg: dict[str, Any]) -> dict[str, Any]:
        if msg.get("method") == "reset":
            report = check_compatibility(self.agent.spec(), EnvSpec.model_validate(msg["args"]["env_spec"]))
            if not report.ok:
                return {"id": msg["id"], "ok": False, "error": "incompatible environment: " + "; ".join(report.errors),
                        "events": []}
        return super().handle(msg)

    def close(self) -> None:                          # the served agent outlives the session
        self.closed = True


class PolicyServer:
    def __init__(self, *, agent: dict[str, Any] | None = None, config: str | None = None, campaign: str | None = None,
                 host: str = "127.0.0.1", port: int = 8000, token: str | None = None,
                 openpi_inputs: dict[str, str] | str = "libero", openpi_max_steps: int = 10_000,
                 log: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True)) -> None:
        self.served = _Agent(agent, config, campaign)
        self.host, self.port, self.token, self.log = host, port, token, log
        self.openpi_inputs = OPENPI_PRESETS[openpi_inputs] if isinstance(openpi_inputs, str) else dict(openpi_inputs)
        self.openpi_max_steps = openpi_max_steps
        self._session = threading.Lock()
        self._server = None
        self.sessions = 0

    # -- HTTP side ------------------------------------------------------------------------------
    def _process_request(self, connection, request):
        path = request.path.split("?")[0]
        if path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        if self.token is not None:
            got = request.headers.get("Authorization", "")
            if not (hmac.compare_digest(got, f"Bearer {self.token}") or hmac.compare_digest(got, f"Api-Key {self.token}")):
                return connection.respond(http.HTTPStatus.UNAUTHORIZED, "unauthorized\n")
        if path == "/meta":
            return connection.respond(http.HTTPStatus.OK, json.dumps(self.metadata(), default=str) + "\n")
        if path not in ("/wc", "/openpi", "/"):
            return connection.respond(http.HTTPStatus.NOT_FOUND, "paths: /wc, /openpi, /meta, /healthz\n")
        return None

    def metadata(self) -> dict[str, Any]:
        spec: AgentSpec = self.served.agent.spec()
        return {"server": "worldcurrent.policy@0.1", "agent": spec.model_dump(mode="json"), "version": self.served.version,
                "protocols": ["wc.agentserver/v0.1", "openpi"], "openpi_inputs": self.openpi_inputs,
                "action_dim": spec.action_dim}

    # -- sessions ------------------------------------------------------------------------------------
    def _handle(self, conn) -> None:
        if not self._session.acquire(blocking=False):
            conn.close(code=1013, reason="busy: one session at a time")
            return
        try:
            if self.served.refresh():
                self.log(f"[policy] serving version {self.served.version}")
            self.sessions += 1
            path = conn.request.path.split("?")[0]
            self.log(f"[policy] session from {conn.remote_address} on {path}")
            (self._openpi if path == "/openpi" else self._wc)(conn)
        except Exception as exc:  # noqa: BLE001 - one bad session must not take the server down
            self.log(f"[policy] session dropped: {exc!r}")
        finally:
            self._session.release()

    def _wc(self, conn) -> None:
        handler = _CheckedHandler(self.served.agent)
        conn.send(json.dumps({"hello": handler.protocol, "transport": "ws", "version": self.served.version}))
        for frame in conn:
            conn.send(json.dumps(handler.handle(json.loads(frame))))
            if handler.closed:
                break

    def _openpi(self, conn) -> None:
        agent = self.served.agent
        conn.send(wire.packb(self.metadata()))
        prompt, steps = None, 0

        def reset(instruction: str) -> None:
            task = TaskInstance(spec=TaskSpec(id="openpi/session", version="1", instruction=instruction,
                                              max_steps=self.openpi_max_steps), seed=0)
            agent.reset(EpisodeContext(episode_id=f"openpi-{uuid.uuid4().hex[:8]}", task=task, env_spec=None,
                                       emit=lambda t, p: None, seed=0))

        for frame in conn:
            t0 = time.perf_counter()
            try:
                obs_dict = wire.unpackb(frame)
                obs = observation_from_openpi(obs_dict, self.openpi_inputs)
                if prompt is None or obs.instruction != prompt or obs_dict.get("reset"):
                    prompt, steps = obs.instruction, 0
                    reset(prompt or "")
                chunk = agent.act(obs)
                steps += int(chunk.execution_steps or len(chunk.actions))
                conn.send(wire.packb({"actions": np.asarray(chunk.actions, dtype=np.float32),
                                      "server_timing": {"infer_ms": (time.perf_counter() - t0) * 1000},
                                      "execution_steps": int(chunk.execution_steps or len(chunk.actions))}))
            except Exception as exc:  # noqa: BLE001 - openpi reports errors as a text frame
                conn.send(f"policy error: {type(exc).__name__}: {exc}")

    # -- lifecycle ------------------------------------------------------------------------------------
    def start(self) -> "PolicyServer":
        import websockets.sync.server as wss

        self._server = wss.serve(self._handle, self.host, self.port, max_size=None, compression=None,
                                 process_request=self._process_request)
        self.port = self._server.socket.getsockname()[1]
        threading.Thread(target=self._server.serve_forever, daemon=True, name="wc-policy-server").start()
        self.log(f"[policy] ws://{self.host}:{self.port}/wc and /openpi (version {self.served.version})")
        return self

    def serve_forever(self) -> None:
        self.start()
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()
        self.served.close()
