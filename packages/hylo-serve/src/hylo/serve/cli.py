"""hylo serve policy | workcell | env | agent: start a serving role (policy.py, workcell.py) or one component.

Registered as the `serve` subcommand of `hylo` (core.cli). `env` / `agent` serve a single plugin over WebSocket
(core.remote.ws), e.g. a simulator in its own venv.
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence


def main(argv: Sequence[str]) -> int:
    """serve a policy, a workcell, or one env / agent component"""
    if argv and argv[0] in ("env", "agent"):
        from hylo.core.remote import ws

        return ws.main(list(argv))
    ap = argparse.ArgumentParser(prog="hylo serve", epilog="also: hylo serve env|agent ... serves one environment or agent "
                                                       "plugin over WebSocket (hylo serve env --help)")
    sub = ap.add_subparsers(dest="role", required=True)
    p = sub.add_parser("policy", help="serve an agent over /wc and /openpi")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--agent", help="agent spec as JSON: {\"type\": ..., \"params\": {...}}")
    src.add_argument("--config", help="experiment config whose agent to serve")
    src.add_argument("--campaign", help="campaign directory: serve its active version, following promotions")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--token", help="require 'Authorization: Bearer <token>' (or openpi's 'Api-Key <token>')")
    p.add_argument("--openpi-inputs", default="libero", help="preset name or JSON mapping of openpi keys to observation fields")
    w = sub.add_parser("workcell", help="run episodes next to a robot / simulator with a remote policy")
    w.add_argument("--config", required=True, help="workcell config (YAML or JSON), see hylo.serve.workcell")
    w.add_argument("--host", default="127.0.0.1")
    w.add_argument("--port", type=int, default=8100)
    w.add_argument("--token")
    a = ap.parse_args(argv)
    if a.role == "policy":
        from .policy import PolicyServer

        inputs = a.openpi_inputs if not a.openpi_inputs.startswith("{") else json.loads(a.openpi_inputs)
        PolicyServer(agent=json.loads(a.agent) if a.agent else None, config=a.config, campaign=a.campaign, host=a.host,
                     port=a.port, token=a.token, openpi_inputs=inputs).serve_forever()
        return 0
    from .workcell import serve_workcell

    return serve_workcell(a.config, host=a.host, port=a.port, token=a.token)
