# hylo-serve

Deployment: what runs next to a robot and on the GPU box that serves its policy. By default its packages depend on
`hylo-core` only, so a robot computer installs nothing else.

| Package | Purpose |
|---|---|
| `packages/hylo-serve` | `hylo serve policy`: an agent behind the Hylo protocol (`/hylo`) and openpi's (`/openpi`), one session at a time, optionally following a campaign's `active.json`. `hylo serve workcell`: episodes on a robot or simulator with a remote policy, the SafetyShield, a real-time runner (aligned chunk merging, hold, watchdog) and an operator channel (e-stop latched until cleared, reset confirmation, verdicts), recorded as ordinary runs. `hylo serve env / agent`: one component over WebSocket. |
| `adapters/hylo-env-lerobot-robot` | real robots through LeRobot's `Robot` interface (SO-100/101, Koch, ...), a simulated arm, a bring-up wave policy |

```bash
uv run hylo serve policy --agent '{"type": "bringup.wave", "params": {"amplitude": 12, "horizon": 15}}' --port 8002
uv run hylo serve workcell --config ../deployments-template/workcells/sim_arm.yaml --port 8102   # from a deployments checkout: workcells/sim_arm.yaml
uv run hylo serve policy --campaign ../bench/runs/evolve/<campaign> --port 8001                   # serve a campaign's active version
```

Interfaces (defined in hylo-core, so callers never import this repository): the workcell API hylo.workcell/v0.1
(`core.remote.workcell`), the policy protocols (`core.remote`, `core.remote.wire`) and a campaign's active version
(`core.deployment`). Architecture and the real-robot path: hylo-docs, docs/design/architecture.md §3.8.

Documentation (hylo-docs, `uv run mkdocs serve -f docs/mkdocs.yml` from the workspace): [deployment](../docs/docs/user-guide/deployment.md), [safety](../docs/docs/user-guide/safety.md), [deployments repository](../docs/docs/configuration/deployments-repo.md).

## License

Apache License 2.0 (see [LICENSE](LICENSE)). Copyright 2026 Hylo AI.
