"""Real-time episodes next to a robot: the core kernel in realtime mode (core.kernel.Timing).

The loop itself (aligned chunk merging, hold action, watchdog, operator channel, virtual / wall clock) lives in
core.kernel, so a benchmark can measure exactly what a workcell runs (`timing: {mode: realtime, clock: virtual}`
in an experiment config). This module keeps the workcell's configuration (`runtime:` in a workcell config) and
the result shape the workcell reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

from pydantic import Field

from worldcurrent.core.events import EventBus
from worldcurrent.core.kernel import ABSOLUTE_MODES, EpisodeRunner, OperatorChannel, Timing, hold_action
from worldcurrent.core.task import TaskInstance
from worldcurrent.core.trace import TraceStore


class RealtimeConfig(Timing):
    """Timing in realtime mode, plus what the workcell records."""

    mode: Literal["sync", "realtime"] = "realtime"
    image_every: int = Field(default=1, ge=0)
    record_privileged: bool = True

    def timing(self) -> Timing:
        return Timing.model_validate(self.model_dump(exclude={"image_every", "record_privileged"}))


@dataclass
class RealtimeResult:
    outcome: Any
    holds: int
    chunks: int
    safety_interventions: int
    verdict: dict[str, Any] | None = None
    events: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


class RealtimeEpisodeRunner:
    def __init__(self, env: Any, agent: Any, *, config: RealtimeConfig | None = None,
                 safety: Mapping[str, Any] | None = None, trace: TraceStore | None = None, bus: EventBus | None = None,
                 operator: OperatorChannel | None = None) -> None:
        self.cfg = config or RealtimeConfig()
        self.operator = operator or OperatorChannel()
        self.runner = EpisodeRunner(env, agent, trace=trace, bus=bus, safety=safety, timing=self.cfg.timing(),
                                    operator=self.operator, image_every=self.cfg.image_every,
                                    record_privileged=self.cfg.record_privileged)
        self.bus, self.hz, self.spec = self.runner.bus, self.runner.hz, self.runner.env_spec

    def run(self, task: TaskInstance, *, episode_id: str | None = None) -> RealtimeResult:
        r = self.runner.run(task, episode_id=episode_id)
        return RealtimeResult(r.outcome, r.holds, r.chunks, r.safety_interventions, r.verdict, r.events, {"status": r.status})


__all__ = ["ABSOLUTE_MODES", "OperatorChannel", "RealtimeConfig", "RealtimeEpisodeRunner", "RealtimeResult", "hold_action"]
