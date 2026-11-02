# Examples

Runnable examples of the runtime layer: `hylo serve policy` and `hylo serve workcell` started as real subprocesses
on free ports, driven through the workcell API (hylo.workcell/v0.1) the way the playground's Operator page drives
them, and measured from the recorded traces. Every workcell episode is an ordinary run that `hylo trace` reads.

```bash
uv run python examples/01_sim_arm_end_to_end.py [--quick] [--out DIR]    # from the serve repository
```

Each script writes `result.json` and prints a results table; the default output is `runs/examples/<slug>/`.
`--quick` takes under 30 s. The simulated arm comes from the deployments repository
(`workcells/sim_arm.yaml`: 30 Hz wall clock, watchdog after 60 holds, reset confirmation, operator verdict), found
through `$HYLO_DEPLOYMENTS` or the side-by-side `deployments-template` checkout.

## Results

Measured on 2026-11-01 (128-core CPU, no GPU), full mode.

| Example | What it shows | Command | Measured result | Runtime |
|---|---|---|---|---|
| `01_sim_arm_end_to_end.py` | Policy server and workcell on the simulated arm, 5 operated episodes (reset confirmed, verdict given), control-loop timing from the traces | `uv run python examples/01_sim_arm_end_to_end.py` | 1500 steps at 30.00 Hz (target 30); period jitter 0.32 ms std, worst p99 deviation 2.75 ms, longest interval 36.4 ms, 0 deadline misses (interval > 1.5 periods); policy round trip p50 3.6 ms, p99 8.7 ms; 1 hold per episode (step 0, before the first chunk) | 52 s (`--quick` 11 s) |
| `02_watchdog_and_hold.py` | A fault-injecting proxy between workcell and policy: 100 ms latency, a 1 s stall, a 3 s stall, a dropped connection | `uv run python examples/02_watchdog_and_hold.py` | 100 ms latency: 4 stale actions dropped per chunk, episode completes. 1 s stall: 29 holds (streak 28), the arm stays still (0.00 deg/step while holding, 1.05 deg/step while moving), episode completes. 3 s stall: first hold 162 ms after the fault, watchdog abort after 61 holds, 2162 ms after the fault. Dropped connection: episode ends with an error 30 ms after the drop; all five episodes are in the run | 38 s (`--quick` 23 s) |
| `03_operator_channel.py` | E-stop mid-episode (10 trials), the latch, reset confirmation, the operator verdict | `uv run python examples/03_operator_channel.py` | 0 actions after the e-stop in 9 of 10 trials, 1 in the other (1.5 ms after the press, already in the loop); abort in the trace at most 34.7 ms after the press (one period is 33.3 ms). New episodes refused with 409 at once and still after 3 s, until `clear_fault`. 0 actions during 2 s of `awaiting_reset`, first action 12 ms after `reset_done`. Verdict "partial" recorded as label `{verdict: partial, by: operator:operator}` | 22 s (`--quick` 11 s) |
| `04_campaign_hot_switch.py` | `hylo serve policy --campaign`: a manual approval and a rollback reach the robot between sessions, never during one | `uv run python examples/04_campaign_hot_switch.py` | Held-out MuJoCo planar reach through the policy server: base (kp 0.02) 0.42 success, 144 steps; a 200-episode session during which the candidate was approved kept serving the base (0.415); after the approval the candidate (kp 0.04) 1.00, 81.5 steps; after `hylo evolve rollback` the base again, identical episode for episode (0.42) | 31 s (`--quick` 20 s) |

## Notes

- Example 01's numbers depend on machine load: the full run above shared the machine with other jobs; a quick
  run on a quiet machine measured 0.05 ms jitter std and a 2.8 ms round trip p50.
- In example 02, after the 3 s stall the watchdog aborts at about 2.2 s, but `episode.end` is written only when the
  late reply arrives (3 s): the runtime waits for an in-flight policy call before closing the episode. The arm is
  stopped at the abort.
- A dropped policy connection ends the episode with status `error`, not `safety_abort`, so the env's `stop()` is
  not called; with no new commands the arm keeps its last target.
- In example 03, a "partial" verdict counts as a failure (success value 0) with the verdict kept in the label.
- Example 02 found a bug, fixed in `packages/hylo-serve/src/hylo/serve/workcell.py` (`_Instruction.on_episode_end`):
  when the policy connection was lost, the closing `on_episode_end` call raised after `episode.end` had been
  written, so the episode was missing from the run's evallog.json and the operator saw "failed". The error is now
  kept in the status (`last.policy_error`) and the episode is recorded.
