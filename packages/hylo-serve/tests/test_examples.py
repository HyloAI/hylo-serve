"""Smoke tests: every script in examples/ runs with --quick and writes its result.json."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[3] / "examples"
SCRIPTS = sorted(p.name for p in EXAMPLES.glob("[0-9][0-9]_*.py"))


def test_the_examples_are_found():
    assert len(SCRIPTS) >= 3


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_runs_quick(script, tmp_path):
    res = subprocess.run([sys.executable, str(EXAMPLES / script), "--quick", "--out", str(tmp_path)], cwd=EXAMPLES.parent,
                         capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stdout[-2000:] + res.stderr[-3000:]
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["quick"] is True
    if script.startswith("01_"):
        s = result["summary"]
        assert s["episodes"] == 2 and s["steps_total"] == 300 and s["verdicts"] == ["success", "success"]
    elif script.startswith("02_"):
        by = {r["scenario"]: r for r in result["scenarios"]}
        assert by["stall 1.0 s"]["status"] == "success" and by["stall 1.0 s"]["holds"] > 20
        assert by["stall 3.0 s"]["status"] == "error" and "watchdog" in by["stall 3.0 s"]["reason"]
        assert by["dropped connection"]["status"] == "error"
    elif script.startswith("03_"):
        s = result["summary"]
        assert s["actions_after_estop_max"] <= 1 and s["latched"]["refused_after_3s"] == 409
        assert s["reset_confirmation"]["actions_before_reset_done"] == 0
        assert s["verdict"]["label"]["verdict"] == "partial"
    elif script.startswith("04_"):
        by = {r["phase"]: r for r in result["phases"]}
        assert by["approve during session"]["served_version"] == "base"
        assert by["after approval"]["served_version"] == result["candidate"]
        assert by["after rollback"]["served_version"] == "base" and result["rollback_restores_base_exactly"]
