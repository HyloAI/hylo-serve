# contract versions and what is known about these plugins (hylo.core.plugins)
META = {
    "contracts": {"envs": "hylo.env/v0.1", "agents": "hylo.agent/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "envs/lerobot.robot": {
            "maturity": "needs-hardware",
            "notes": "runs with the simulated arm (sim_arm); no real arm connected yet",
        },
    },
}
