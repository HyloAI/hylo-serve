# contract versions and what is known about these plugins (worldcurrent.core.plugins)
META = {
    "contracts": {"envs": "wc.env/v0.1", "agents": "wc.agent/v0.1"},
    "maturity": "implemented",
    "plugins": {
        "envs/lerobot.robot": {
            "maturity": "needs-hardware",
            "notes": "runs with the simulated arm (sim_arm); no real arm connected yet",
        },
    },
}
