from mopforge.gpu import DistributedConfig, build_torchrun_command, launch_torchrun_dry_run


def test_distributed_config_and_torchrun_dry_run_command():
    config = DistributedConfig(strategy="torchrun", nproc_per_node=2, dry_run=True)
    command = build_torchrun_command("config.json", config)
    assert command[0] == "torchrun"
    assert "--nproc_per_node" in command
    payload = launch_torchrun_dry_run("config.json", config)
    assert payload["executes"] is False
    assert payload["command"][0] == "torchrun"
