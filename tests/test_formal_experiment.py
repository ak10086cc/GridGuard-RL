from copy import deepcopy
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
from stable_baselines3 import SAC
import torch

from gridguard.controllers import load_baseline_config
from gridguard.environment import load_environment_config
from gridguard.formal_experiment import summarize_test, train_seed
from gridguard.learning import actor_digest, load_sac_config
from gridguard.policy_evaluation import evaluate_policies
from gridguard.scenario import load_config

ROOT = Path(__file__).resolve().parents[1]


def small_configuration():
    grid = load_config(ROOT / "configs/ieee33.toml")
    grid["scenario"].update(steps=4, step_minutes=360)
    settings = load_environment_config(ROOT / "configs/environment.toml")
    settings["episode"]["max_steps"] = 4
    baselines = load_baseline_config(ROOT / "configs/baselines.toml")
    config = load_sac_config(ROOT / "configs/sac_formal.toml", formal=True)
    config["run"].update(total_timesteps=64, validation_interval=32)
    config["sac"].update(learning_starts=8, buffer_size=128, batch_size=8, net_arch=[16, 16], log_every_updates=8)
    config["formal"].update(checkpoint_interval=16, progress_interval=8)
    config["scenarios"]["validation_seeds"] = [2001]
    return grid, settings, baselines, config


def test_atomic_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    grid, settings, baselines, config = small_configuration()
    provenance = {"source_sha256": {"test": "fixed"}, "dependencies": {"torch": torch.__version__}}
    full = train_seed(grid, settings, baselines, config, tmp_path / "full", provenance)
    paused = train_seed(grid, settings, baselines, config, tmp_path / "resumed", provenance, pause_at=16)
    assert paused["status"] == "paused_at_checkpoint" and paused["timesteps"] == 16
    resumed = train_seed(grid, settings, baselines, config, tmp_path / "resumed", provenance, resume=True)
    assert full["timesteps"] == resumed["timesteps"] == 64
    assert full["gradient_updates"] == resumed["gradient_updates"] == 56
    assert full["best_validation_return"] == resumed["best_validation_return"]
    models = []
    for label in ("full", "resumed"):
        with zipfile.ZipFile(tmp_path / label / "last_checkpoint.zip") as archive:
            manifest = json.loads(archive.read("checkpoint.json"))
            assert manifest["replay_size"] == 64
            path = tmp_path / f"{label}_model.zip"
            path.write_bytes(archive.read("model.zip"))
        models.append(SAC.load(path, device="cpu"))
    assert actor_digest(models[0]) == actor_digest(models[1])
    assert all(torch.equal(v, models[1].critic.state_dict()[k]) for k, v in models[0].critic.state_dict().items())
    with pytest.raises(ValueError, match="different inputs"):
        bad = deepcopy(config); bad["sac"]["gamma"] = .9
        train_seed(grid, settings, baselines, bad, tmp_path / "resumed", provenance, resume=True)


def test_formal_mode_keeps_smoke_budget_guard():
    path = ROOT / "configs/sac_formal.toml"
    assert load_sac_config(path, formal=True)["run"]["total_timesteps"] == 50000
    with pytest.raises(ValueError):
        load_sac_config(path)


def test_seed_aggregation_does_not_replicate_deterministic_baselines(tmp_path):
    grid, settings, baselines, _ = small_configuration()
    class ZeroPolicy:
        def predict(self, obs, deterministic):
            return np.zeros(7, dtype=np.float32), None
    deterministic = evaluate_policies(grid, settings, baselines, [9001], ["unity_pf", "volt_var", "dsg"], retain_raw=True)
    results = [("deterministic", deterministic)]
    for seed in (7, 17, 27):
        results.append((str(seed), evaluate_policies(grid, settings, baselines, [9001], ["sac"], model=ZeroPolicy(), retain_raw=True)))
    report = summarize_test(results, tmp_path)
    assert len(report["per_seed"]) == 6
    sac = next(x for x in report["aggregates"] if x["method"] == "sac")
    assert sac["training_seed_count"] == 3 and sac["voltage_violation_rate_seed_std"] == pytest.approx(0, abs=1e-15)
    baseline = report["aggregates"][0]
    assert baseline["training_seed_count"] is None and baseline["voltage_violation_rate_seed_std"] is None
