from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from stable_baselines3 import SAC

from gridguard.controllers import load_baseline_config
from gridguard.environment import load_environment_config
from gridguard.learning import ScenarioPoolEnv, estimate_budget, load_sac_config, scenario_manifest
from gridguard.policy_evaluation import evaluate_policies
from gridguard.scenario import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def configs():
    return (load_config(ROOT / "configs/ieee33.toml"), load_environment_config(ROOT / "configs/environment.toml"),
            load_baseline_config(ROOT / "configs/baselines.toml"), load_sac_config(ROOT / "configs/sac.toml"))


def test_whole_day_splits_have_unique_content_and_do_not_mutate_source(configs):
    grid, _, _, sac = configs
    original = deepcopy(grid)
    derived, manifest = scenario_manifest(grid, sac)
    assert grid == original
    assert derived["scenario"]["pv_cloud_min"] == .35
    assert [len(manifest[k]) for k in ("train", "validation", "test")] == [32, 2, 4]
    hashes = [x["whole_day_sha256"] for entries in manifest.values() for x in entries]
    assert len(hashes) == len(set(hashes))
    assert all(x["steps"] == 96 for entries in manifest.values() for x in entries)


def test_different_seed_labels_cannot_hide_duplicate_data(configs):
    grid, _, _, sac = configs
    grid["scenario"]["load_noise_std"] = 0
    sac["scenarios"].update(pv_cloud_min=1., pv_cloud_max=1.)
    with pytest.raises(ValueError, match="Duplicate whole-day"):
        scenario_manifest(grid, sac)


@pytest.mark.parametrize("old,new", [("validation_seeds = [2001, 2002]", "validation_seeds = [1001, 2002]"),
                                      ("test_seeds = [3001, 3002, 3003, 3004]", "test_seeds = [42, 3002, 3003, 3004]"),
                                      ("total_timesteps = 768", "total_timesteps = 100000"),
                                      ("total_timesteps = 768", "total_timesteps = 96"),
                                      ("batch_size = 64", "batch_size = 8192")])
def test_invalid_or_leaking_configuration_is_rejected(tmp_path, old, new):
    path = tmp_path / "invalid_sac.toml"
    path.write_text((ROOT / "configs/sac.toml").read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError):
        load_sac_config(path)


def test_training_pool_cannot_reset_into_validation_or_test(configs):
    grid, settings, _, sac = configs
    a = ScenarioPoolEnv(grid, settings, sac["scenarios"]["train_seeds"], 7)
    b = ScenarioPoolEnv(grid, settings, sac["scenarios"]["train_seeds"], 7)
    seq_a, seq_b = [], []
    for i in range(3):
        seq_a.append(a.reset(seed=7 if i == 0 else None)[1]["scenario_seed"])
        seq_b.append(b.reset(seed=7 if i == 0 else None)[1]["scenario_seed"])
    assert seq_a == seq_b and len(set(seq_a)) == 3
    assert set(seq_a).issubset(sac["scenarios"]["train_seeds"])
    with pytest.raises(ValueError):
        a.reset(options={"scenario_seed": 3001})
    a.close(); b.close()


def test_shared_sac_evaluator_uses_physical_actions_and_failure_metrics(configs, monkeypatch):
    grid, settings, baselines, _ = configs
    grid["scenario"].update(steps=4, step_minutes=360)
    settings["episode"]["max_steps"] = 4
    class ZeroPolicy:
        def predict(self, obs, deterministic):
            assert deterministic and obs.shape == (60,)
            return np.zeros(7, dtype=np.float32), None
    result = evaluate_policies(grid, settings, baselines, [2001], ["sac", "unity_pf"], ZeroPolicy())
    a, b = result["summaries"]
    assert a["episode_complete"] and b["episode_complete"]
    assert a["voltage_violation_rate"] == b["voltage_violation_rate"]
    assert a["mean_line_loss_mw"] == b["mean_line_loss_mw"]
    assert a["mean_abs_q_mvar"] == 0 and a["voltage_node_time_denominator"] == 132


def test_standard_sb3_model_save_load_preserves_deterministic_actions(configs, tmp_path):
    # No learning here; actual gradient updates are checked by the authorized smoke run.
    grid, settings, _, sac = configs
    env = ScenarioPoolEnv(grid, settings, sac["scenarios"]["train_seeds"], 7)
    model = SAC("MlpPolicy", env, device="cpu", seed=7, buffer_size=128, policy_kwargs={"net_arch": [16, 16]})
    obs, _ = env.reset(seed=7)
    expected, _ = model.predict(obs, deterministic=True)
    model.save(tmp_path / "model.zip")
    loaded = SAC.load(tmp_path / "model.zip", device="cpu")
    actual, _ = loaded.predict(obs, deterministic=True)
    assert np.array_equal(actual, expected)
    env.close()


def test_budget_is_extrapolation_not_an_execution_schedule(configs):
    timing = {"collection_and_other_s_per_step": .04, "s_per_gradient_update": .005,
              "mean_validation_round_s": 8., "validation_comparison_s": 60.}
    result = estimate_budget(configs[3], timing)
    assert result["seeds"] == [7, 17, 27] and result["timesteps_per_seed"] == 50000
    assert not result["executed"]
    assert result["estimated_training_total_s"] > 3 * 50000 * .04
