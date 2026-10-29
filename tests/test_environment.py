from copy import deepcopy
from pathlib import Path

from gymnasium.utils.env_checker import check_env, data_equivalence
import numpy as np
import pytest

from gridguard.environment import GridGuardEnv, load_environment_config
from gridguard.network import FlowResult
from gridguard.scenario import load_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def short_env():
    grid = load_config(ROOT / "configs/ieee33.toml")
    grid["scenario"].update(steps=4, step_minutes=360)
    settings = load_environment_config(ROOT / "configs/environment.toml")
    settings["episode"]["max_steps"] = 4
    return GridGuardEnv(grid, settings)


def test_official_checker_and_spaces():
    env = GridGuardEnv()
    check_env(env, skip_render_check=True)
    assert env.action_space.shape == (7,) and env.observation_space.shape == (60,)
    assert env.observation_space.dtype == np.float32
    env.close()


def test_reset_has_no_residual_and_returned_arrays_do_not_alias(short_env):
    first, info = short_env.reset(seed=42)
    short_env.step(np.ones(7))
    short_env.step(np.ones(7))
    short_env.step(np.ones(7))  # Active PV at noon; last command is nonzero.
    assert np.any(short_env.previous_q != 0)
    again, other = short_env.reset(seed=42)
    assert np.array_equal(first, again) and info == other
    assert np.array_equal(short_env.previous_q, np.zeros(7))
    assert short_env.t == 0 and short_env.episode_return == 0
    again[:] = 999
    assert np.isfinite(short_env.step(np.zeros(7))[1])


@pytest.mark.parametrize("action", [np.full(7, np.nan), np.full(7, np.inf),
                                     np.full(7, 1.001), np.zeros(6), np.zeros((7, 1)), np.full(7, 1j)])
def test_invalid_action_rejected_without_mutation(short_env, action):
    short_env.reset(seed=42)
    before = short_env.model.current_q.copy()
    with pytest.raises(ValueError):
        short_env.step(action)
    assert short_env.t == 0 and short_env.elapsed_steps == 0
    assert np.array_equal(short_env.model.current_q, before)


def test_action_mapping_clock_rewards_and_terminal_semantics(short_env):
    short_env.reset(seed=42)
    for index in range(4):
        cap = short_env.current_inputs["qmax_mvar"]
        obs, reward, terminated, truncated, info = short_env.step(np.ones(7))
        assert info["step_index"] == index
        assert np.allclose(info["executed_q_mvar"], cap)
        assert reward == pytest.approx(sum(info["reward_contributions"].values()))
        assert short_env.observation_space.contains(obs)
        assert terminated == (index == 3) and not truncated
        if index < 3:
            assert short_env.current_inputs["step_index"] == index + 1
    assert info["episode_complete"] and info["termination_reason"] == "day_complete"
    assert obs[short_env.observation_slices["time"]][-1] == 0
    with pytest.raises(RuntimeError):
        short_env.step(np.zeros(7))


def test_external_limit_returns_valid_next_state_for_bootstrap(short_env):
    short_env.reset(seed=42, options={"max_steps": 2})
    short_env.step(np.zeros(7))
    obs, _, terminated, truncated, info = short_env.step(np.zeros(7))
    assert not terminated and truncated
    assert info["termination_reason"] == "external_step_limit"
    assert info["observation_valid"] and not info["episode_complete"]
    assert obs[short_env.observation_slices["time"]][-1] == .5


def test_no_future_leakage_into_current_obs_action_mapping_or_reward(short_env):
    other = GridGuardEnv(short_env.grid_config, short_env.settings)
    a, _ = short_env.reset(seed=42)
    b, _ = other.reset(seed=42)
    for _ in range(2):
        short_env.step(np.zeros(7)); other.step(np.zeros(7))
    before = short_env._last_observation.copy()
    # Change only inputs after the noon decision; no current observable may change.
    other.day["load_scale"][3] = .55
    other.day["pv_fraction"][3] = .5
    assert np.array_equal(before, other._last_observation)
    out_a = short_env.step(np.full(7, .2))
    out_b = other.step(np.full(7, .2))
    assert out_a[1] == out_b[1]
    assert out_a[4]["post_voltage_pu"] == out_b[4]["post_voltage_pu"]
    assert out_a[4]["executed_q_mvar"] == out_b[4]["executed_q_mvar"]
    assert not np.array_equal(out_a[0], out_b[0])  # Next observations can legitimately differ.


def test_same_seed_rollout_and_different_seed_scenarios(short_env):
    def rollout(seed):
        obs, _ = short_env.reset(seed=seed)
        short_env.action_space.seed(123)
        values = [obs]
        while True:
            item = short_env.step(short_env.action_space.sample())
            values.append(item)
            if item[2] or item[3]:
                break
        return values
    assert data_equivalence(rollout(42), rollout(42), exact=True)
    first = short_env.day["load_scale"].copy()
    rollout(43)
    assert not np.array_equal(first, short_env.day["load_scale"])


def test_uncontrollable_night_metrics_retained_but_costs_masked(short_env):
    short_env.reset(seed=42)
    _, reward, _, _, info = short_env.step(np.ones(7))
    assert not info["controllable"] and reward == 0
    assert info["post_voltage_pu"] is not None and info["reward_raw"]["voltage"] > 0
    assert np.array_equal(info["executed_q_mvar"], np.zeros(7))
    assert all(value == 0 for value in info["reward_contributions"].values())


def test_forced_capability_shrink_recorded_and_previous_action_in_observation(short_env):
    short_env.reset(seed=42)
    short_env.step(np.zeros(7)); short_env.step(np.zeros(7))
    obs, _, _, _, noon_info = short_env.step(np.ones(7))
    previous = np.array(noon_info["executed_q_mvar"])
    assert np.any(previous > 0)
    assert np.allclose(obs[short_env.observation_slices["previous_q_pu"]], previous / short_env.model.s_rated)
    _, _, _, _, night_info = short_env.step(np.ones(7))
    assert np.allclose(night_info["capability_adjustment_mvar"], -previous)
    assert np.array_equal(night_info["precontrol_q_mvar"], np.zeros(7))


def test_postcontrol_failure_has_no_stale_voltage_and_explicit_penalty(short_env, monkeypatch):
    short_env.reset(seed=42)
    monkeypatch.setattr(short_env.model, "solve", lambda: FlowResult(False, None, None, None, 0, "test AC failure"))
    obs, reward, term, trunc, info = short_env.step(np.zeros(7))
    assert term and not trunc and reward == -100
    assert info["post_voltage_pu"] is None and info["line_loss_mw"] is None
    assert not info["observation_valid"] and info["failure_stage"] == "postcontrol"
    assert info["reward_raw"]["loss"] is None
    assert np.array_equal(obs, np.zeros(60))  # Invalid absorbing FEATURES, not measured zero voltage.
    with pytest.raises(RuntimeError):
        short_env.step(np.zeros(7))


def test_next_precontrol_failure_preserves_current_reward_and_metrics(short_env, monkeypatch):
    short_env.reset(seed=42)
    original = short_env.model.solve
    calls = 0
    def solve():
        nonlocal calls
        calls += 1
        return original() if calls == 1 else FlowResult(False, None, None, None, 0, "next-state failure")
    monkeypatch.setattr(short_env.model, "solve", solve)
    obs, reward, term, trunc, info = short_env.step(np.zeros(7))
    assert term and not trunc and reward == 0
    assert info["postcontrol_converged"] and info["post_voltage_pu"] is not None
    assert info["next_precontrol_failed"] and info["failed_step_index"] == 1
    assert not info["observation_valid"] and obs[-1] == 0


def test_initial_failure_raises_and_valid_reset_recovers(short_env, monkeypatch):
    from gridguard.network import GridModel
    original = GridModel.solve
    monkeypatch.setattr(GridModel, "solve", lambda self: FlowResult(False, None, None, None, 0, "reset failure"))
    with pytest.raises(RuntimeError, match="Initial pre-control"):
        short_env.reset(seed=42)
    with pytest.raises(RuntimeError):
        short_env.step(np.zeros(7))
    monkeypatch.setattr(GridModel, "solve", original)
    assert short_env.reset(seed=42)[0][-1] == 1


def test_invalid_reward_config_is_rejected(short_env):
    config = deepcopy(short_env.settings)
    config["reward"]["voltage_weight"] = float("nan")
    with pytest.raises(ValueError):
        GridGuardEnv(short_env.grid_config, config)
