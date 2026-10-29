from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from gridguard.controllers import load_baseline_config
from gridguard.enhanced import EnhancedEnv, load_enhanced_config, screen_candidates
from gridguard.environment import load_environment_config
from gridguard.network import FlowResult
from gridguard.scenario import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    return (load_config(ROOT/'configs/ieee33.toml'), load_environment_config(ROOT/'configs/environment.toml'),
            load_baseline_config(ROOT/'configs/baselines.toml'), load_enhanced_config(ROOT/'configs/enhanced.toml'))


def test_residual_mapping_device_capability_and_reset(inputs):
    env = EnhancedEnv(*inputs, shield=False)
    env.reset(seed=42)
    # Midday has nonzero capability; move the environment through the day causally.
    for _ in range(40):
        _, _, done, truncated, _ = env.step(np.zeros(env.n_pv))
        assert not (done or truncated)
    cap = env.cap.copy()
    assert np.any(cap > 0)
    action = np.ones(env.n_pv, dtype=np.float32)
    _, _, _, _, info = env.step(action)
    base = np.asarray(info['dsg_q_mvar'])
    expected = np.clip(base + .25 * cap, -cap, cap)
    np.testing.assert_allclose(info['raw_q_mvar'], base + .25 * cap, atol=1e-12)
    np.testing.assert_allclose(info['proposed_q_mvar'], expected, atol=1e-12)
    assert info['capability_limited'] == bool(np.any(np.abs(base + .25 * cap - expected) > 1e-12))
    np.testing.assert_allclose(info['executed_q_mvar'], expected, atol=1e-12)
    assert np.all(np.abs(info['executed_q_mvar']) <= cap + 1e-12)
    first, _ = env.reset(seed=42)
    assert np.all(env.previous_q == 0)
    again, _ = env.reset(seed=42)
    np.testing.assert_array_equal(first, again)
    env.close()


def test_screen_rechecks_fallback_and_has_bounded_failure(inputs, monkeypatch):
    env = EnhancedEnv(*inputs, shield=True)
    env.reset(seed=42)
    calls = []
    actual = env.model.solve
    def sometimes_failed():
        calls.append(env.model.current_q.copy())
        if len(calls) == 1:
            return FlowResult(False, None, None, None, 0., 'injected_failure')
        return actual()
    monkeypatch.setattr(env.model, 'solve', sometimes_failed)
    state = env.current_inputs
    result = screen_candidates(env.model, state, env.pre_flow,
        [('proposal', np.zeros(env.n_pv)), ('previous', env.previous_q.copy())], env.enhanced['control'])
    assert len(calls) == 2 and result['candidate'] == 'previous' and result['pf_failures'] == 1
    assert result['attempts'][0]['error'] == 'injected_failure'
    assert result['attempts'][1]['converged']
    monkeypatch.setattr(env.model, 'solve', lambda: FlowResult(False, None, None, None, 0., 'injected_failure'))
    result = screen_candidates(env.model, state, env.pre_flow,
        [(str(i), np.zeros(env.n_pv)) for i in range(8)], env.enhanced['control'])
    assert result['q'] is None and len(result['attempts']) == 5 and result['pf_failures'] == 5
    env.close()


def test_all_candidates_rejected_is_explicit_failure(inputs, monkeypatch):
    env = EnhancedEnv(*inputs, mode='direct', shield=True)
    env.reset(seed=42)
    monkeypatch.setattr(env.model, 'solve', lambda: FlowResult(False, None, None, None, 0., 'forced_failure'))
    observation, reward, terminated, truncated, info = env.step(np.zeros(env.n_pv))
    assert terminated and not truncated and reward == -100
    assert info['failure_reason'] == 'all_candidates_rejected'
    assert info['executed_q_mvar'] is None and info['post_voltage_pu'] is None
    assert info['screen_pf_calls'] == info['screen_pf_failures'] == 5
    assert observation[-1] == 0
    env.close()


def test_existing_violation_is_not_called_voltage_feasible(inputs):
    env = EnhancedEnv(*inputs, mode='direct', shield=True)
    env.reset(seed=42)
    assert np.max(env.model.voltage_violation(env.pre_flow.voltage_pu)) > 0
    _, _, _, _, info = env.step(np.zeros(env.n_pv))
    assert info['screen_voltage_feasible'] is False
    assert info['intervention_reason'] == 'preexisting_voltage_violation'
    env.close()


def test_residual_decision_does_not_use_future_inputs(inputs):
    grid, settings, baselines, enhanced = deepcopy(inputs)
    grid['scenario'].update(steps=4, step_minutes=360)
    settings['episode']['max_steps'] = 4
    envs = [EnhancedEnv(grid, settings, baselines, enhanced, shield=True) for _ in range(2)]
    for env in envs:
        env.reset(seed=42)
        env.t = 2  # Same actual midday inputs, with active PV.
        assert env._prepare(2) is None
    envs[1].day['load_scale'][3] = grid['scenario']['load_min']
    a = np.full(7, .3, dtype=np.float32)
    first = envs[0].step(a)
    second = envs[1].step(a)
    assert first[1] == pytest.approx(second[1], abs=1e-12)
    for key in ('dsg_q_mvar', 'raw_q_mvar', 'executed_q_mvar', 'post_voltage_pu'):
        np.testing.assert_allclose(first[4][key], second[4][key], atol=1e-12, rtol=0)
    assert first[4]['candidate'] == second[4]['candidate']
    for env in envs: env.close()
