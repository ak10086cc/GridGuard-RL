from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from gridguard.controllers import control, critical_bus, load_baseline_config, rpei, volt_var_target
from gridguard.evaluation import aggregate, numerical_signature, signatures_match, step_record
from gridguard.network import FlowResult, GridModel
from gridguard.scenario import load_config


@pytest.fixture
def configs():
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs/ieee33.toml"), load_baseline_config(root / "configs/baselines.toml")


def test_rpei_direction_normalization_and_zero_denominators():
    # Hhat=[1,.5], SVhat=[1,.5], penalty for absorption=[1,0].
    assert rpei([2, 1], [.04, .02], [-.1, .2], -1).tolist() == pytest.approx([.5, .25])
    assert rpei([2, 1], [.04, .02], [-.1, .2], 1).tolist() == pytest.approx([1, .125])
    assert rpei([0, 0], [0, 0], [0, 0], -1).tolist() == [0, 0]


def test_mixed_voltage_rule_and_exact_bounds():
    assert critical_bus(np.array([.94, 1.08]), .95, 1.05) == (1, -1)
    assert critical_bus(np.array([.90, 1.06]), .95, 1.05) == (0, 1)
    assert critical_bus(np.array([.95, 1.05]), .95, 1.05) == (None, 0)


def test_volt_var_curve_sign_deadband_and_cap(configs):
    settings = configs[1]["volt_var"]
    raw, target = volt_var_target(np.array([.94, .98, 1., 1.02, 1.06]), np.ones(5), np.full(5, .2), settings)
    assert raw.tolist() == pytest.approx([.44, 0, 0, 0, -.44])
    assert target.tolist() == pytest.approx([.2, 0, 0, 0, -.2])


@pytest.mark.parametrize("method", ["unity_pf", "volt_var", "dsg"])
def test_real_high_pv_bounds_repeated_state_and_final_network_consistency(configs, method):
    grid, baseline = configs
    model = GridModel(grid)
    result = control(method, model, 1, model.p_rated, baseline)
    assert result.flow.converged and result.algorithm_converged
    assert np.all(np.abs(result.q_mvar) <= model.capability(model.p_rated) + 1e-12)
    assert np.allclose(model.current_q, result.q_mvar)
    assert np.array_equal(model.last_result.voltage_pu, result.flow.voltage_pu)
    if method == "unity_pf":
        assert np.array_equal(result.q_mvar, np.zeros(7))
        assert result.flow.voltage_pu.max() > 1.05
    else:
        assert result.flow.voltage_pu.max() <= 1.05
        assert np.any(result.q_mvar < 0)
    # Reusing a model at a later point cannot leak earlier probes or dispatch.
    zero = control(method, model, 1, np.zeros(7), baseline)
    assert zero.flow.converged and np.array_equal(zero.q_mvar, np.zeros(7))
    again = control(method, model, 1, model.p_rated, baseline)
    assert signatures_match(numerical_signature(result), numerical_signature(again), 1e-10)
    if method == "dsg":
        assert result.trace["ranking_evaluations"] == 1
        assert result.trace["ranking"][0]["pv_bus"] == 18


def test_dsg_feasible_point_has_no_dispatch(configs):
    grid, baseline = configs
    model = GridModel(grid)
    result = control("dsg", model, 1, model.p_rated * .5, baseline)
    assert result.status == "voltage_feasible_no_dispatch"
    assert result.pf_calls == 1
    assert np.array_equal(result.q_mvar, np.zeros(7))


class LinearGrid:
    """Small known response used to exercise search/failure branches, not feeder evidence."""
    def __init__(self, fail=None):
        self.config = {"network": {"voltage_min_pu": .95, "voltage_max_pu": 1.05, "reference_pu": 1.},
                       "pv": {"buses": [2, 3]}}
        self.bus_map = {1: 0, 2: 1, 3: 2}
        self.s_rated = np.ones(2)
        self.current_q = np.zeros(2)
        self.fail = fail
        self.commands = []

    def capability(self, p):
        return np.ones(2)

    def set_operating_point(self, load, p, q):
        self.current_q = q.copy()
        self.commands.append(q.copy())

    def solve(self):
        q = self.current_q
        if self.fail and self.fail(q):
            return FlowResult(False, None, None, None, 0., "controlled test failure")
        voltage = np.array([1., 1.08 + .06 * q[0] + .02 * q[1], 1.02 + .01 * q[0] + .04 * q[1]])
        return FlowResult(True, voltage, float(.1 - .02 * q[0] + .03 * q[1]), 0., 0.)

    def voltage_violation(self, voltage):
        return np.maximum(.95 - voltage, 0) + np.maximum(voltage - 1.05, 0)


def test_sensitivity_perturbations_are_independent_and_once(configs):
    model = LinearGrid()
    result = control("dsg", model, 1, np.ones(2) * .5, configs[1])
    assert np.array_equal(model.commands[0], [0, 0])
    assert np.allclose(model.commands[1], [.01, 0])
    assert np.allclose(model.commands[2], [0, .01])
    values = {r["pv_bus"]: r for r in result.trace["ranking"]}
    assert values[2]["sv_pu_per_mvar"] == pytest.approx(.06)
    assert values[3]["sv_pu_per_mvar"] == pytest.approx(.02)
    assert values[2]["sl_mw_per_mvar"] == pytest.approx(-.02)
    assert values[3]["sl_mw_per_mvar"] == pytest.approx(.03)
    assert result.trace["ranking_evaluations"] == 1
    assert np.max(model.voltage_violation(result.flow.voltage_pu)) == 0


def test_dsg_budget_is_bounded_and_final_state_is_accepted(configs):
    baseline = deepcopy(configs[1])
    baseline["dsg"]["max_powerflow_calls"] = 3
    model = LinearGrid()
    result = control("dsg", model, 1, np.ones(2) * .5, baseline)
    assert result.pf_calls <= 3
    assert result.status == "powerflow_budget_exhausted"
    assert not result.algorithm_converged
    assert result.flow.converged
    assert np.array_equal(result.q_mvar, model.current_q)
    assert np.array_equal(result.q_mvar, [0, 0])


@pytest.mark.parametrize("method", ["unity_pf", "volt_var", "dsg"])
def test_initial_failure_cannot_be_reported_as_safe_or_converged(configs, method):
    result = control(method, LinearGrid(fail=lambda q: True), 1, np.zeros(2), configs[1])
    assert not result.flow.converged and not result.algorithm_converged
    assert result.pf_failures == 1 and result.flow.voltage_pu is None


def test_dsg_failed_candidates_are_recorded_with_no_fake_values(configs):
    model = LinearGrid(fail=lambda q: np.any(q < -.6))
    result = control("dsg", model, 1, np.ones(2) * .5, configs[1])
    assert result.pf_failures > 0
    assert result.trace["failed_candidates"]
    assert result.flow.converged
    assert np.array_equal(result.q_mvar, model.current_q)


def test_volt_var_iteration_limit_is_distinct_from_powerflow_success(configs):
    baseline = deepcopy(configs[1])
    baseline["volt_var"]["max_iterations"] = 1
    model = GridModel(configs[0])
    result = control("volt_var", model, 1, model.p_rated, baseline)
    assert result.flow.converged
    assert not result.algorithm_converged and result.status == "iteration_limit"


def test_dsg_bisection_does_not_skip_feasible_interior_when_endpoint_overshoots(configs):
    class OvershootGrid(LinearGrid):
        def solve(self):
            result = super().solve()
            result.voltage_pu[2] = 1.01 + .08 * self.current_q[0] + .04 * self.current_q[1]
            return result
    result = control("dsg", OvershootGrid(), 1, np.ones(2) * .5, configs[1])
    assert result.status == "voltage_feasible"
    assert result.flow.voltage_pu.min() >= .95 and result.flow.voltage_pu.max() <= 1.05
    assert result.q_mvar[0] == pytest.approx(-.5, abs=1e-5)


def test_dsg_search_iteration_limit_is_not_silently_converged(configs):
    baseline = deepcopy(configs[1])
    baseline["dsg"]["max_bisection_iterations"] = 1
    result = control("dsg", LinearGrid(), 1, np.ones(2) * .5, baseline)
    assert result.flow.converged
    assert result.trace["search_incomplete"]
    assert not result.algorithm_converged


@pytest.mark.parametrize("method", ["volt_var", "dsg"])
def test_positive_injection_at_undervoltage_with_night_var_explicitly_enabled(configs, method):
    grid = deepcopy(configs[0])
    grid["pv"]["night_var_enabled"] = True
    model = GridModel(grid)
    result = control(method, model, 1, np.zeros(7), configs[1])
    assert result.flow.converged
    assert np.any(result.q_mvar > 0)
    assert result.flow.voltage_pu.min() > .9131


def test_failed_step_has_no_delta_from_previous_success(configs):
    model = GridModel(configs[0])
    result = control("unity_pf", model, 1, model.p_rated, configs[1])
    result.flow = FlowResult(False, None, None, None, 0, "test failure")
    row = step_record(model, result, "case", "unity_pf", 1, 1, 1, 1, np.zeros(7))
    assert row["mean_abs_delta_q_mvar"] is None


def test_metrics_keep_failure_denominators_and_do_not_bridge_failure(configs):
    model = GridModel(configs[0])
    result = control("unity_pf", model, 1, model.p_rated, configs[1])
    good = step_record(model, result, "case", "unity_pf", 0, 0, 1, 1, None)
    bad = dict(good, converged=False, violated_node_count=None, vmin_pu=None,
               vmax_pu=None, mean_abs_deviation_pu=None, max_abs_deviation_pu=None,
               line_loss_mw=None, algorithm_converged=False)
    report = aggregate([good, bad], 2, 15)
    assert report["valid_sample_coverage"] == .5
    assert report["voltage_node_time_denominator"] == 33
    assert report["final_step_failure_rate"] == .5
    assert report["action_transition_steps"] == 0
    assert not report["complete_episode"]
    failed = aggregate([bad], 1, 15)
    assert failed["voltage_violation_rate"] is None
    assert failed["mean_line_loss_mw"] is None and failed["line_loss_mwh_valid_steps"] is None
