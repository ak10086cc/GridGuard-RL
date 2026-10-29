from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from pandapower.powerflow import LoadflowNotConverged

from gridguard.network import GridModel, action_to_q, reactive_capability
from gridguard.scenario import load_config, make_day
from gridguard.verification import physical_checks, run_day


@pytest.fixture
def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/ieee33.toml")


def test_network_mapping_units_and_reference(config):
    model = GridModel(config)
    assert len(model.bus_map) == 33
    assert model.bus_map[1] == 0 and model.bus_map[33] == 32
    assert model.net.line.in_service.sum() == 32
    assert np.allclose(model.net.bus.vn_kv, 12.66)
    assert model.net.sn_mva == 10
    assert model.base_p.sum() == pytest.approx(3.715)
    assert model.base_q.sum() == pytest.approx(2.3)
    model.set_operating_point(1, np.zeros(7))
    flow = model.solve()
    assert flow.converged
    assert flow.voltage_pu.min() == pytest.approx(0.91309, abs=2e-5)
    assert flow.loss_mw == pytest.approx(0.202677, abs=2e-6)
    assert abs(flow.balance_error_mw) < 1e-6


@pytest.mark.parametrize("p,s", [([1.1], [1]), ([-.1], [1]), ([0], [0]),
                                   ([np.nan], [1]), ([0], [np.inf]), ([0, 0], [1])])
def test_capability_rejects_invalid_values(p, s):
    with pytest.raises(ValueError):
        reactive_capability(p, s)


def test_capability_and_action_boundaries():
    assert reactive_capability([0, .8, 1], [1, 1, 1]).tolist() == pytest.approx([1, .6, 0])
    assert reactive_capability([0], [1], night_var_enabled=False)[0] == 0
    assert action_to_q([-1, 0, 1], [.6, .2, .6]).tolist() == [-.6, 0, .6]
    for action in ([1.001], [np.nan], [0, 1]):
        with pytest.raises(ValueError):
            action_to_q(action, [.6])


def test_rejected_setpoint_is_atomic_and_success_invalidates_results(config):
    model = GridModel(config)
    model.set_operating_point(1, model.p_rated)
    assert model.solve().converged
    old_p = model.net.load.p_mw.copy()
    with pytest.raises(ValueError):
        model.set_operating_point(.5, model.s_rated * 1.01)
    assert model.net.load.p_mw.equals(old_p)
    assert model.last_result.converged
    model.set_operating_point(.5, model.p_rated * .5)
    assert model.last_result is None and not model.net.converged
    assert model.net.res_bus.empty or model.net.res_bus.vm_pu.isna().all()


def test_actual_powerflow_failure_does_not_reuse_previous_results(config):
    model = GridModel(config)
    model.set_operating_point(1, model.p_rated)
    assert model.solve().converged
    model.config = deepcopy(config)
    model.config["network"]["max_iteration"] = 1
    result = model.solve()
    assert not result.converged and result.voltage_pu is None and result.loss_mw is None
    assert result.error and not model.net.converged
    assert model.net.res_bus.empty or model.net.res_bus.vm_pu.isna().all()


def test_synthetic_day_reproducibility_and_night(config):
    day = make_day(config)
    assert len(day["hour"]) == 96
    assert day["hour"][-1] == 23.75
    assert all(np.array_equal(day[k], make_day(config)[k]) for k in day)
    altered = deepcopy(config)
    altered["scenario"]["seed"] += 1
    assert not np.array_equal(day["pv_fraction"], make_day(altered)["pv_fraction"])
    assert np.all(day["pv_fraction"][(day["hour"] <= 6) | (day["hour"] >= 18)] == 0)
    assert np.all((day["pv_fraction"] >= 0) & (day["pv_fraction"] <= 1))
    assert np.ptp(day["load_scale"]) > 0


@pytest.mark.parametrize("old,new", [("inverter_s_to_p_ratio = 1.2", "inverter_s_to_p_ratio = 0.9"),
                                      ("steps = 96", "steps = 95"),
                                      ("buses = [6, 14, 18, 22, 25, 30, 33]", "buses = [6, 6]")])
def test_config_rejects_inconsistent_inputs(tmp_path, old, new):
    text = (Path(__file__).resolve().parents[1] / "configs/ieee33.toml").read_text(encoding="utf-8")
    path = tmp_path / "invalid.toml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_physical_sign_and_candidate_control(config):
    report = physical_checks(config)
    assert report["rated_pv_zero_q"]["vmax_pu"] > 1.05
    assert report["best_candidate"]["total_violation_pu"] < report["original_total_violation_pu"]
    assert all(report["invalid_inputs_rejected"].values())
    assert report["stress"]["vmin_pu"] < .95


def test_failed_day_has_no_fabricated_voltage_and_zero_coverage(config, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise LoadflowNotConverged("injected test failure")
    monkeypatch.setattr("gridguard.network.pp.runpp", fail)
    report = run_day(config, tmp_path)
    assert report["failed_steps"] == 96
    assert report["valid_sample_coverage"] == 0
    assert report["vmin_pu"] is None and report["voltage_violation_rate"] is None
    assert not report["complete_episode"]
    assert not (tmp_path / "bus_voltages.csv").exists()
