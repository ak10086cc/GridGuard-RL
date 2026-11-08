"""Balanced AC snapshots; MW/Mvar, sgen positive Q means injection."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandapower as pp
from pandapower.networks import case33bw
from pandapower.powerflow import LoadflowNotConverged


def reactive_capability(p_mw, s_mva, *, night_var_enabled=True) -> np.ndarray:
    p, s = np.asarray(p_mw, dtype=float), np.asarray(s_mva, dtype=float)
    if p.shape != s.shape or not np.isfinite(p).all() or not np.isfinite(s).all():
        raise ValueError("P and S must have equal shapes and finite values")
    if np.any(s <= 0) or np.any(p < 0) or np.any(p > s):
        raise ValueError("Require S > 0 and 0 <= P <= S; active curtailment is not implemented")
    # Multiplication form avoids unnecessary cancellation near P=S.
    q = np.sqrt((s - p) * (s + p))
    return q if night_var_enabled else np.where(p > 0, q, 0.0)


def action_to_q(action, capability) -> np.ndarray:
    action, capability = np.asarray(action, dtype=float), np.asarray(capability, dtype=float)
    if (action.shape != capability.shape or not np.isfinite(action).all()
            or not np.isfinite(capability).all() or np.any(capability < 0)
            or np.any(np.abs(action) > 1)):
        raise ValueError("Actions must be finite, match capability and lie in [-1,1]")
    return action * capability


@dataclass
class FlowResult:
    converged: bool
    voltage_pu: np.ndarray | None
    loss_mw: float | None
    balance_error_mw: float | None
    elapsed_s: float
    error: str | None = None


class GridModel:
    def __init__(self, config: dict):
        self.config = config
        self.net = case33bw()
        # The bundled case preserves the MATPOWER bus order: 1..33 -> 0..32.
        if not np.array_equal(self.net.bus.index.to_numpy(), np.arange(33)):
            raise ValueError("Unexpected case33bw bus mapping; review before use")
        self.bus_map = {number: number - 1 for number in range(1, 34)}
        self.net.sn_mva = config["network"]["sn_mva"]
        self.net.bus.loc[:, "min_vm_pu"] = config["network"]["voltage_min_pu"]
        self.net.bus.loc[:, "max_vm_pu"] = config["network"]["voltage_max_pu"]
        self.base_p = self.net.load.p_mw.to_numpy(copy=True)
        self.base_q = self.net.load.q_mvar.to_numpy(copy=True)
        p = config["pv"]
        count = len(p["buses"])
        self.p_rated = np.full(count, self.base_p.sum() * p["installed_to_base_load_ratio"] / count)
        self.s_rated = self.p_rated * p["inverter_s_to_p_ratio"]
        self.pv_indices = [pp.create_sgen(self.net, bus=self.bus_map[b], p_mw=0,
                                        q_mvar=0, sn_mva=float(self.s_rated[j]), name=f"PV_bus_{b}")
                           for j, b in enumerate(p["buses"])]
        self.current_p = np.zeros(count)
        self.current_q = np.zeros(count)
        self.last_result: FlowResult | None = None

    def capability(self, p_mw) -> np.ndarray:
        return reactive_capability(p_mw, self.s_rated,
                                   night_var_enabled=self.config["pv"]["night_var_enabled"])

    def set_operating_point(self, load_scale: float, p_mw, q_mvar=None):
        p = np.asarray(p_mw, dtype=float)
        q = np.zeros_like(self.p_rated) if q_mvar is None else np.asarray(q_mvar, dtype=float)
        if not np.isfinite(load_scale) or load_scale < 0 or p.shape != self.p_rated.shape:
            raise ValueError("Invalid load scale or PV dimension")
        cap = self.capability(p)
        if q.shape != p.shape or not np.isfinite(q).all() or np.any(np.abs(q) > cap + 1e-12):
            raise ValueError("Q exceeds current inverter capability or has invalid values")
        # Validate before mutating: rejected inputs never partially update the grid.
        self.net.load.loc[:, "p_mw"] = self.base_p * load_scale
        self.net.load.loc[:, "q_mvar"] = self.base_q * load_scale
        self.net.sgen.loc[self.pv_indices, "p_mw"] = p
        self.net.sgen.loc[self.pv_indices, "q_mvar"] = q
        self.current_p, self.current_q = p.copy(), q.copy()
        self.last_result = None
        self.net.converged = False
        pp.reset_results(self.net)

    def solve(self) -> FlowResult:
        n = self.config["network"]
        start = perf_counter()
        # Never expose stale result tables following a failed calculation.
        self.last_result = None
        self.net.converged = False
        pp.reset_results(self.net)
        try:
            pp.runpp(self.net, algorithm=n["algorithm"], init="flat", numba=False,
                     calculate_voltage_angles=True, max_iteration=n["max_iteration"],
                     tolerance_mva=n["tolerance_mva"], check_connectivity=True)
            voltage = self.net.res_bus.vm_pu.to_numpy(copy=True)
            losses = self.net.res_line.pl_mw.to_numpy()
            balance = (self.net.res_ext_grid.p_mw.sum() + self.net.res_sgen.p_mw.sum()
                       - self.net.res_load.p_mw.sum() - losses.sum())
            if (not self.net.converged or not np.isfinite(voltage).all()
                    or not np.isfinite(losses).all() or not np.isfinite(balance)):
                raise FloatingPointError("Non-finite or unsolved AC power flow")
            result = FlowResult(True, voltage, float(losses.sum()), float(balance), perf_counter() - start)
        except (LoadflowNotConverged, FloatingPointError, np.linalg.LinAlgError) as exc:
            self.net.converged = False
            pp.reset_results(self.net)
            result = FlowResult(False, None, None, None, perf_counter() - start, str(exc))
        self.last_result = result
        return result

    def voltage_violation(self, voltage) -> np.ndarray:
        n = self.config["network"]
        v = np.asarray(voltage)
        return np.maximum(n["voltage_min_pu"] - v, 0) + np.maximum(v - n["voltage_max_pu"], 0)
