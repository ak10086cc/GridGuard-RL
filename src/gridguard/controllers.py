"""Deterministic baselines: paper-based DSG with explicitly documented conventions."""

from dataclasses import dataclass
from time import perf_counter
import tomllib

import numpy as np

from .network import FlowResult, GridModel


def load_baseline_config(path):
    with open(path, "rb") as stream:
        c = tomllib.load(stream)
    vv, d, e = (c[k] for k in ("volt_var", "dsg", "evaluation"))
    points = np.asarray(vv["voltage_points_pu"], dtype=float)
    if points.shape != (4,) or not np.isfinite(points).all() or np.any(np.diff(points) <= 0):
        raise ValueError("Volt-VAR requires four finite, increasing voltage points")
    positive = [vv[k] for k in ("qmax_to_s_ratio", "relaxation", "voltage_tolerance_pu",
                                "q_tolerance_mvar", "fixed_point_tolerance_mvar")]
    positive += [d[k] for k in ("perturbation_fraction", "voltage_tolerance_pu", "q_tolerance_mvar")]
    positive += [e["repeat_atol"], e["action_change_threshold_mvar"]]
    if not np.isfinite(positive).all() or np.any(np.asarray(positive) <= 0):
        raise ValueError("Baseline tolerances/scales must be positive and finite")
    if vv["relaxation"] > 1 or vv["qmax_to_s_ratio"] > 1 or d["perturbation_fraction"] > 1:
        raise ValueError("Relaxation and capability ratios must be <= 1")
    for section, keys in ((vv, ["max_iterations"]),
                          (d, ["max_bisection_iterations", "max_backtracks", "max_powerflow_calls"])):
        if any(type(section[k]) is not int or section[k] < 1 for k in keys):
            raise ValueError("Iteration budgets must be positive integers")
    if d["max_powerflow_calls"] < 2:
        raise ValueError("DSG needs at least an initial and a final AC evaluation")
    if d["mixed_violation_rule"] != "largest_limit_exceedance_overvoltage_on_tie":
        raise ValueError("Unsupported mixed-voltage rule")
    names = []
    for scenario in e["scenarios"]:
        names.append(scenario["name"])
        if type(scenario["seed"]) is not int or scenario["seed"] < 0:
            raise ValueError("Scenario seed must be a nonnegative integer")
        if not 0 <= scenario["pv_cloud_min"] <= scenario["pv_cloud_max"] <= 1:
            raise ValueError("Invalid scenario cloud range")
    if not names or len(set(names)) != len(names):
        raise ValueError("Scenario names must be nonempty and unique")
    return c


def critical_bus(voltage, vmin, vmax):
    """Paper II-B; mixed violations use the project's explicit priority rule."""
    over, under = float(np.max(voltage) - vmax), float(vmin - np.min(voltage))
    if max(over, under) <= 0:
        return None, 0
    return (int(np.argmax(voltage)), -1) if over >= under else (int(np.argmin(voltage)), 1)


def rpei(headroom, sv, sl, direction):
    """Paper (6): Hhat * |SV|hat / (1 + max(d*SL,0)hat)."""
    def normalize(values):
        values = np.asarray(values, dtype=float)
        maximum = float(np.max(values))
        return values / maximum if maximum > 0 else np.zeros_like(values)
    return normalize(np.maximum(headroom, 0)) * normalize(np.abs(sv)) / (
        1 + normalize(np.maximum(direction * np.asarray(sl), 0)))


def volt_var_target(voltage, s_rated, capability, settings):
    raw = np.interp(voltage, settings["voltage_points_pu"], [1., 0., 0., -1.])
    raw = raw * settings["qmax_to_s_ratio"] * s_rated
    return raw, np.clip(raw, -capability, capability)


@dataclass
class ControlResult:
    raw_q_mvar: np.ndarray
    q_mvar: np.ndarray
    flow: FlowResult
    status: str
    algorithm_converged: bool
    pf_calls: int
    pf_failures: int
    decision_s: float
    trace: dict


class BudgetExceeded(RuntimeError):
    pass


class OperatingPoint:
    """Shared device constraints and counted AC evaluations for every baseline."""
    def __init__(self, model, load_scale, p_mw, budget):
        self.model, self.load, self.p = model, load_scale, np.asarray(p_mw, dtype=float).copy()
        self.cap = model.capability(self.p)
        self.calls = self.failures = 0
        self.budget = budget
        self.failed_candidates = []
        self.start = perf_counter()

    def evaluate(self, q):
        if self.calls >= self.budget:
            raise BudgetExceeded("AC power-flow call budget exhausted")
        raw = np.asarray(q, dtype=float)
        if raw.shape != self.cap.shape or not np.isfinite(raw).all():
            raise ValueError("Invalid controller command")
        actual = np.clip(raw, -self.cap, self.cap)
        self.model.set_operating_point(self.load, self.p, actual)
        flow = self.model.solve()
        self.calls += 1
        if not flow.converged:
            self.failures += 1
            self.failed_candidates.append({"q_mvar": actual.tolist(), "error": flow.error})
        return actual.copy(), flow

    def finish(self, raw, q, flow, status, converged, trace):
        trace["failed_candidates"] = self.failed_candidates
        return ControlResult(np.asarray(raw).copy(), q.copy(), flow, status, converged,
                             self.calls, self.failures, perf_counter() - self.start, trace)


def unity_pf(op):
    raw = np.zeros_like(op.cap)
    q, flow = op.evaluate(raw)
    return op.finish(raw, q, flow, "zero_q" if flow.converged else "powerflow_failed",
                     flow.converged, {})


def volt_var(op, config):
    s = config["volt_var"]
    q, flow = op.evaluate(np.zeros_like(op.cap))
    trace = {"iterations": 0}
    if not flow.converged:
        return op.finish(q, q, flow, "initial_powerflow_failed", False, trace)
    pv_indices = [op.model.bus_map[b] for b in op.model.config["pv"]["buses"]]
    for iteration in range(1, s["max_iterations"] + 1):
        old_q, old_flow = q.copy(), flow
        raw_target, target = volt_var_target(flow.voltage_pu[pv_indices], op.model.s_rated, op.cap, s)
        command = (1 - s["relaxation"]) * q + s["relaxation"] * target
        q, flow = op.evaluate(command)
        trace.update(iterations=iteration, last_raw_target_mvar=raw_target.tolist())
        if not flow.converged:
            # Explicitly reevaluate the last accepted command at THIS operating point.
            q, flow = op.evaluate(old_q)
            return op.finish(old_q, q, flow, "iteration_powerflow_failed", False, trace)
        _, fixed_target = volt_var_target(flow.voltage_pu[pv_indices], op.model.s_rated, op.cap, s)
        fixed_error = float(np.max(np.abs(q - fixed_target)))
        trace["fixed_point_error_mvar"] = fixed_error
        if (np.max(np.abs(flow.voltage_pu - old_flow.voltage_pu)) <= s["voltage_tolerance_pu"]
                and np.max(np.abs(q - old_q)) <= s["q_tolerance_mvar"]
                and fixed_error <= s["fixed_point_tolerance_mvar"]):
            return op.finish(command, q, flow, "fixed_point_converged", True, trace)
    return op.finish(q, q, flow, "iteration_limit", False, trace)


def dsg(op, config):
    s = config["dsg"]
    q, base = op.evaluate(np.zeros_like(op.cap))
    trace = {"ranking": [], "dispatch": [], "ranking_evaluations": 0}
    if not base.converged:
        return op.finish(q, q, base, "initial_powerflow_failed", False, trace)
    n = op.model.config["network"]
    vmin, vmax = n["voltage_min_pu"], n["voltage_max_pu"]
    bus, direction = critical_bus(base.voltage_pu, vmin, vmax)
    if bus is None:
        return op.finish(q, q, base, "voltage_feasible_no_dispatch", True, trace)
    if not np.any(op.cap > 0):
        return op.finish(q, q, base, "no_reactive_capability", True, trace)
    trace.update(critical_bus=bus + 1, direction=direction, ranking_evaluations=1)
    sv, sl = np.zeros_like(q), np.zeros_like(q)
    eligible = np.zeros(q.size, dtype=bool)
    accepted_q, accepted_flow = q.copy(), base
    search_incomplete = False
    sensitivity_complete = True

    def severity(flow):
        violation = op.model.voltage_violation(flow.voltage_pu)
        return float(np.max(violation)), float(violation.sum())

    def gap(flow):
        return (float(np.max(flow.voltage_pu)) - vmax) if direction < 0 else (vmin - float(np.min(flow.voltage_pu)))

    def non_worsening(flow, previous):
        a, b = severity(flow), severity(previous)
        return (a[0] <= b[0] + s["voltage_tolerance_pu"]
                and a[1] <= b[1] + s["voltage_tolerance_pu"]
                and gap(flow) < gap(previous))

    op.budget -= 1  # Reserve final execution within, rather than beyond, the configured budget.
    try:
        # All perturbations start from exactly the SAME zero-Q operating point.
        for j in range(q.size):
            delta = s["perturbation_fraction"] * op.cap[j]
            if delta <= 0:
                continue
            probe = np.zeros_like(q)
            probe[j] = delta
            _, flow = op.evaluate(probe)
            if flow.converged:
                sv[j] = (flow.voltage_pu[bus] - base.voltage_pu[bus]) / delta
                sl[j] = (flow.loss_mw - base.loss_mw) / delta
                eligible[j] = sv[j] > 0
            else:
                sensitivity_complete = False
        scores = rpei(op.cap, sv, sl, direction)
        order = np.argsort(-scores, kind="stable")
        trace["ranking"] = [{"pv_bus": op.model.config["pv"]["buses"][j], "sv_pu_per_mvar": float(sv[j]),
                             "sl_mw_per_mvar": float(sl[j]), "headroom_mvar": float(op.cap[j]),
                             "rpei": float(scores[j]), "eligible": bool(eligible[j])} for j in order]
        for j in order:
            if not eligible[j] or scores[j] <= 0:
                continue
            previous_q, previous_flow = accepted_q.copy(), accepted_flow
            endpoint = previous_q.copy()
            endpoint[j] = direction * op.cap[j]
            endpoint_q, endpoint_flow = op.evaluate(endpoint)
            candidate_q, candidate_flow = endpoint_q, endpoint_flow
            bisections = 0
            bracket_width = None
            # Bisect the all-bus limit in the original correction direction;
            # the RPEI critical bus/ranking remains fixed, even if the limiting bus shifts.
            # A full-capability endpoint may overshoot the opposite limit; this does
            # not justify skipping an intermediate feasible command.
            if endpoint_flow.converged and gap(endpoint_flow) <= -s["voltage_tolerance_pu"]:
                lo, hi = 0., float(op.cap[j])
                for _ in range(s["max_bisection_iterations"]):
                    if hi - lo <= s["q_tolerance_mvar"]:
                        break
                    mid = (lo + hi) / 2
                    trial = previous_q.copy()
                    trial[j] = direction * mid
                    trial_q, trial_flow = op.evaluate(trial)
                    bisections += 1
                    if not trial_flow.converged:
                        # A failed solve establishes no voltage bracket.
                        trace["dispatch"].append({"pv_bus": op.model.config["pv"]["buses"][j],
                                                  "event": "bisection_powerflow_failed"})
                        search_incomplete = True
                        break
                    if gap(trial_flow) <= -s["voltage_tolerance_pu"]:
                        hi = mid
                        candidate_q, candidate_flow = trial_q, trial_flow
                    else:
                        lo = mid
                bracket_width = hi - lo
                search_incomplete |= bracket_width > s["q_tolerance_mvar"]
            if (candidate_flow.converged
                    and non_worsening(candidate_flow, previous_flow)):
                accepted_q, accepted_flow = candidate_q.copy(), candidate_flow
            else:
                # Project convention for paper's undefined "non-worsening":
                # bounded halving, accepting only lower critical voltage error
                # without increasing max OR total all-bus violation.
                for k in range(1, s["max_backtracks"] + 1):
                    trial = previous_q.copy()
                    trial[j] = direction * op.cap[j] / (2 ** k)
                    trial_q, trial_flow = op.evaluate(trial)
                    if trial_flow.converged and non_worsening(trial_flow, previous_flow):
                        accepted_q, accepted_flow = trial_q.copy(), trial_flow
                        break
            trace["dispatch"].append({"pv_bus": op.model.config["pv"]["buses"][j],
                                      "q_mvar": float(accepted_q[j]), "bisections": bisections,
                                      "bracket_width_mvar": bracket_width,
                                      "retained": bool(not np.array_equal(previous_q, accepted_q))})
            if severity(accepted_flow)[0] == 0:
                break
        status = "voltage_feasible" if severity(accepted_flow)[0] == 0 else "residual_voltage_violation"
    except BudgetExceeded:
        status = "powerflow_budget_exhausted"
    # Restore the reserved final-execution slot within the total configured budget.
    # It too is counted; no stale last-probe results are returned.
    op.budget += 1
    final_q, final_flow = op.evaluate(accepted_q)
    if not final_flow.converged:
        status = "final_powerflow_failed"
    trace["search_incomplete"] = bool(search_incomplete)
    trace["sensitivity_complete"] = sensitivity_complete
    algorithm_ok = (final_flow.converged and status != "powerflow_budget_exhausted"
                    and not search_incomplete and sensitivity_complete)
    return op.finish(accepted_q, final_q, final_flow, status, algorithm_ok, trace)


METHODS = ("unity_pf", "volt_var", "dsg")


def control(method: str, model: GridModel, load_scale, p_mw, config) -> ControlResult:
    if method not in METHODS:
        raise ValueError(f"Unknown baseline: {method}")
    budget = config["dsg"]["max_powerflow_calls"] if method == "dsg" else config["volt_var"]["max_iterations"] + 2
    op = OperatingPoint(model, load_scale, p_mw, budget)
    if method == "unity_pf":
        return unity_pf(op)
    return volt_var(op, config) if method == "volt_var" else dsg(op, config)
