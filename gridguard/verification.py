"""Physical acceptance checks and a complete uncontrolled development day."""

import csv
from time import perf_counter

import numpy as np

from .network import GridModel, action_to_q, reactive_capability
from .scenario import make_day


def require_flow(result):
    if not result.converged:
        raise RuntimeError(f"Required verification flow failed: {result.error}")
    return result


def snapshot(result) -> dict:
    return {"converged": result.converged,
            "vmin_pu": float(result.voltage_pu.min()) if result.converged else None,
            "vmax_pu": float(result.voltage_pu.max()) if result.converged else None,
            "loss_mw": result.loss_mw, "balance_error_mw": result.balance_error_mw,
            "elapsed_s": result.elapsed_s, "error": result.error}


def physical_checks(config: dict) -> dict:
    model = GridModel(config)
    zero = np.zeros_like(model.p_rated)
    model.set_operating_point(1.0, zero)
    base = require_flow(model.solve())
    if abs(base.balance_error_mw) > 1e-6:
        raise AssertionError("Base active-power balance error exceeds 1 W")
    model.set_operating_point(1.0, model.p_rated)
    pv = require_flow(model.solve())
    cap = model.capability(model.p_rated)
    v = config["verification"]
    j = config["pv"]["buses"].index(v["probe_bus"])
    i = model.bus_map[v["probe_bus"]]
    delta = cap[j] * v["probe_fraction_of_capability"]
    if delta <= 0:
        raise ValueError("Probe requires positive capability at rated active output")
    probes = {}
    for sign, name in ((1, "injection"), (-1, "absorption")):
        q = zero.copy()
        q[j] = sign * delta
        model.set_operating_point(1.0, model.p_rated, q)
        flow = require_flow(model.solve())
        probes[name] = {**snapshot(flow), "probe_bus_voltage_pu": float(flow.voltage_pu[i]),
                        "q_mvar": float(q[j])}
    if not (probes["absorption"]["probe_bus_voltage_pu"] < pv.voltage_pu[i]
            < probes["injection"]["probe_bus_voltage_pu"]):
        raise AssertionError("Reactive sign check failed at configured probe bus")

    # A finite, documented capability probe, not a DSG implementation or a safety layer.
    selected = [config["pv"]["buses"].index(b) for b in v["candidate_buses"]]
    candidates = []
    best = None
    original_violation = float(model.voltage_violation(pv.voltage_pu).sum())
    for fraction in v["candidate_fractions"]:
        q = zero.copy()
        q[selected] = -fraction * cap[selected]
        model.set_operating_point(1.0, model.p_rated, q)
        flow = model.solve()
        violation = float(model.voltage_violation(flow.voltage_pu).sum()) if flow.converged else None
        item = {"fraction": fraction, "q_mvar": q.tolist(), **snapshot(flow),
                "total_violation_pu": violation}
        candidates.append(item)
        if violation is not None and (best is None or violation < best["total_violation_pu"]):
            best = item
    if original_violation <= 0 or best is None or best["total_violation_pu"] >= original_violation:
        raise AssertionError("No demonstrated voltage-control need/improvement at rated PV")

    low_p_cap = model.capability(0.5 * model.p_rated)
    if not np.all(low_p_cap > cap):
        raise AssertionError("Capability should decrease with increasing positive active output")
    if not np.allclose(reactive_capability(model.s_rated, model.s_rated), 0):
        raise AssertionError("P=S must have zero circular capability")
    rejections = {}
    for name, operation in {
        "p_above_s": lambda: model.set_operating_point(1, model.s_rated * 1.01),
        "negative_s": lambda: reactive_capability(np.array([0.1]), np.array([-1.])),
        "nonfinite_p": lambda: model.set_operating_point(1, np.full_like(zero, np.nan)),
        "q_above_capability": lambda: model.set_operating_point(1, model.p_rated, cap * 1.01),
    }.items():
        try:
            operation()
        except ValueError:
            rejections[name] = True
        else:
            raise AssertionError(f"Invalid input accepted: {name}")
    # Exercise feasible positive/negative capability boundaries through the real solver.
    boundary = {}
    for name, action in (("positive", np.ones_like(cap)), ("negative", -np.ones_like(cap))):
        model.set_operating_point(1, model.p_rated, action_to_q(action, cap))
        boundary[name] = snapshot(require_flow(model.solve()))

    # Preserve a no-solar, no-night-VAR stress point; voltage violation != solver failure.
    model.set_operating_point(v["stress_load_scale"], zero)
    stress = require_flow(model.solve())
    stress_summary = snapshot(stress)
    stress_summary["night_capability_mvar"] = model.capability(zero).tolist()
    stress_summary["interpretation"] = "No night VAR under configured assumption; not a general infeasibility proof"
    return {"network": {"buses": len(model.net.bus),
                         "closed_lines": int(model.net.line.in_service.sum()),
                         "base_mva": model.net.sn_mva,
                         "nominal_kv": sorted(model.net.bus.vn_kv.unique().tolist()),
                         "load_mw": float(model.base_p.sum()), "load_mvar": float(model.base_q.sum()),
                         "bus_number_to_index": model.bus_map,
                         "pv_buses": config["pv"]["buses"],
                         "p_rated_mw_each": model.p_rated.tolist(), "s_rated_mva_each": model.s_rated.tolist()},
            "base_no_pv": snapshot(base), "rated_pv_zero_q": snapshot(pv),
            "reactive_probe": probes, "qmax_at_rated_p_mvar": cap.tolist(),
            "invalid_inputs_rejected": rejections, "capability_boundary": boundary,
            "candidate_probes": candidates, "best_candidate": best,
            "original_total_violation_pu": original_violation, "stress": stress_summary}


def run_day(config: dict, output) -> dict:
    model = GridModel(config)
    day = make_day(config)
    repeat = make_day(config)
    if not all(np.array_equal(day[k], repeat[k]) for k in day):
        raise AssertionError("Scenario seed is not reproducible")
    rows, voltage_rows, pf_times = [], [], []
    start = perf_counter()
    successful = 0
    violations = 0
    abs_deviations = []
    balance_errors = []
    for t, hour in enumerate(day["hour"]):
        p = model.p_rated * day["pv_fraction"][t]
        model.set_operating_point(float(day["load_scale"][t]), p)
        flow = model.solve()
        pf_times.append(flow.elapsed_s)
        row = {"step": t, "hour": float(hour), "load_scale": float(day["load_scale"][t]),
               "pv_fraction": float(day["pv_fraction"][t]), "pv_total_mw": float(p.sum()),
               "load_total_mw": float(model.net.load.p_mw.sum()),
               "instantaneous_pv_to_load_ratio": float(p.sum() / model.net.load.p_mw.sum()),
               "q_total_mvar": 0.0, **snapshot(flow)}
        if flow.converged:
            successful += 1
            violations += int(np.count_nonzero(model.voltage_violation(flow.voltage_pu) > 0))
            abs_deviations.extend(np.abs(flow.voltage_pu - config["network"]["reference_pu"]).tolist())
            balance_errors.append(abs(flow.balance_error_mw))
            for bus, index in model.bus_map.items():
                voltage_rows.append({"step": t, "bus_number": bus,
                                     "voltage_pu": float(flow.voltage_pu[index])})
        rows.append(row)
    total_s = perf_counter() - start
    for filename, data in (("day_metrics.csv", rows), ("bus_voltages.csv", voltage_rows)):
        if data:
            with (output / filename).open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
    valid = [r for r in rows if r["converged"]]
    steps = len(rows)
    summary = {"method": "zero_q_development_reference", "source": config["scenario"]["source"],
               "seed": config["scenario"]["seed"], "planned_steps": steps,
               "attempted_steps": steps, "successful_steps": successful, "failed_steps": steps - successful,
               "valid_sample_coverage": successful / steps, "final_step_failure_rate": (steps - successful) / steps,
               "voltage_evaluated_bus_numbers": list(model.bus_map),
               "voltage_node_time_denominator": successful * len(model.bus_map),
               "max_instantaneous_pv_to_load_ratio": max(r["instantaneous_pv_to_load_ratio"] for r in rows),
               "voltage_violation_rate": violations / (successful * 33) if successful else None,
               "vmin_pu": min(r["vmin_pu"] for r in valid) if valid else None,
               "vmax_pu": max(r["vmax_pu"] for r in valid) if valid else None,
               "mean_absolute_voltage_deviation_pu": float(np.mean(abs_deviations)) if successful else None,
               "max_absolute_voltage_deviation_pu": max(abs_deviations) if successful else None,
               "mean_line_loss_mw_valid_steps": float(np.mean([r["loss_mw"] for r in valid])) if successful else None,
               "line_loss_mwh_valid_steps": (sum(r["loss_mw"] for r in valid) * config["scenario"]["step_minutes"] / 60) if successful else None,
               "max_balance_error_mw": max(balance_errors) if successful else None,
               "complete_episode": successful == steps,
               "full_horizon_no_failure_no_violation": successful == steps and violations == 0,
               "pf_time_mean_s": float(np.mean(pf_times)), "pf_time_p95_s": float(np.percentile(pf_times, 95)),
               "episode_elapsed_s": total_s, "timing_scope": "set operating point + AC solves + in-memory metrics; excludes CSV/plot/imports",
               "scenario_exactly_reproducible": True}
    plot_day(config, rows, output)
    return summary


def plot_day(config, rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hour = [r["hour"] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, layout="constrained")
    axes[0].plot(hour, [r["load_scale"] for r in rows], label="Load scale")
    axes[0].plot(hour, [r["pv_fraction"] for r in rows], label="PV/nameplate")
    axes[0].set_ylabel("Input scale")
    axes[0].legend()
    for key, label in (("vmin_pu", "Minimum bus voltage"), ("vmax_pu", "Maximum bus voltage")):
        axes[1].plot(hour, [r[key] if r[key] is not None else np.nan for r in rows], label=label)
    for limit in (config["network"]["voltage_min_pu"], config["network"]["voltage_max_pu"]):
        axes[1].axhline(limit, color="red", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Voltage (p.u.)")
    axes[1].legend()
    axes[2].plot(hour, [1000 * r["loss_mw"] if r["loss_mw"] is not None else np.nan for r in rows])
    axes[2].set_ylabel("Line loss (kW)")
    axes[2].set_xlabel("Hour (synthetic day)")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle("IEEE 33-bus | zero Q | synthetic development scenario")
    fig.savefig(output / "voltage_day.png", dpi=150)
    plt.close(fig)
