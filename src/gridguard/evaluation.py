"""Common full-day baseline evaluation; failures remain visible in all denominators."""

from copy import deepcopy
import csv
import hashlib
import json
from time import perf_counter

import numpy as np

from .controllers import METHODS, control
from .network import GridModel
from .scenario import make_day


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def step_record(model, result, scenario, method, step, hour, load_scale, pv_fraction, previous_q):
    f = result.flow
    valid = f.converged
    deviations = np.abs(f.voltage_pu - model.config["network"]["reference_pu"]) if valid else None
    changed = previous_q is not None and valid
    delta = np.abs(result.q_mvar - previous_q) if changed else None
    row = {"scenario": scenario, "method": method, "step": step, "hour": hour,
           "load_scale": load_scale, "pv_fraction": pv_fraction,
           "pv_total_mw": float(model.current_p.sum()), "load_total_mw": float(model.net.load.p_mw.sum()),
           "converged": valid, "controller_status": result.status,
           "algorithm_converged": result.algorithm_converged,
           "vmin_pu": float(f.voltage_pu.min()) if valid else None,
           "vmax_pu": float(f.voltage_pu.max()) if valid else None,
           "violated_node_count": int(np.count_nonzero(model.voltage_violation(f.voltage_pu))) if valid else None,
           "mean_abs_deviation_pu": float(deviations.mean()) if valid else None,
           "max_abs_deviation_pu": float(deviations.max()) if valid else None,
           "line_loss_mw": f.loss_mw, "balance_error_mw": f.balance_error_mw,
           "mean_abs_q_mvar": float(np.mean(np.abs(result.q_mvar))),
           "total_abs_q_mvar": float(np.abs(result.q_mvar).sum()),
           "mean_abs_delta_q_mvar": float(delta.mean()) if changed else None,
           "max_abs_delta_q_mvar": float(delta.max()) if changed else None,
           "pf_calls": result.pf_calls, "pf_attempt_failures": result.pf_failures,
           "decision_s": result.decision_s, "final_pf_s": f.elapsed_s,
           "error": f.error}
    return row


def aggregate(rows, planned_steps, step_minutes):
    valid = [r for r in rows if r["converged"]]
    transitions = [r for r in rows if r["mean_abs_delta_q_mvar"] is not None]
    n = len(rows)
    successes = len(valid)
    def average(key, records):
        values = [r[key] for r in records if r[key] is not None]
        return float(np.mean(values)) if values else None
    return {"planned_steps": planned_steps, "attempted_steps": n, "successful_steps": successes,
            "final_step_failure_rate": (n - successes) / n if n else None,
            "valid_sample_coverage": successes / planned_steps,
            "complete_episode": successes == planned_steps,
            "voltage_node_time_denominator": successes * 33,
            "voltage_violation_rate": sum(r["violated_node_count"] for r in valid) / (successes * 33) if successes else None,
            "mean_abs_deviation_pu": average("mean_abs_deviation_pu", valid),
            "max_abs_deviation_pu": max(r["max_abs_deviation_pu"] for r in valid) if valid else None,
            "vmin_pu": min(r["vmin_pu"] for r in valid) if valid else None,
            "vmax_pu": max(r["vmax_pu"] for r in valid) if valid else None,
            "mean_line_loss_mw": average("line_loss_mw", valid),
            "line_loss_mwh_valid_steps": sum(r["line_loss_mw"] for r in valid) * step_minutes / 60 if valid else None,
            "mean_abs_q_mvar": average("mean_abs_q_mvar", rows),
            "mean_total_abs_q_mvar": average("total_abs_q_mvar", rows),
            "mean_abs_delta_q_mvar": average("mean_abs_delta_q_mvar", transitions),
            "action_transition_steps": len(transitions),
            "algorithm_nonconverged_steps": sum(not r["algorithm_converged"] for r in rows),
            "pf_calls": sum(r["pf_calls"] for r in rows),
            "pf_attempt_failures": sum(r["pf_attempt_failures"] for r in rows),
            "mean_decision_s": average("decision_s", rows),
            "full_horizon_no_failure_no_violation": successes == planned_steps and all(r["violated_node_count"] == 0 for r in valid)}


def numerical_signature(result):
    """Exclude timing; compare final action, voltage, loss AND algorithm/failed-candidate state."""
    return {"q_mvar": result.q_mvar.tolist(), "converged": result.flow.converged,
            "voltage": result.flow.voltage_pu.tolist() if result.flow.converged else None,
            "loss_mw": result.flow.loss_mw, "status": result.status,
            "algorithm_converged": result.algorithm_converged,
            "pf_calls": result.pf_calls, "pf_failures": result.pf_failures,
            "trace": result.trace}


def signatures_match(first, second, atol):
    def close(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
        if isinstance(a, float) and isinstance(b, float):
            return bool(np.isclose(a, b, rtol=0, atol=atol))
        return a == b
    return close(first, second)


def evaluate_baselines(grid_config, baseline_config, output):
    rows, actions, voltages, traces, summaries, snapshots = [], [], [], [], [], []
    repeat_results = []
    input_hashes = {}
    start = perf_counter()
    for scene in baseline_config["evaluation"]["scenarios"]:
        config = deepcopy(grid_config)
        for key in ("seed", "pv_cloud_min", "pv_cloud_max"):
            config["scenario"][key] = scene[key]
        day = make_day(config)
        data_hash = hashlib.sha256(b"".join(day[k].tobytes() for k in sorted(day))).hexdigest()
        input_hashes[scene["name"]] = data_hash
        for method in METHODS:
            model = GridModel(config)  # Clean network and zero-Q initialization for each method.
            method_rows, signatures = [], []
            previous_q = None
            threshold_changes = 0
            for t, hour in enumerate(day["hour"]):
                p = model.p_rated * day["pv_fraction"][t]
                result = control(method, model, float(day["load_scale"][t]), p, baseline_config)
                row = step_record(model, result, scene["name"], method, t, float(hour),
                                  float(day["load_scale"][t]), float(day["pv_fraction"][t]), previous_q)
                method_rows.append(row)
                signatures.append(numerical_signature(result))
                cap = model.capability(p)
                for j, bus in enumerate(config["pv"]["buses"]):
                    change = float(result.q_mvar[j] - previous_q[j]) if previous_q is not None and result.flow.converged else None
                    if change is not None and abs(change) > baseline_config["evaluation"]["action_change_threshold_mvar"]:
                        threshold_changes += 1
                    actions.append({"scenario": scene["name"], "method": method, "step": t,
                                    "pv_bus": bus, "p_mw": float(p[j]), "qmax_mvar": float(cap[j]),
                                    "raw_q_mvar": float(result.raw_q_mvar[j]), "executed_q_mvar": float(result.q_mvar[j]),
                                    "delta_q_mvar": change})
                if result.flow.converged:
                    previous_q = result.q_mvar.copy()
                    for bus, index in model.bus_map.items():
                        voltages.append({"scenario": scene["name"], "method": method, "step": t,
                                         "bus_number": bus, "voltage_pu": float(result.flow.voltage_pu[index])})
                else:
                    previous_q = None  # Never bridge a failed step as an adjacent transition.
                traces.append({"scenario": scene["name"], "method": method, "step": t,
                               "status": result.status, **result.trace})
            rows.extend(method_rows)
            summary = {"scenario": scene["name"], "method": method,
                       **aggregate(method_rows, len(day["hour"]), config["scenario"]["step_minutes"]),
                       "threshold_change_count": threshold_changes,
                       "input_sha256": data_hash}
            summaries.append(summary)
            # Repeat the entire day from a fresh model. This is determinism checking,
            # NOT additional statistical trials or independent random seeds.
            repeated = GridModel(config)
            identical = True
            for t in range(len(day["hour"])):
                result = control(method, repeated, float(day["load_scale"][t]),
                                 repeated.p_rated * day["pv_fraction"][t], baseline_config)
                identical &= signatures_match(signatures[t], numerical_signature(result), baseline_config["evaluation"]["repeat_atol"])
            repeat_results.append({"scenario": scene["name"], "method": method,
                                   "full_day_repeat_matches": bool(identical)})
            print(f"{scene['name']} / {method}: {summary['successful_steps']}/{summary['planned_steps']} AC solved; repeat={identical}", flush=True)

    # Small static checks isolate feasible, overvoltage and capability-limited points.
    for case, load, fraction in (("no_pv_base", 1., 0.), ("feasible_mid_pv", 1., .5),
                                 ("rated_pv", 1., 1.), ("night_stress", grid_config["verification"]["stress_load_scale"], 0.)):
        for method in METHODS:
            model = GridModel(grid_config)
            result = control(method, model, load, model.p_rated * fraction, baseline_config)
            row = step_record(model, result, case, method, 0, None, load, fraction, None)
            row["q_mvar"] = result.q_mvar.tolist()
            row["trace"] = result.trace
            snapshots.append(row)
    write_csv(output / "comparison.csv", summaries)
    write_csv(output / "step_metrics.csv", rows)
    write_csv(output / "actions.csv", actions)
    write_csv(output / "bus_voltages.csv", voltages)
    with (output / "controller_trace.jsonl").open("w", encoding="utf-8") as stream:
        for trace in traces:
            stream.write(json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n")
    plot_comparison(rows, grid_config, output)
    return {"summaries": summaries, "static_cases": snapshots, "determinism_checks": repeat_results,
            "scenario_input_sha256": input_hashes, "elapsed_with_repeat_and_output_s": perf_counter() - start,
            "evaluation_contract": {"buses_evaluated": list(range(1, 34)),
                                    "initial_q_each_step": "zero for all methods; no warm start or rate limit",
                                    "extra_shared_safety_layer": False,
                                    "failed_steps": "omit physical metrics; retain coverage/final failure/controller failure",
                                    "action_metrics": "commands across attempted steps; deltas only consecutive successful steps; exclude initial zero transition",
                                    "repeat_is_statistical_replication": False,
                                    "formal_train_validation_test_split": False}}


def plot_comparison(rows, grid_config, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scenes = list(dict.fromkeys(r["scenario"] for r in rows))
    fig, axes = plt.subplots(4, len(scenes), figsize=(7 * len(scenes), 11), squeeze=False, layout="constrained")
    for col, scene in enumerate(scenes):
        for method in METHODS:
            records = [r for r in rows if r["scenario"] == scene and r["method"] == method]
            for index, key, scale in ((0, "vmin_pu", 1), (1, "vmax_pu", 1), (2, "line_loss_mw", 1000), (3, "total_abs_q_mvar", 1)):
                axes[index, col].plot([r["hour"] for r in records],
                                      [r[key] * scale if r[key] is not None else np.nan for r in records], label=method)
        axes[0, col].set_title(scene)
        axes[0, col].axhline(grid_config["network"]["voltage_min_pu"], color="red", linestyle="--", linewidth=.8)
        axes[1, col].axhline(grid_config["network"]["voltage_max_pu"], color="red", linestyle="--", linewidth=.8)
        for index, label in enumerate(("Minimum V (p.u.)", "Maximum V (p.u.)", "Line loss (kW)", "Sum |Q| (Mvar)")):
            axes[index, col].set_ylabel(label)
            axes[index, col].grid(alpha=.25)
        axes[0, col].legend()
        axes[3, col].set_xlabel("Hour")
    fig.suptitle("Deterministic baselines | synthetic development days | same device capabilities")
    fig.savefig(output / "baseline_comparison.png", dpi=130)
    plt.close(fig)
