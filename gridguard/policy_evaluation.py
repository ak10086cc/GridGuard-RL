"""One environment-based metric pipeline for SAC and the deterministic baselines."""

import hashlib
import json
from time import perf_counter

import numpy as np

from .controllers import control
from .environment import GridGuardEnv
from .evaluation import aggregate, write_csv


def evaluate_policies(grid_config, environment_config, baseline_config, seeds, methods, model=None, output=None, *, retain_raw=False):
    summaries, rows, commands, signatures = [], [], [], {}
    for seed in seeds:
        for method in methods:
            env = GridGuardEnv(grid_config, environment_config)
            obs, _ = env.reset(seed=seed)
            episode_rows, episode_reward, previous_q = [], 0., None
            digest = hashlib.sha256()
            threshold_changes = 0
            while True:
                decision_start = perf_counter()
                inputs = env.current_inputs
                internal_calls = internal_failures = 0
                controller_ok = True
                compute_start = perf_counter()
                if method == "sac":
                    if model is None:
                        raise ValueError("SAC evaluation requires a model")
                    action, _ = model.predict(obs, deterministic=True)
                    controller_status = "sac_deterministic"
                else:
                    chosen = control(method, env.model, inputs["load_scale"], inputs["p_mw"], baseline_config)
                    internal_calls, internal_failures = chosen.pf_calls, chosen.pf_failures
                    controller_ok, controller_status = chosen.algorithm_converged, chosen.status
                    action = np.divide(chosen.q_mvar, inputs["qmax_mvar"], out=np.zeros(env.n_pv), where=inputs["qmax_mvar"] > 0)
                    action = np.clip(action, -1, 1).astype(np.float32)
                compute_s = perf_counter() - compute_start
                current_obs = obs.copy()
                obs, reward, terminated, truncated, info = env.step(action)
                decision_s = perf_counter() - decision_start
                valid = info["postcontrol_converged"]
                q = np.asarray(info["executed_q_mvar"])
                voltage = np.asarray(info["post_voltage_pu"]) if valid else None
                deviation = np.abs(voltage - grid_config["network"]["reference_pu"]) if valid else None
                delta = np.abs(q - previous_q) if previous_q is not None and valid else None
                if delta is not None:
                    threshold_changes += int(np.count_nonzero(delta > baseline_config["evaluation"]["action_change_threshold_mvar"]))
                # Includes the baseline's internal search, current post-control solve,
                # and next-state pre-control solve (if attempted). Reset solve is separate.
                next_pre_attempted = info["termination_reason"] not in ("day_complete", "postcontrol_failure")
                row = {"scenario": f"seed_{seed}", "method": method, "step": info["step_index"], "hour": info["hour"],
                       "load_scale": info["load_scale"], "pv_total_mw": info["pv_total_mw"],
                       "converged": valid, "controller_status": controller_status,
                       "algorithm_converged": controller_ok and valid,
                       "vmin_pu": float(voltage.min()) if valid else None, "vmax_pu": float(voltage.max()) if valid else None,
                       "violated_node_count": int(np.count_nonzero(env.model.voltage_violation(voltage))) if valid else None,
                       "mean_abs_deviation_pu": float(deviation.mean()) if valid else None,
                       "max_abs_deviation_pu": float(deviation.max()) if valid else None,
                       "line_loss_mw": info["line_loss_mw"], "balance_error_mw": info["balance_error_mw"],
                       "mean_abs_q_mvar": float(np.abs(q).mean()), "total_abs_q_mvar": float(np.abs(q).sum()),
                       "mean_abs_delta_q_mvar": float(delta.mean()) if delta is not None else None,
                       "max_abs_delta_q_mvar": float(delta.max()) if delta is not None else None,
                       "pf_calls": internal_calls + 1 + int(next_pre_attempted),
                       "pf_attempt_failures": internal_failures + int(not valid) + int(info["next_precontrol_failed"]),
                       "decision_s": decision_s, "final_pf_s": env.last_post_flow.elapsed_s,
                       "policy_or_controller_s": compute_s, "reward": reward,
                       "next_precontrol_failed": info["next_precontrol_failed"], "termination_reason": info["termination_reason"],
                       "error": info["failure_reason"]}
                episode_rows.append(row); rows.append(row); episode_reward += reward
                for j, bus in enumerate(grid_config["pv"]["buses"]):
                    commands.append({"scenario": f"seed_{seed}", "method": method, "step": info["step_index"],
                                     "pv_bus": bus, "raw_action": float(action[j]), "q_mvar": float(q[j]),
                                     "qmax_mvar": info["qmax_mvar"][j]})
                digest.update(current_obs.astype(np.float32).tobytes())
                digest.update(np.asarray(action, dtype=np.float32).tobytes())
                digest.update(json.dumps([reward, info["post_voltage_pu"], info["line_loss_mw"], terminated, truncated], allow_nan=False).encode())
                previous_q = q.copy() if valid else None
                if terminated or truncated:
                    break
            summary = {"scenario": f"seed_{seed}", "method": method,
                       **aggregate(episode_rows, env.horizon, grid_config["scenario"]["step_minutes"]),
                       "episode_complete": info["episode_complete"], "episode_return": episode_reward,
                       "threshold_change_count": threshold_changes,
                       "next_precontrol_failures": sum(x["next_precontrol_failed"] for x in episode_rows),
                       "mean_pure_policy_inference_s": float(np.mean([x["policy_or_controller_s"] for x in episode_rows])) if method == "sac" else None,
                       "initial_precontrol_calls": 1, "trajectory_sha256": digest.hexdigest()}
            summaries.append(summary); signatures[f"{seed}/{method}"] = digest.hexdigest()
            env.close()
    if output is not None:
        write_csv(output / "comparison.csv", summaries)
        write_csv(output / "evaluation_steps.csv", rows)
        write_csv(output / "evaluation_actions.csv", commands)
    result = {"summaries": summaries, "signatures": signatures,
              "metric_contract": "same full-day environment and capabilities; 33-node valid sample denominator plus coverage/failures; no extra shield; full decisions include next pre-control solve"}
    if retain_raw:
        result.update(raw_steps=rows, raw_actions=commands, step_minutes=grid_config["scenario"]["step_minutes"])
    return result
