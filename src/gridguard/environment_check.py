"""Official API check and complete-day random/baseline integration runs, not training."""

from time import perf_counter

from gymnasium.utils.env_checker import check_env, data_equivalence
import numpy as np

from .controllers import control
from .environment import GridGuardEnv
from .evaluation import write_csv


def run_episode(env, policy, baseline_config, scenario_seed, action_seed):
    obs, _ = env.reset(seed=scenario_seed)
    env.action_space.seed(action_seed)
    steps, actions, signatures = [], [], []
    baseline_failures = 0
    start = perf_counter()
    while True:
        if not env.observation_space.contains(obs):
            raise AssertionError("Observation space mismatch")
        inputs = env.current_inputs
        if policy == "random":
            action = env.action_space.sample()
        else:
            selected = control(policy, env.model, inputs["load_scale"], inputs["p_mw"], baseline_config)
            baseline_failures += not selected.algorithm_converged
            # Normalized physical command; only roundoff is clipped in this adapter.
            action = np.divide(selected.q_mvar, inputs["qmax_mvar"],
                               out=np.zeros(env.n_pv), where=inputs["qmax_mvar"] > 0)
            action = np.clip(action, -1, 1).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if abs(reward - sum(info["reward_contributions"].values())) > 1e-9:
            raise AssertionError("Reward components do not sum to returned reward")
        q, cap = np.asarray(info["executed_q_mvar"]), np.asarray(info["qmax_mvar"])
        if np.any(np.abs(q) > cap + 1e-12):
            raise AssertionError("Inverter capability violation")
        voltage = info["post_voltage_pu"]
        valid = info["postcontrol_converged"]
        violated = int(np.count_nonzero(env.model.voltage_violation(np.asarray(voltage)))) if valid else None
        row = {"policy": policy, "step": info["step_index"], "hour": info["hour"],
               "load_scale": info["load_scale"], "pv_total_mw": info["pv_total_mw"],
               "postcontrol_converged": valid, "next_precontrol_failed": info["next_precontrol_failed"],
               "vmin_pu": min(voltage) if valid else None, "vmax_pu": max(voltage) if valid else None,
               "violated_nodes": violated, "line_loss_mw": info["line_loss_mw"],
               "reward": reward, "controllable": info["controllable"],
               "terminated": terminated, "truncated": truncated,
               "termination_reason": info["termination_reason"], "failure_stage": info["failure_stage"],
               "failure_reason": info["failure_reason"]}
        for name in ("voltage", "loss", "violation", "movement"):
            row[f"raw_{name}"] = info["reward_raw"][name]
            row[f"reward_{name}"] = info["reward_contributions"][name]
        row["reward_failure"] = info["reward_contributions"]["failure"]
        steps.append(row)
        for j, bus in enumerate(env.grid_config["pv"]["buses"]):
            actions.append({"policy": policy, "step": info["step_index"], "pv_bus": bus,
                            "raw_action": info["raw_action"][j], "q_mvar": float(q[j]), "qmax_mvar": float(cap[j]),
                            "previous_q_mvar": info["previous_q_mvar"][j],
                            "precontrol_q_mvar": info["precontrol_q_mvar"][j],
                            "capability_adjustment_mvar": info["capability_adjustment_mvar"][j]})
        signatures.append((obs.copy(), reward, terminated, truncated, info))
        if terminated or truncated:
            break
    valid_rows = [x for x in steps if x["postcontrol_converged"]]
    completed = bool(info["episode_complete"] and len(steps) == env.horizon)
    summary = {"policy": policy, "scenario_seed": scenario_seed, "action_seed": action_seed,
               "attempted_steps": len(steps), "planned_steps": env.horizon,
               "successful_postcontrol_steps": len(valid_rows),
               "postcontrol_failures": len(steps) - len(valid_rows),
               "next_precontrol_failures": sum(x["next_precontrol_failed"] for x in steps),
               "baseline_algorithm_nonconverged": int(baseline_failures),
               "complete_episode": completed, "terminated": terminated, "truncated": truncated,
               "final_reason": info["termination_reason"],
               "reward_sum": float(sum(x["reward"] for x in steps)),
               "reward_masked_steps": sum(not x["controllable"] for x in steps),
               "valid_sample_coverage": len(valid_rows) / env.horizon,
               "voltage_node_time_denominator": len(valid_rows) * 33,
               "voltage_violation_rate": sum(x["violated_nodes"] for x in valid_rows) / (len(valid_rows) * 33) if valid_rows else None,
               "elapsed_s_with_policy_and_environment": perf_counter() - start}
    return summary, steps, actions, signatures


def verify_environment(grid_config, settings, baseline_config, output):
    env = GridGuardEnv(grid_config, settings)
    check_env(env, skip_render_check=True)
    definition = {"observation_shape": list(env.observation_space.shape),
                  "observation_dtype": str(env.observation_space.dtype),
                  "observation_slices": {k: [s.start, s.stop] for k, s in env.observation_slices.items()},
                  "action_shape": list(env.action_space.shape), "action_bounds": [-1, 1],
                  "horizon": env.horizon, "time_step_minutes": grid_config["scenario"]["step_minutes"],
                  "failure_observation": "absorbing zero feature vector with validity=0; not physical zero voltages",
                  "normal_day_end": "terminated=True; truncated=False",
                  "external_limit": "truncated=True with valid next observation",
                  "simulator_failure": "terminated=True, explicit invalid-state encoding and failure stage",
                  "night_reward": settings["reward"]["uncontrollable_mode"],
                  "training_performed": False}
    env.close()
    summaries, steps, actions, repeats = [], [], [], []
    scenario_seed = settings["verification"]["scenario_seed"]
    action_seed = settings["verification"]["action_seed"]
    for policy in ("random", "unity_pf", "volt_var", "dsg"):
        env = GridGuardEnv(grid_config, settings)
        summary, rows, commands, signature = run_episode(env, policy, baseline_config, scenario_seed, action_seed)
        summaries.append(summary); steps.extend(rows); actions.extend(commands)
        env.close()
        if policy in ("random", "unity_pf"):
            repeated = GridGuardEnv(grid_config, settings)
            _, _, _, other = run_episode(repeated, policy, baseline_config, scenario_seed, action_seed)
            equivalent = data_equivalence(signature, other, exact=True)
            repeats.append({"policy": policy, "full_day_exact_repeat": equivalent})
            repeated.close()
        print(f"Environment {policy}: {summary['successful_postcontrol_steps']}/{summary['planned_steps']} successful steps; {summary['final_reason']}", flush=True)
    write_csv(output / "episode_steps.csv", steps)
    write_csv(output / "episode_actions.csv", actions)
    plot_episodes(steps, output)
    return {"official_check_env_passed": True, "render_check_skipped_no_render_mode": True,
            "definition": definition, "episodes": summaries, "determinism_checks": repeats,
            "purpose": "API and simulation integration only; initial reward weights; no policy performance claims"}


def plot_episodes(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout="constrained")
    for policy in dict.fromkeys(x["policy"] for x in rows):
        values = [x for x in rows if x["policy"] == policy]
        for ax, key in zip(axes, ("vmin_pu", "vmax_pu", "reward")):
            ax.plot([x["hour"] for x in values],
                    [x[key] if x[key] is not None else np.nan for x in values], label=policy)
    for ax, label in zip(axes, ("Minimum V (p.u.)", "Maximum V (p.u.)", "Step reward (initial weights)")):
        ax.set_ylabel(label); ax.grid(alpha=.25)
    axes[0].legend()
    axes[2].set_xlabel("Hour")
    fig.suptitle("Gymnasium integration checks | synthetic day | no training")
    fig.savefig(output / "environment_episodes.png", dpi=140)
    plt.close(fig)
