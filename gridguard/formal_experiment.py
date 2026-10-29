"""Fixed-budget multi-seed training, atomic recovery and held-out final evaluation."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
from time import perf_counter, sleep
import zipfile

import cloudpickle
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from .evaluation import aggregate, write_csv
from .learning import (MeasuredSAC, ScenarioPoolEnv, ValidationCallback, actor_digest,
                       plot_training, scenario_manifest)
from .policy_evaluation import evaluate_policies

ROOT = Path(__file__).resolve().parent.parent


def atomic_replace(source, destination):
    # Windows virus/index scanners can briefly retain a newly closed archive.
    # Retry the replacement without ever deleting the previous valid checkpoint.
    for attempt in range(10):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 9:
                raise
            sleep(.05 * (attempt + 1))


def json_write(path, value):
    # Staging stays in tmp/; replacement is atomic on the same project volume.
    staging = ROOT / "tmp" / f"write_{os.getpid()}_{path.name}"
    staging.parent.mkdir(exist_ok=True)
    staging.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    atomic_replace(staging, path)


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class CheckpointPause(RuntimeError):
    pass


class RecoveryManager:
    """Checkpoint only AFTER replay insertion and gradient updates are complete."""
    def __init__(self, output, callback, config, run_digest, prior_seconds=0., pause_at=None):
        self.output, self.callback, self.config = Path(output), callback, config
        self.run_digest, self.prior_seconds = run_digest, prior_seconds
        self.start = perf_counter()
        self.pause_at = pause_at
        self.stage = ROOT / "tmp" / f"checkpoint_seed{config['run']['seed']}_{os.getpid()}"
        self.stage.mkdir(parents=True, exist_ok=True)

    def progress(self, model, status="training"):
        elapsed = self.prior_seconds + perf_counter() - self.start
        value = {"status": status, "seed": self.config["run"]["seed"], "timesteps": model.num_timesteps,
                 "target_timesteps": self.config["run"]["total_timesteps"], "gradient_updates": model._n_updates,
                 "active_elapsed_s": elapsed, "best_validation_step": self.callback.best_step,
                 "best_mean_validation_return": self.callback.best_score if np.isfinite(self.callback.best_score) else None,
                 "validation_rounds": len(self.callback.validation_rows)}
        json_write(self.output / "progress.json", value)

    def save(self, model):
        model.save(self.stage / "model.zip")
        model.save_replay_buffer(self.stage / "replay.pkl")
        state = {"vec_env": model.get_env(), "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                 "torch_rng": torch.get_rng_state(), "active_seconds": self.prior_seconds + perf_counter() - self.start,
                 "callback": {k: getattr(self.callback, k) for k in
                              ("validation_seconds", "validation_rows", "episode_rows", "best_score", "best_signatures", "best_step", "_last_print_step")}}
        (self.stage / "runtime.pkl").write_bytes(cloudpickle.dumps(state))
        files = ["model.zip", "replay.pkl", "runtime.pkl"]
        if (self.output / "best_model.zip").exists():
            shutil.copyfile(self.output / "best_model.zip", self.stage / "best_model.zip")
            files.append("best_model.zip")
        manifest = {"run_digest": self.run_digest, "timesteps": model.num_timesteps,
                    "gradient_updates": model._n_updates, "replay_size": model.replay_buffer.size(),
                    "sha256": {name: digest_file(self.stage / name) for name in files}}
        (self.stage / "checkpoint.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        pending = self.stage / "last_checkpoint.zip"
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in [*files, "checkpoint.json"]:
                archive.write(self.stage / name, name)
        atomic_replace(pending, self.output / "last_checkpoint.zip")
        write_csv(self.output / "training_episodes.csv", self.callback.episode_rows)
        write_csv(self.output / "training_updates.csv", model.update_rows)
        write_csv(self.output / "validation.csv", self.callback.validation_rows)

    def __call__(self, model):
        formal = self.config["formal"]
        checkpoint = model.num_timesteps % formal["checkpoint_interval"] == 0
        pause_requested = self.pause_at == model.num_timesteps or (ROOT / "tmp/formal_pause_requested.flag").exists()
        if checkpoint or pause_requested:
            self.save(model)
        if model.num_timesteps % formal["progress_interval"] == 0:
            self.progress(model)
        if pause_requested:
            self.progress(model, "paused_at_checkpoint")
            raise CheckpointPause("Paused after atomic checkpoint")


def restore_checkpoint(path, callback, config, run_digest):
    stage = ROOT / "tmp" / f"restore_seed{config['run']['seed']}_{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        expected = {"model.zip", "replay.pkl", "runtime.pkl", "checkpoint.json", "best_model.zip"}
        if not set(archive.namelist()).issubset(expected):
            raise ValueError("Unexpected recovery archive members")
        archive.extractall(stage)
    manifest = json.loads((stage / "checkpoint.json").read_text(encoding="utf-8"))
    if manifest["run_digest"] != run_digest:
        raise ValueError("Recovery configuration/source digest mismatch")
    for name, digest in manifest["sha256"].items():
        if digest_file(stage / name) != digest:
            raise ValueError(f"Corrupted recovery member: {name}")
    state = cloudpickle.loads((stage / "runtime.pkl").read_bytes())
    model = MeasuredSAC.load(stage / "model.zip", env=state["vec_env"], device="cpu", force_reset=False)
    model.load_replay_buffer(stage / "replay.pkl")
    if model.num_timesteps != manifest["timesteps"] or model.replay_buffer.size() != manifest["replay_size"]:
        raise ValueError("Recovery counters mismatch")
    for key, value in state["callback"].items():
        setattr(callback, key, value)
    if "best_model.zip" in manifest["sha256"]:
        shutil.copyfile(stage / "best_model.zip", callback.output / "best_model.zip")
    random.setstate(state["python_rng"]); np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    return model, state["active_seconds"]


def build_model(grid, settings, config):
    env = ScenarioPoolEnv(grid, settings, config["scenarios"]["train_seeds"], config["run"]["seed"])
    s = config["sac"]
    model = MeasuredSAC("MlpPolicy", Monitor(env), seed=config["run"]["seed"], device="cpu", verbose=0,
                        learning_rate=s["learning_rate"], buffer_size=s["buffer_size"], learning_starts=s["learning_starts"],
                        batch_size=s["batch_size"], tau=s["tau"], gamma=s["gamma"], train_freq=s["train_freq"],
                        gradient_steps=s["gradient_steps"], ent_coef=s["ent_coef"], policy_kwargs={"net_arch": s["net_arch"]})
    model.update_rows, model.log_every_updates = [], s["log_every_updates"]
    return model


def train_seed(grid, settings, baselines, config, output, provenance, resume=False, pause_at=None):
    output = Path(output)
    torch.set_num_threads(config["run"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    run_data = {"grid": grid, "environment": settings, "baselines": baselines, "sac": config,
                "source_sha256": provenance["source_sha256"], "dependencies": provenance["dependencies"]}
    run_digest = hashlib.sha256(json.dumps(run_data, sort_keys=True).encode()).hexdigest()
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        if json.loads((output / "run_config.json").read_text(encoding="utf-8"))["run_digest"] != run_digest:
            raise ValueError("Existing run uses different inputs or code")
        if (output / "summary.json").exists():
            return json.loads((output / "summary.json").read_text(encoding="utf-8"))
    else:
        output.mkdir(parents=True)
        json_write(output / "run_config.json", {"run_digest": run_digest, **run_data})
    callback = ValidationCallback(grid, settings, baselines, config, output)
    checkpoint = output / "last_checkpoint.zip"
    if resume and checkpoint.exists():
        model, prior = restore_checkpoint(checkpoint, callback, config, run_digest)
    else:
        model, prior = build_model(grid, settings, config), 0.
    manager = RecoveryManager(output, callback, config, run_digest, prior, pause_at)
    model._after_update_hook = manager
    before = actor_digest(model)
    remaining = config["run"]["total_timesteps"] - model.num_timesteps
    try:
        if remaining:
            model.learn(total_timesteps=remaining, reset_num_timesteps=model.num_timesteps == 0,
                        callback=callback, log_interval=None, progress_bar=False)
    except CheckpointPause:
        model.get_env().close()
        return {"status": "paused_at_checkpoint", "seed": config["run"]["seed"], "timesteps": model.num_timesteps}
    except Exception as exc:
        manager.progress(model, "failed")
        json_write(output / "failure.json", {"error": repr(exc), "timesteps": model.num_timesteps})
        model.get_env().close()
        raise
    if not checkpoint.exists():
        manager.save(model)
    else:
        with zipfile.ZipFile(checkpoint) as archive:
            saved_step = json.loads(archive.read("checkpoint.json"))["timesteps"]
        if saved_step != model.num_timesteps:
            manager.save(model)
    if callback.best_step is None:
        raise RuntimeError("No validated checkpoint was selected")
    selected = SAC.load(output / "best_model.zip", device="cpu")
    verification = evaluate_policies(grid, settings, baselines, config["scenarios"]["validation_seeds"], ["sac"], selected)
    if verification["signatures"] != callback.best_signatures:
        raise AssertionError("Saved best model changed validation trajectories")
    plot_training(callback.episode_rows, callback.validation_rows, output)
    result = {"status": "training_complete", "seed": config["run"]["seed"], "timesteps": model.num_timesteps,
              "gradient_updates": model._n_updates, "replay_size": model.replay_buffer.size(),
              "actor_changed_in_this_process": actor_digest(model) != before if remaining else None,
              "best_validation_step": callback.best_step, "best_validation_return": callback.best_score,
              "best_model_sha256": digest_file(output / "best_model.zip"), "best_reload_trajectory_exact": True,
              "active_elapsed_s": prior + perf_counter() - manager.start,
              "complete_training_episodes": sum(x["complete_episode"] for x in callback.episode_rows),
              "failed_training_episodes": sum(not x["complete_episode"] for x in callback.episode_rows),
              "last_training_episode_steps": len(model.get_env().envs[0].rewards),
              "test_performance_evaluated_during_training": False, "run_digest": run_digest}
    json_write(output / "summary.json", result)
    manager.progress(model, "training_complete")
    model.get_env().close()
    return result


def summarize_test(results, output):
    per_scenario, per_run, all_steps, all_actions = [], [], [], []
    for label, result in results:
        for row in result["summaries"]:
            per_scenario.append({"run": label, **row})
        # Re-pool raw steps before computing seed-level metrics, preserving failures.
        data = result["raw_steps"]
        for method in dict.fromkeys(x["method"] for x in data):
            rows = [x for x in data if x["method"] == method]
            cases = [x for x in result["summaries"] if x["method"] == method]
            pooled = aggregate(rows, sum(x["planned_steps"] for x in cases), result["step_minutes"])
            per_run.append({"run": label, "method": method, **pooled,
                            "episode_completion_rate": float(np.mean([x["episode_complete"] for x in cases])),
                            "safe_full_episode_rate": float(np.mean([x["full_horizon_no_failure_no_violation"] for x in cases])),
                            "next_precontrol_failures": sum(x["next_precontrol_failures"] for x in cases),
                            "mean_pure_policy_inference_s": float(np.mean([x["mean_pure_policy_inference_s"] for x in cases])) if method == "sac" else None})
        all_steps.extend({"run": label, **x} for x in data)
        all_actions.extend({"run": label, **x} for x in result["raw_actions"])
    metrics = ("voltage_violation_rate", "mean_abs_deviation_pu", "max_abs_deviation_pu", "mean_line_loss_mw",
               "line_loss_mwh_valid_steps", "mean_abs_q_mvar", "mean_abs_delta_q_mvar", "final_step_failure_rate",
               "valid_sample_coverage", "episode_completion_rate", "safe_full_episode_rate", "mean_decision_s",
               "mean_pure_policy_inference_s", "pf_attempt_failures", "next_precontrol_failures")
    aggregates = []
    for method in ("unity_pf", "volt_var", "dsg", "sac"):
        runs = [x for x in per_run if x["method"] == method]
        row = {"method": method, "training_seed_count": len(runs) if method == "sac" else None}
        for metric in metrics:
            values = [x[metric] for x in runs if x[metric] is not None]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_seed_std"] = float(np.std(values, ddof=1)) if method == "sac" and len(values) > 1 else None
        aggregates.append(row)
    write_csv(output / "test_per_scenario.csv", per_scenario)
    write_csv(output / "test_per_seed.csv", per_run)
    write_csv(output / "test_aggregate.csv", aggregates)
    write_csv(output / "test_steps.csv", all_steps)
    write_csv(output / "test_actions.csv", all_actions)
    plot_test(aggregates, all_steps, output)
    return {"per_scenario": per_scenario, "per_seed": per_run, "aggregates": aggregates,
            "std_definition": "sample standard deviation (ddof=1) across trained seeds after pooling all test days per seed; deterministic baselines run once",
            "all_seeds_reported": True}


def plot_test(aggregate_rows, steps, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, metric, scale, ylabel in ((axes[0], "voltage_violation_rate", 100, "Voltage violation (%)"),
                                      (axes[1], "mean_line_loss_mw", 1000, "Mean line loss (kW)")):
        for i, row in enumerate(aggregate_rows):
            mean = row[f"{metric}_mean"]
            if mean is not None:
                ax.bar(i, mean * scale, color=f"C{i}")
                sd = row[f"{metric}_seed_std"]
                if sd is not None:
                    ax.errorbar(i, mean * scale, yerr=sd * scale, fmt="none", color="black", capsize=5)
        ax.set_xticks(range(4), ["Unity PF", "Volt-VAR", "DSG", "SAC"])
        ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Held-out synthetic days | SAC: mean +/- training-seed SD")
    fig.savefig(output / "test_comparison.png", dpi=150); plt.close(fig)
    scenes = list(dict.fromkeys(x["scenario"] for x in steps))
    fig, axes = plt.subplots(2, len(scenes), figsize=(5 * len(scenes), 7), squeeze=False, layout="constrained")
    groups = list(dict.fromkeys((x["run"], x["method"]) for x in steps))
    for col, scene in enumerate(scenes):
        for run, method in groups:
            data = [x for x in steps if x["scenario"] == scene and x["run"] == run and x["method"] == method]
            label = f"SAC {run}" if method == "sac" else method
            for row, key in enumerate(("vmin_pu", "vmax_pu")):
                axes[row, col].plot([x["hour"] for x in data], [x[key] if x[key] is not None else np.nan for x in data], label=label, linewidth=1)
        axes[0, col].set_title(scene)
        axes[0, col].axhline(.95, color="red", linestyle="--", linewidth=.7)
        axes[1, col].axhline(1.05, color="red", linestyle="--", linewidth=.7)
        axes[1, col].set_xlabel("Hour")
        for row in range(2): axes[row, col].grid(alpha=.2)
    axes[0, 0].set_ylabel("Minimum V (p.u.)"); axes[1, 0].set_ylabel("Maximum V (p.u.)")
    axes[0, 0].legend(fontsize=7)
    fig.savefig(output / "test_voltage_curves.png", dpi=120); plt.close(fig)


def run_formal(grid_config, settings, baselines, config, output, provenance, resume=False):
    output = Path(output)
    formal = config["formal"]
    if formal["seeds"] != [7, 17, 27] or config["run"]["total_timesteps"] != 50000:
        raise ValueError("Formal entry is restricted to the approved 3 x 50000 budget")
    if not 1 <= formal["workers"] <= 2:
        raise ValueError("Approved local run uses at most two workers")
    if (config["run"]["validation_interval"], config["sac"]["learning_starts"], config["sac"]["buffer_size"],
            formal["checkpoint_interval"]) != (5000, 1000, 100000, 5000):
        raise ValueError("Formal schedule differs from the approved validation/warmup/buffer/checkpoint budget")
    if settings["episode"]["max_steps"] != grid_config["scenario"]["steps"]:
        raise ValueError("Formal evaluation requires whole-day episodes")
    grid, manifest = scenario_manifest(grid_config, config)
    freeze = {"grid": grid_config, "environment": settings, "baselines": baselines, "config": config,
              "source_sha256": provenance["source_sha256"], "dependencies": provenance["dependencies"]}
    freeze_hash = hashlib.sha256(json.dumps(freeze, sort_keys=True).encode()).hexdigest()
    plan_file = output / "frozen_plan.json"
    if plan_file.exists():
        if not resume or json.loads(plan_file.read_text(encoding="utf-8"))["freeze_hash"] != freeze_hash:
            raise ValueError("Existing experiment does not match this frozen run")
        if (output / "summary.json").exists():
            return json.loads((output / "summary.json").read_text(encoding="utf-8"))
    else:
        if any(output.iterdir()):
            raise ValueError("Refusing to initialize inside an unrecognized nonempty experiment directory")
        json_write(plan_file, {"freeze_hash": freeze_hash, **freeze, "provenance": provenance})
        json_write(output / "scenario_manifest.json", manifest)
        with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            names = list(provenance["source_sha256"])
            names += [str(p.relative_to(ROOT)) for p in (ROOT / "configs").glob("*.toml")]
            for name in names: archive.write(ROOT / name, name)
    tag = output.name.removeprefix("sac_formal_")
    run_paths = {seed: output.parent / f"sac_seed{seed}_{tag}" for seed in formal["seeds"]}
    json_write(output / "experiment_status.json", {"status": "training", "runs": {str(k): str(v.relative_to(output.parent)) for k, v in run_paths.items()}, "test_evaluation_started": False})
    with ProcessPoolExecutor(max_workers=formal["workers"]) as pool:
        futures = {}
        for seed, path in run_paths.items():
            per_seed = deepcopy(config); per_seed["run"]["seed"] = seed
            futures[pool.submit(train_seed, grid, settings, baselines, per_seed, path, provenance, resume)] = seed
        trained = [future.result() for future in as_completed(futures)]
    if any(x["status"] != "training_complete" for x in trained):
        json_write(output / "experiment_status.json", {"status": "paused", "test_evaluation_started": False})
        return {"status": "paused", "training": trained}
    # Freeze all selected models BEFORE the first held-out power flow is solved.
    selected = {seed: digest_file(path / "best_model.zip") for seed, path in run_paths.items()}
    json_write(output / "selected_models.json", {"sha256": selected, "selection": "validation mean return only"})
    json_write(output / "experiment_status.json", {"status": "final_test_evaluation", "test_evaluation_started": True})
    torch.set_num_threads(1)
    tests = config["scenarios"]["test_seeds"]
    all_results = []
    baseline = evaluate_policies(grid, settings, baselines, tests, ["unity_pf", "volt_var", "dsg"], retain_raw=True)
    all_results.append(("deterministic", baseline))
    for seed, path in run_paths.items():
        if digest_file(path / "best_model.zip") != selected[seed]:
            raise ValueError("Selected checkpoint changed after test freeze")
        model = SAC.load(path / "best_model.zip", device="cpu")
        result = evaluate_policies(grid, settings, baselines, tests, ["sac"], model=model, retain_raw=True)
        all_results.append((str(seed), result))
    report = summarize_test(all_results, output)
    report.update(status="complete", training=sorted(trained, key=lambda x: x["seed"]),
                  freeze_hash=freeze_hash, source_snapshot_sha256=digest_file(output / "source_snapshot.zip"),
                  test_seeds=tests, selected_model_sha256=selected, extra_safety_layer=False)
    json_write(output / "summary.json", report)
    json_write(output / "experiment_status.json", {"status": "complete", "test_evaluation_started": True})
    return report
