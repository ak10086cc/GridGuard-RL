"""Single-seed SAC smoke workflow; whole-day splits and validation-only selection."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter
import tomllib

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from .environment import GridGuardEnv
from .evaluation import write_csv
from .policy_evaluation import evaluate_policies
from .scenario import make_day


def load_sac_config(path, *, formal=False):
    with Path(path).open("rb") as stream:
        c = tomllib.load(stream)
    r, s, d = (c[k] for k in ("run", "sac", "scenarios"))
    for values in (r, s):
        keys = (("seed", "torch_threads", "total_timesteps", "validation_interval") if values is r else
                ("buffer_size", "learning_starts", "batch_size", "train_freq", "gradient_steps", "log_every_updates"))
        if any(type(values[k]) is not int or values[k] < (0 if k == "seed" else 1) for k in keys):
            raise ValueError("SAC seeds/counts must be valid integers")
    if r["device"] != "cpu":
        raise ValueError("This validated smoke configuration uses CPU only")
    ceiling = 50000 if formal else 2048
    if not s["learning_starts"] < r["total_timesteps"] <= ceiling:
        raise ValueError(f"Steps must exceed warmup and be <= {ceiling} for the selected workflow")
    if s["batch_size"] > min(s["buffer_size"], s["learning_starts"]):
        raise ValueError("Batch size exceeds initial replay samples or buffer")
    if r["validation_interval"] > r["total_timesteps"] or r["total_timesteps"] % r["validation_interval"]:
        raise ValueError("Smoke validation intervals must divide the training length")
    if not np.isfinite([s["learning_rate"], s["tau"], s["gamma"]]).all():
        raise ValueError("SAC numeric parameters must be finite")
    if s["learning_rate"] <= 0 or not 0 < s["tau"] <= 1 or not 0 < s["gamma"] <= 1:
        raise ValueError("Invalid SAC learning rate/tau/gamma")
    if s["ent_coef"] != "auto" or not s["net_arch"] or any(type(x) is not int or x < 1 for x in s["net_arch"]):
        raise ValueError("Invalid entropy setting/network architecture")
    pools = [d[k] for k in ("train_seeds", "validation_seeds", "test_seeds", "reserved_development_seeds")]
    for seeds in pools:
        if not seeds or len(set(seeds)) != len(seeds) or any(type(x) is not int or x < 0 for x in seeds):
            raise ValueError("Scenario seed pools must be nonempty, unique nonnegative integers")
    for i, pool in enumerate(pools):
        if any(set(pool) & set(other) for other in pools[i + 1:]):
            raise ValueError("Train/validation/test/development scenario seeds must be disjoint")
    if not 0 <= d["pv_cloud_min"] <= d["pv_cloud_max"] <= 1:
        raise ValueError("Invalid cloud range")
    return c


def scenario_manifest(grid_config, config):
    """Compare complete curve content, not just seed labels. No test outcomes are solved."""
    grid = deepcopy(grid_config)
    grid["scenario"].update(pv_cloud_min=config["scenarios"]["pv_cloud_min"],
                            pv_cloud_max=config["scenarios"]["pv_cloud_max"])
    manifest, seen = {}, {}
    for split in ("train", "validation", "test"):
        entries = []
        for seed in config["scenarios"][f"{split}_seeds"]:
            case = deepcopy(grid)
            case["scenario"]["seed"] = seed
            day = make_day(case)
            digest = hashlib.sha256(b"".join(day[k].astype(np.float64).tobytes() for k in sorted(day))).hexdigest()
            if digest in seen:
                raise ValueError(f"Duplicate whole-day content: {seen[digest]} and {split}/{seed}")
            seen[digest] = f"{split}/{seed}"
            entries.append({"seed": seed, "whole_day_sha256": digest, "steps": case["scenario"]["steps"]})
        manifest[split] = entries
    return grid, manifest


class ScenarioPoolEnv(GridGuardEnv):
    def __init__(self, grid, settings, seeds, seed):
        super().__init__(grid, settings)
        self.pool = tuple(seeds)
        self.sampler = np.random.default_rng(seed)
        self.order = []
        self.used_scenarios = []
        self.step_seconds = 0.

    def reset(self, *, seed=None, options=None):
        if options and "scenario_seed" in options:
            raise ValueError("Training scenarios must come from the declared training pool")
        if seed is not None:
            self.sampler = np.random.default_rng(seed)
            self.order = []
        if not self.order:
            self.order = self.sampler.permutation(self.pool).tolist()
        scene = self.order.pop()
        self.used_scenarios.append(scene)
        return super().reset(seed=seed, options={**(options or {}), "scenario_seed": scene})

    def step(self, action):
        start = perf_counter()
        result = super().step(action)
        self.step_seconds += perf_counter() - start
        return result


class MeasuredSAC(SAC):
    """Timing/finite-loss instrumentation; learning remains SB3's SAC.train."""
    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["_after_update_hook"]

    def train(self, gradient_steps, batch_size=64):
        start = perf_counter()
        super().train(gradient_steps, batch_size)
        self.update_seconds = getattr(self, "update_seconds", 0.) + perf_counter() - start
        metrics = {key: float(self.logger.name_to_value[f"train/{key}"])
                   for key in ("actor_loss", "critic_loss", "ent_coef")}
        if not np.isfinite(list(metrics.values())).all():
            raise FloatingPointError("SAC produced non-finite loss/entropy")
        if self._n_updates % self.log_every_updates == 0:
            self.update_rows.append({"timesteps": self.num_timesteps, "updates": self._n_updates, **metrics})
        hook = getattr(self, "_after_update_hook", None)
        if hook is not None:
            hook(self)


class ValidationCallback(BaseCallback):
    def __init__(self, grid, settings, baselines, config, output):
        super().__init__(verbose=0)
        self.grid, self.settings, self.baselines = grid, settings, baselines
        self.config, self.output = config, output
        self.validation_seconds = 0.
        self.validation_rows, self.episode_rows = [], []
        self.best_score = -np.inf
        self.best_signatures = None
        self.best_step = None
        self._last_print_step = 0

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                row = {"timesteps": self.num_timesteps, "scenario_seed": info["scenario_seed"],
                       "episode_return": info["episode"]["r"], "length": info["episode"]["l"],
                       "complete_episode": info["episode_complete"], "failure_stage": info["failure_stage"]}
                self.episode_rows.append(row)
                if self.num_timesteps >= self._last_print_step + self.config["run"].get("episode_print_interval", 96):
                    print(f"SAC {self.num_timesteps}/{self.config['run']['total_timesteps']} steps; training day {info['scenario_seed']}", flush=True)
                    self._last_print_step = self.num_timesteps
        if self.num_timesteps % self.config["run"]["validation_interval"] == 0:
            start = perf_counter()
            result = evaluate_policies(self.grid, self.settings, self.baselines,
                                       self.config["scenarios"]["validation_seeds"], ["sac"], self.model)
            if not all(x["episode_complete"] for x in result["summaries"]):
                raise RuntimeError("Validation did not complete all planned days")
            score = float(np.mean([x["episode_return"] for x in result["summaries"]]))
            self.validation_rows.append({"timesteps": self.num_timesteps, "updates_before_this_step_training": self.model._n_updates,
                                         "mean_validation_return": score})
            if score > self.best_score:
                self.best_score, self.best_step = score, self.num_timesteps
                self.best_signatures = result["signatures"]
                self.model.save(self.output / "best_model.zip")
            self.validation_seconds += perf_counter() - start
        return True


def actor_digest(model):
    return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.actor.parameters())).hexdigest()


def estimate_budget(config, timings):
    p = config["budget_proposal"]
    updates = max(0, p["timesteps_per_seed"] - p["learning_starts"]) * config["sac"]["gradient_steps"] / config["sac"]["train_freq"]
    validations = (p["timesteps_per_seed"] + p["validation_interval"] - 1) // p["validation_interval"]
    one_seed = p["timesteps_per_seed"] * timings["collection_and_other_s_per_step"] + updates * timings["s_per_gradient_update"] + validations * timings["mean_validation_round_s"]
    total = len(p["seeds"]) * one_seed
    return {"executed": False, "seeds": p["seeds"], "timesteps_per_seed": p["timesteps_per_seed"],
            "validation_interval": p["validation_interval"], "learning_starts": p["learning_starts"],
            "buffer_size": p["buffer_size"], "estimated_s_per_seed": one_seed,
            "estimated_training_total_s": total, "planning_range_s": [total * .7, total * 1.5],
            "final_shared_test_evaluation_additional_s_estimate": timings["validation_comparison_s"] * len(config["scenarios"]["test_seeds"]) / len(config["scenarios"]["validation_seeds"]) * len(p["seeds"]),
            "limits": "CPU smoke extrapolation; includes periodic validation, excludes install and large-buffer serialization; test baseline work may be shared across seeds",
            "saving": "one exclusive results directory per seed; best validation model, last model plus matching last replay buffer; no per-step checkpoints"}


def run_sac_smoke(grid_config, environment_config, baseline_config, config, output):
    if environment_config["episode"]["max_steps"] != grid_config["scenario"]["steps"]:
        raise ValueError("Training/evaluation smoke requires complete-day episodes")
    if config["run"]["total_timesteps"] % grid_config["scenario"]["steps"]:
        raise ValueError("Smoke training must contain whole episodes")
    torch.set_num_threads(config["run"]["torch_threads"])
    torch.use_deterministic_algorithms(True)
    grid, manifest = scenario_manifest(grid_config, config)
    (output / "scenario_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checker = GridGuardEnv(grid, environment_config)
    check_env(checker, warn=True)
    checker.close()
    training_env = ScenarioPoolEnv(grid, environment_config, config["scenarios"]["train_seeds"], config["run"]["seed"])
    s = config["sac"]
    model = MeasuredSAC("MlpPolicy", Monitor(training_env), seed=config["run"]["seed"], device="cpu", verbose=0,
                        learning_rate=s["learning_rate"], buffer_size=s["buffer_size"], learning_starts=s["learning_starts"],
                        batch_size=s["batch_size"], tau=s["tau"], gamma=s["gamma"], train_freq=s["train_freq"],
                        gradient_steps=s["gradient_steps"], ent_coef=s["ent_coef"], policy_kwargs={"net_arch": s["net_arch"]})
    model.update_rows, model.log_every_updates = [], s["log_every_updates"]
    before = actor_digest(model)
    callback = ValidationCallback(grid, environment_config, baseline_config, config, output)
    start = perf_counter()
    model.learn(total_timesteps=config["run"]["total_timesteps"], callback=callback, log_interval=None, progress_bar=False)
    learn_seconds = perf_counter() - start
    changed = actor_digest(model) != before
    if not changed or model._n_updates <= 0:
        raise AssertionError("Smoke did not actually update actor parameters")
    model.save(output / "last_model.zip")
    model.save_replay_buffer(output / "last_replay_buffer.pkl")
    last = SAC.load(output / "last_model.zip", device="cpu")
    last.load_replay_buffer(output / "last_replay_buffer.pkl")
    if last.replay_buffer.size() != model.replay_buffer.size():
        raise AssertionError("Last-model replay buffer did not reload")
    best = SAC.load(output / "best_model.zip", device="cpu")
    reload_result = evaluate_policies(grid, environment_config, baseline_config, config["scenarios"]["validation_seeds"], ["sac"], best)
    reload_exact = reload_result["signatures"] == callback.best_signatures
    if not reload_exact:
        raise AssertionError("Loaded best model changed validation actions/physics/rewards")
    comparison_start = perf_counter()
    comparison = evaluate_policies(grid, environment_config, baseline_config, config["scenarios"]["validation_seeds"],
                                   ["unity_pf", "volt_var", "dsg", "sac"], best, output)
    comparison_seconds = perf_counter() - comparison_start
    if not all(x["episode_complete"] and x["algorithm_nonconverged_steps"] == 0 for x in comparison["summaries"]):
        raise RuntimeError("Unified validation comparison failed to complete")
    write_csv(output / "training_episodes.csv", callback.episode_rows)
    write_csv(output / "training_updates.csv", model.update_rows)
    write_csv(output / "validation.csv", callback.validation_rows)
    plot_training(callback.episode_rows, callback.validation_rows, output)
    update_seconds = model.update_seconds
    timings = {"learn_wall_s_including_validation": learn_seconds,
               "validation_callback_s": callback.validation_seconds,
               "training_s_excluding_validation": learn_seconds - callback.validation_seconds,
               "environment_step_s": training_env.step_seconds, "gradient_update_s": update_seconds,
               "s_per_gradient_update": update_seconds / model._n_updates,
               "collection_and_other_s_per_step": max(learn_seconds - callback.validation_seconds - update_seconds, 0) / model.num_timesteps,
               "mean_validation_round_s": callback.validation_seconds / len(callback.validation_rows),
               "validation_comparison_s": comparison_seconds}
    result = {"scope": "single-seed smoke; not convergence, superiority or independent test evidence",
              "sb3_env_checker_passed": True, "device": str(model.device), "timesteps": model.num_timesteps,
              "gradient_updates": model._n_updates, "actor_parameters_changed": changed,
              "best_validation_step": callback.best_step, "best_mean_validation_return": callback.best_score,
              "best_reload_trajectory_exact": reload_exact, "last_replay_size": model.replay_buffer.size(),
              "training_scenario_resets": training_env.used_scenarios,
              "training_complete_episodes": sum(x["complete_episode"] for x in callback.episode_rows),
              "test_performance_evaluated": False,
              "selection_rule": "highest mean return on declared validation days only; earlier checkpoint wins ties",
              "checkpoint_timing": "callback runs before this step's gradient update; final model is saved after all updates",
              "comparison": comparison, "timings": timings, "formal_budget_proposal": estimate_budget(config, timings)}
    model.get_env().close()
    return result


def plot_training(episodes, validation, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), layout="constrained")
    axes[0].plot([x["timesteps"] for x in episodes], [x["episode_return"] for x in episodes], marker=".")
    axes[0].set_ylabel("Training episode return")
    axes[1].plot([x["timesteps"] for x in validation], [x["mean_validation_return"] for x in validation], marker="o")
    axes[1].set_ylabel("Mean validation return")
    for ax in axes:
        ax.set_xlabel("Environment steps"); ax.grid(alpha=.25)
    fig.suptitle("SAC smoke workflow only | no convergence claim")
    fig.savefig(output / "training_curve.png", dpi=140)
    plt.close(fig)
