"""Causal, finite-day Gymnasium environment; no learned policy or safety shield."""

from copy import deepcopy
from pathlib import Path
import tomllib

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .network import GridModel, action_to_q
from .scenario import load_config, make_day


ROOT = Path(__file__).resolve().parent.parent


def load_environment_config(path):
    if isinstance(path, dict):
        c = deepcopy(path)
    else:
        with Path(path).open("rb") as stream:
            c = tomllib.load(stream)
    r = c["reward"]
    weights = [r[k] for k in ("voltage_weight", "loss_weight", "violation_weight", "movement_weight")]
    if not np.isfinite(weights).all() or np.any(np.asarray(weights) < 0):
        raise ValueError("Reward weights must be finite and nonnegative")
    if not np.isfinite(r["failure_penalty"]) or r["failure_penalty"] <= 0:
        raise ValueError("failure_penalty must be positive")
    scale = c["observation"]["voltage_scale_pu"]
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Voltage scale must be positive and finite")
    if type(c["episode"]["max_steps"]) is not int or c["episode"]["max_steps"] < 1:
        raise ValueError("max_steps must be a positive integer")
    if r["uncontrollable_mode"] != "mask_normal_costs":
        raise ValueError("Unsupported uncontrollable reward convention")
    for k in ("scenario_seed", "action_seed"):
        if type(c["verification"][k]) is not int or c["verification"][k] < 0:
            raise ValueError("Verification seeds must be nonnegative integers")
    return c


class GridGuardEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, grid_config=None, environment_config=None):
        super().__init__()
        self.grid_config = deepcopy(grid_config) if isinstance(grid_config, dict) else load_config(
            grid_config or ROOT / "configs/ieee33.toml")
        self.settings = load_environment_config(environment_config or ROOT / "configs/environment.toml")
        self.horizon = self.grid_config["scenario"]["steps"]
        if self.settings["episode"]["max_steps"] > self.horizon:
            raise ValueError("max_steps cannot exceed the configured day")
        self.n_pv = len(self.grid_config["pv"]["buses"])
        self.action_space = spaces.Box(-1., 1., shape=(self.n_pv,), dtype=np.float32)
        # tanh(voltage deviation / fixed scale), P/nameplate, total load P/Q/base,
        # Qmax/S, previous executed Q/S, sin/cos time, remaining day, valid flag.
        widths = {"voltage_tanh": 33, "pv_pu": self.n_pv, "load_pq_pu": 2,
                  "qmax_pu": self.n_pv, "previous_q_pu": self.n_pv,
                  "time": 3, "powerflow_valid": 1}
        self.observation_slices = {}
        start = 0
        for name, width in widths.items():
            self.observation_slices[name] = slice(start, start + width)
            start += width
        low = [-1.] * 33 + [0.] * self.n_pv + [0.] * 2 + [0.] * self.n_pv + [-1.] * self.n_pv + [-1., -1., 0., 0.]
        high = [1.] * 33 + [1.] * self.n_pv + [self.grid_config["scenario"]["load_max"]] * 2 + [1.] * self.n_pv + [1.] * self.n_pv + [1., 1., 1., 1.]
        self.observation_space = spaces.Box(np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32))
        self._needs_reset = True
        self.model = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._needs_reset = True
        options = {} if options is None else dict(options)
        if set(options) - {"max_steps", "scenario_seed"}:
            raise ValueError("Supported reset options: max_steps, scenario_seed")
        self.max_steps = options.get("max_steps", self.settings["episode"]["max_steps"])
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= self.horizon:
            raise ValueError("Invalid episode step limit")
        scenario_seed = options.get("scenario_seed", seed)
        if scenario_seed is None:
            scenario_seed = int(self.np_random.integers(0, 2**31 - 1))
        if type(scenario_seed) is not int or scenario_seed < 0:
            raise ValueError("scenario_seed must be a nonnegative integer")
        self.scenario_seed = scenario_seed
        config = deepcopy(self.grid_config)
        config["scenario"]["seed"] = scenario_seed
        self.day = make_day(config)
        self.model = GridModel(config)
        self.t = self.elapsed_steps = 0
        self.previous_q = np.zeros(self.n_pv)
        self.episode_return = 0.
        self._last_observation = None
        self.last_post_flow = None
        error = self._prepare(0)
        if error is not None:
            raise RuntimeError(f"Initial pre-control state unavailable: {error}")
        self._needs_reset = False
        obs = self._observation(self.pre_flow, 0, self.horizon)
        return obs, {"scenario_seed": scenario_seed, "step_index": 0,
                     "precontrol_converged": True, "precontrol_q_mvar": self.initial_q.tolist(),
                     "observation_valid": True}

    @property
    def current_inputs(self):
        if self._needs_reset:
            raise RuntimeError("Call reset before requesting a decision state")
        return {"step_index": self.t, "load_scale": float(self.day["load_scale"][self.t]),
                "p_mw": self.model.p_rated * self.day["pv_fraction"][self.t],
                "qmax_mvar": self.cap.copy()}

    def _prepare(self, index):
        p = self.model.p_rated * self.day["pv_fraction"][index]
        try:
            self.cap = self.model.capability(p)
            self.initial_q = np.clip(self.previous_q, -self.cap, self.cap)
            self.model.set_operating_point(float(self.day["load_scale"][index]), p, self.initial_q)
            self.pre_flow = self.model.solve()
        except ValueError as exc:
            self.pre_flow = None
            return str(exc)
        return None if self.pre_flow.converged else self.pre_flow.error

    def _observation(self, flow, index, remaining):
        p = self.model.p_rated * self.day["pv_fraction"][index]
        hour = float(self.day["hour"][index])
        load = float(self.day["load_scale"][index])
        features = np.concatenate([
            np.tanh((flow.voltage_pu - self.grid_config["network"]["reference_pu"]) /
                    self.settings["observation"]["voltage_scale_pu"]),
            p / self.model.p_rated, [load, load], self.model.capability(p) / self.model.s_rated,
            self.previous_q / self.model.s_rated,
            [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), remaining / self.horizon, 1.]
        ]).astype(np.float32)
        if not self.observation_space.contains(features):
            raise RuntimeError("Computed observation lies outside declared space")
        self._last_observation = features.copy()
        return features

    def _absorbing_observation(self):
        # A dedicated invalid terminal encoding, NOT measured zero voltages or a
        # previous physical snapshot. The final validity feature is zero.
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _reward(self, flow, q, previous_q, cap):
        scale = self.settings["observation"]["voltage_scale_pu"]
        r = self.settings["reward"]
        raw = {"voltage": float(np.mean(np.abs(flow.voltage_pu - self.grid_config["network"]["reference_pu"]))) / scale,
               "loss": float(flow.loss_mw / self.model.base_p.sum()),
               "violation": float(self.model.voltage_violation(flow.voltage_pu).mean()) / scale,
               "movement": float(np.mean(np.abs(q - previous_q) / self.model.s_rated))}
        available = bool(np.any(cap > 0))
        contributions = {name: -r[f"{name}_weight"] * value if available else 0. for name, value in raw.items()}
        contributions["failure"] = 0.
        return float(sum(contributions.values())), raw, contributions, available

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError("Call reset before step or after episode completion")
        if np.iscomplexobj(action):
            raise ValueError("Actions must be real-valued")
        raw_action = np.asarray(action, dtype=float)
        # Reject malformed commands before any state, counter or network mutation.
        q = action_to_q(raw_action, self.cap)
        index = self.t
        inputs = self.current_inputs
        previous_q = self.previous_q.copy()
        initial_q = self.initial_q.copy()
        cap = self.cap.copy()
        self.model.set_operating_point(inputs["load_scale"], inputs["p_mw"], q)
        post_flow = self.model.solve()
        self.last_post_flow = post_flow  # Diagnostic timing/result; excluded from seeded Gym info.
        self.elapsed_steps += 1
        info = {"step_index": index, "hour": float(self.day["hour"][index]),
                "scenario_seed": self.scenario_seed, "load_scale": inputs["load_scale"],
                "pv_total_mw": float(inputs["p_mw"].sum()),
                "raw_action": raw_action.tolist(), "executed_q_mvar": q.tolist(),
                "qmax_mvar": cap.tolist(), "previous_q_mvar": previous_q.tolist(),
                "precontrol_q_mvar": initial_q.tolist(),
                "capability_adjustment_mvar": (initial_q - previous_q).tolist(),
                "postcontrol_converged": post_flow.converged,
                "post_voltage_pu": post_flow.voltage_pu.tolist() if post_flow.converged else None,
                "line_loss_mw": post_flow.loss_mw, "balance_error_mw": post_flow.balance_error_mw,
                "next_precontrol_failed": False,
                "failure_stage": None, "failure_reason": None}
        if not post_flow.converged:
            penalty = -float(self.settings["reward"]["failure_penalty"])
            self._needs_reset = True
            self.episode_return += penalty
            info.update(failure_stage="postcontrol", failure_reason=post_flow.error,
                        observation_valid=False, episode_complete=False, termination_reason="postcontrol_failure",
                        reward_raw={name: None for name in ("voltage", "loss", "violation", "movement")},
                        reward_contributions={"voltage": 0., "loss": 0., "violation": 0., "movement": 0., "failure": penalty},
                        controllable=bool(np.any(cap > 0)))
            return self._absorbing_observation(), penalty, True, False, info

        reward, raw, contributions, available = self._reward(post_flow, q, previous_q, cap)
        self.previous_q = q.copy()
        self.episode_return += reward
        self.t += 1
        info.update(reward_raw=raw, reward_contributions=contributions, controllable=available,
                    observation_valid=True, episode_complete=False, termination_reason=None)
        # Compute current metrics/reward BEFORE reading any next exogenous input.
        if self.t == self.horizon:
            self._needs_reset = True
            info.update(episode_complete=True, termination_reason="day_complete")
            obs = self._observation(post_flow, index, 0)
            return obs, reward, True, False, info
        error = self._prepare(self.t)
        if error is not None:
            self._needs_reset = True
            info.update(next_precontrol_failed=True, failure_stage="next_precontrol",
                        failure_reason=error, failed_step_index=self.t,
                        observation_valid=False, termination_reason="next_precontrol_failure")
            # Current reward stays valid. Do not charge an unexecuted future step
            # to this action or substitute a failed next voltage measurement.
            return self._absorbing_observation(), reward, True, False, info
        obs = self._observation(self.pre_flow, self.t, self.horizon - self.t)
        truncated = self.elapsed_steps >= self.max_steps
        if truncated:
            self._needs_reset = True
            info["termination_reason"] = "external_step_limit"
        return obs, reward, False, truncated, info

    def close(self):
        self._needs_reset = True
        self.model = None
