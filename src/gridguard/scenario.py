"""Validated config and reproducible synthetic day; no measured-data claims."""

from pathlib import Path
import tomllib

import numpy as np


def load_config(path: str | Path) -> dict:
    with Path(path).open("rb") as stream:
        config = tomllib.load(stream)
    n, p, s, v = (config[key] for key in ("network", "pv", "scenario", "verification"))
    numeric = [n["sn_mva"], n["voltage_min_pu"], n["voltage_max_pu"],
               n["reference_pu"], n["tolerance_mva"], p["installed_to_base_load_ratio"],
               p["inverter_s_to_p_ratio"], s["step_minutes"], s["sunrise_hour"],
               s["sunset_hour"], s["load_min"], s["load_max"], s["load_noise_std"],
               s["pv_cloud_min"], s["pv_cloud_max"], v["probe_fraction_of_capability"],
               v["stress_load_scale"], *v["candidate_fractions"]]
    if not np.isfinite(np.asarray(numeric, dtype=float)).all():
        raise ValueError("Configuration numbers must be finite")
    if not (0 < n["voltage_min_pu"] < n["reference_pu"] < n["voltage_max_pu"]):
        raise ValueError("Invalid voltage limits/reference")
    if n["sn_mva"] <= 0 or n["tolerance_mva"] <= 0:
        raise ValueError("Base power and PF tolerance must be positive")
    if n["algorithm"] != "nr" or type(n["max_iteration"]) is not int or n["max_iteration"] < 1:
        raise ValueError("Stage 1 supports nr and a positive iteration count")
    buses = p["buses"]
    if (not buses or len(set(buses)) != len(buses)
            or any(type(b) is not int or not 2 <= b <= 33 for b in buses)):
        raise ValueError("PV bus numbers must be unique integers in [2,33]")
    if p["installed_to_base_load_ratio"] <= 0 or p["inverter_s_to_p_ratio"] < 1:
        raise ValueError("PV capacity must be positive and inverter S/P must be >= 1")
    if type(p["night_var_enabled"]) is not bool:
        raise ValueError("night_var_enabled must be boolean")
    if s["source"] != "synthetic_development_only":
        raise ValueError("Only the labelled synthetic development scenario is implemented")
    if type(s["seed"]) is not int or s["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(s["steps"]) is not int or s["steps"] < 2 or s["step_minutes"] <= 0:
        raise ValueError("Invalid time grid")
    if not np.isclose(s["steps"] * s["step_minutes"], 1440):
        raise ValueError("This generator requires one full 24-hour day")
    if not 0 <= s["sunrise_hour"] < s["sunset_hour"] <= 24:
        raise ValueError("Invalid sunrise/sunset")
    if not (0 < s["load_min"] <= s["load_max"] and s["load_noise_std"] >= 0):
        raise ValueError("Invalid load scales/noise")
    if not 0 <= s["pv_cloud_min"] <= s["pv_cloud_max"] <= 1:
        raise ValueError("Cloud factors must lie in [0,1]")
    if v["probe_bus"] not in buses or not 0 < v["probe_fraction_of_capability"] <= 1:
        raise ValueError("Invalid reactive probe")
    if (not v["candidate_buses"] or not set(v["candidate_buses"]).issubset(buses)
            or len(set(v["candidate_buses"])) != len(v["candidate_buses"])):
        raise ValueError("Invalid candidate PV buses")
    if not v["candidate_fractions"] or any(not 0 < f <= 1 for f in v["candidate_fractions"]):
        raise ValueError("Candidate fractions must lie in (0,1]")
    if v["stress_load_scale"] <= 0:
        raise ValueError("Stress load scale must be positive")
    return config


def make_day(config: dict) -> dict[str, np.ndarray]:
    """Each seed describes a complete day, not a train/validation/test split."""
    s = config["scenario"]
    rng = np.random.default_rng(s["seed"])
    hour = np.arange(s["steps"]) * s["step_minutes"] / 60.0
    load = s["load_min"] + (s["load_max"] - s["load_min"]) * (
        0.5 + 0.5 * np.cos(2 * np.pi * (hour - 20) / 24))
    load = np.clip(load + rng.normal(0, s["load_noise_std"], hour.size),
                   s["load_min"], s["load_max"])
    phase = (hour - s["sunrise_hour"]) / (s["sunset_hour"] - s["sunrise_hour"])
    solar = np.where((phase > 0) & (phase < 1), np.sin(np.pi * phase), 0.0)
    cloud = rng.uniform(s["pv_cloud_min"], s["pv_cloud_max"], hour.size)
    return {"hour": hour, "load_scale": load, "pv_fraction": solar * cloud}
