import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

# Set before importing numerical/plotting dependencies, even without run.ps1.
import os
ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "tmp"
TMP.mkdir(exist_ok=True)
sys.dont_write_bytecode = True
for name, relative in (("MPLCONFIGDIR", "matplotlib"), ("NUMBA_CACHE_DIR", "numba")):
    os.environ[name] = str(TMP / relative)

from .scenario import load_config
from .verification import physical_checks, run_day


def provenance():
    def installed(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    files = [*sorted((ROOT / "gridguard").glob("*.py")), *sorted((ROOT / "tests").glob("*.py")),
             ROOT / "pyproject.toml", ROOT / "run.ps1", ROOT / "setup.cfg", ROOT / "requirements-lock.txt"]
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {name: installed(name) for name in ("pandapower", "numpy", "pandas", "scipy", "matplotlib", "pytest", "gymnasium", "torch", "stable-baselines3")},
            "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


def main():
    parser = argparse.ArgumentParser(description="GridGuard simulation, baseline evaluation and SAC smoke workflow")
    parser.add_argument("task", choices=["info", "check", "simulate", "baselines", "env-check", "sac-smoke", "sac-evaluate", "sac-formal", "demo"])
    parser.add_argument("--config", default="configs/ieee33.toml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--baseline-config", default="configs/baselines.toml")
    parser.add_argument("--environment-config", default="configs/environment.toml")
    parser.add_argument("--sac-config", default="configs/sac.toml")
    parser.add_argument("--model")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.task == "info":
        print(json.dumps(provenance(), ensure_ascii=False, indent=2))
        return
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    baseline_path = Path(args.baseline_config).resolve()
    if args.task in ("baselines", "env-check", "sac-smoke", "sac-evaluate", "sac-formal", "demo"):
        from .controllers import load_baseline_config
        baseline_config = load_baseline_config(baseline_path)
    environment_path = Path(args.environment_config).resolve()
    if args.task in ("env-check", "sac-smoke", "sac-evaluate", "sac-formal", "demo"):
        from .environment import load_environment_config
        environment_config = load_environment_config(environment_path)
    sac_path = Path(args.sac_config).resolve()
    if args.task in ("sac-smoke", "sac-evaluate", "sac-formal", "demo"):
        from .learning import load_sac_config
        sac_config = load_sac_config(sac_path, formal=args.task in ("sac-formal", "demo") or args.task == "sac-evaluate" and "formal" in sac_path.stem)
        if args.task == "sac-evaluate" and (not args.model or not Path(args.model).is_file()):
            parser.error("sac-evaluate requires an existing --model checkpoint")
    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    default = ROOT / ("results" if args.task == "check" else "tmp") / f"stage01_{args.task}_seed{config['scenario']['seed']}_{suffix}"
    if args.task == "baselines":
        default = ROOT / "results" / f"stage02_baselines_{suffix}"
    if args.task == "env-check":
        default = ROOT / "results" / f"stage03_environment_{suffix}"
    if args.task in ("sac-smoke", "sac-evaluate"):
        default = TMP / f"stage04_{args.task}_seed{sac_config['run']['seed']}_{suffix}"
    if args.task == "sac-formal":
        default = ROOT / "results" / f"sac_formal_{suffix}"
    if args.task == "demo":
        default = TMP / f"demo_seed42_{suffix}"
    output = Path(args.output).resolve() if args.output else default
    if args.task == "sac-smoke" and not output.is_relative_to(TMP):
        parser.error("This smoke stage stores disposable outputs under project tmp/ only")
    if args.task == "demo" and not output.is_relative_to(TMP):
        parser.error("Demo outputs must be under project tmp/")
    # No accidental overwrites, including a directory left by a failed earlier run.
    output.mkdir(parents=True, exist_ok=args.resume and args.task == "sac-formal")
    if args.task == "demo":
        from .demo import run_demo
        model_path = Path(args.model) if args.model else ROOT / "results/sac_resume_20261029/sac_seed7_20261028/best_model.zip"
        result = run_demo(config, environment_config, baseline_config, sac_config, model_path, output)
        print(json.dumps({"output": str(output), **result}, ensure_ascii=False, indent=2))
        return
    if args.task == "sac-formal":
        from .formal_experiment import run_formal
        result = run_formal(config, environment_config, baseline_config, sac_config, output, provenance(), args.resume)
        print(json.dumps({"output": str(output), "status": result["status"]}, ensure_ascii=False))
        return
    shutil.copyfile(config_path, output / "run_config.toml")
    report = {"provenance": provenance(), "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()}
    if args.task in ("sac-smoke", "sac-evaluate"):
        for path, name in ((baseline_path, "baseline_config"), (environment_path, "environment_config"), (sac_path, "sac_config")):
            shutil.copyfile(path, output / f"{name}.toml")
            report[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        if args.task == "sac-smoke":
            from .learning import run_sac_smoke
            report["sac_smoke"] = run_sac_smoke(config, environment_config, baseline_config, sac_config, output)
        else:
            import torch
            from stable_baselines3 import SAC
            from .learning import scenario_manifest
            from .policy_evaluation import evaluate_policies
            torch.set_num_threads(sac_config["run"]["torch_threads"])
            grid, manifest = scenario_manifest(config, sac_config)
            model = SAC.load(args.model, device="cpu")
            report["checkpoint_sha256"] = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
            report["split"] = args.split
            report["evaluation"] = evaluate_policies(grid, environment_config, baseline_config,
                sac_config["scenarios"][f"{args.split}_seeds"], ["unity_pf", "volt_var", "dsg", "sac"], model, output)
            if not all(x["episode_complete"] for x in report["evaluation"]["summaries"]):
                raise RuntimeError("Evaluation did not complete planned episodes")
        (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        details = report.get("sac_smoke", {})
        print(json.dumps({"output": str(output), "timings": details.get("timings"), "formal_budget_proposal": details.get("formal_budget_proposal")}, ensure_ascii=False, indent=2))
        return
    if args.task == "env-check":
        from .environment_check import verify_environment
        for path, name in ((baseline_path, "baseline_config"), (environment_path, "environment_config")):
            shutil.copyfile(path, output / f"{name}.toml")
            report[f"{name}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        report["environment"] = verify_environment(config, environment_config, baseline_config, output)
        (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "environment": report["environment"]}, ensure_ascii=False, indent=2))
        if (not all(e["complete_episode"] and e["baseline_algorithm_nonconverged"] == 0 for e in report["environment"]["episodes"])
                or not all(e["full_day_exact_repeat"] for e in report["environment"]["determinism_checks"])):
            raise RuntimeError("Environment integration verification failed; inspect summary.json")
        return
    if args.task == "baselines":
        from .evaluation import evaluate_baselines
        shutil.copyfile(baseline_path, output / "baseline_config.toml")
        report["baseline_config_sha256"] = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
        report["baselines"] = evaluate_baselines(config, baseline_config, output)
        (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "summaries": report["baselines"]["summaries"]}, ensure_ascii=False, indent=2))
        if not all(r["full_day_repeat_matches"] for r in report["baselines"]["determinism_checks"]):
            raise RuntimeError("Determinism check failed; inspect baseline summary")
        return
    if args.task == "check":
        report["physical_checks"] = physical_checks(config)
    report["day"] = run_day(config, output)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "day": report["day"]}, ensure_ascii=False, indent=2))
    if not report["day"]["complete_episode"]:
        raise RuntimeError("Complete-day verification failed; see summary.json and day_metrics.csv")


if __name__ == "__main__":
    main()
