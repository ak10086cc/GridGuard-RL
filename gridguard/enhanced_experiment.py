"""Single-seed resumable residual SAC and matched-scenario evaluation entry point."""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import zipfile

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .__main__ import provenance
from .controllers import load_baseline_config
from .enhanced import EnhancedEnv, EnhancedPoolEnv, load_enhanced_config
from .environment import load_environment_config
from .evaluation import aggregate, write_csv
from .formal_experiment import CheckpointPause, RecoveryManager, digest_file, json_write, restore_checkpoint
from .learning import MeasuredSAC, actor_digest, load_sac_config, scenario_manifest
from .policy_evaluation import evaluate_policies
from .scenario import load_config

ROOT = Path(__file__).resolve().parent.parent


def evaluate_enhanced(grid, settings, baselines, enhanced, seeds, policy, *, mode, shield):
    label = f'{mode}_sac' + ('_screened' if shield else '_raw')
    summaries, steps, actions, signatures = [], [], [], {}
    for seed in seeds:
        env = EnhancedEnv(grid, settings, baselines, enhanced, mode=mode, shield=shield)
        obs, _ = env.reset(seed=seed)
        episode_rows, total, previous = [], 0., None
        signature = hashlib.sha256()
        while True:
            start = perf_counter()
            policy_start = perf_counter()
            action, _ = policy.predict(obs, deterministic=True)
            policy_seconds = perf_counter() - policy_start
            before = obs.copy()
            obs, reward, terminated, truncated, info = env.step(action)
            elapsed = perf_counter() - start
            valid = info['postcontrol_converged']
            q = np.asarray(info['executed_q_mvar']) if info['executed_q_mvar'] is not None else None
            voltage = np.asarray(info['post_voltage_pu']) if valid else None
            deviation = np.abs(voltage - grid['network']['reference_pu']) if valid else None
            delta = np.abs(q - previous) if previous is not None and q is not None and valid else None
            next_attempt = info['termination_reason'] not in ('day_complete', 'postcontrol_failure', 'enhanced_control_failure')
            final_attempt = info['termination_reason'] != 'enhanced_control_failure'
            row = {'scenario': f'seed_{seed}', 'method': label, 'step': info['step_index'], 'hour': info['hour'],
                   'load_scale': info['load_scale'], 'pv_total_mw': info['pv_total_mw'], 'converged': valid,
                   'controller_status': info['dsg_status'] or 'sac', 'algorithm_converged': valid and info['failure_stage'] != 'enhanced_control',
                   'vmin_pu': float(voltage.min()) if valid else None, 'vmax_pu': float(voltage.max()) if valid else None,
                   'violated_node_count': int(np.count_nonzero(env.model.voltage_violation(voltage))) if valid else None,
                   'mean_abs_deviation_pu': float(deviation.mean()) if valid else None,
                   'max_abs_deviation_pu': float(deviation.max()) if valid else None,
                   'line_loss_mw': info['line_loss_mw'], 'balance_error_mw': info['balance_error_mw'],
                   'mean_abs_q_mvar': float(np.abs(q).mean()) if q is not None else None,
                   'total_abs_q_mvar': float(np.abs(q).sum()) if q is not None else None,
                   'mean_abs_delta_q_mvar': float(delta.mean()) if delta is not None else None,
                   'max_abs_delta_q_mvar': float(delta.max()) if delta is not None else None,
                   'pf_calls': info['dsg_pf_calls'] + info['screen_pf_calls'] + int(final_attempt) + int(next_attempt),
                   'pf_attempt_failures': info['dsg_pf_failures'] + info['screen_pf_failures'] + int(final_attempt and not valid) + int(info['next_precontrol_failed']),
                   'decision_s': elapsed, 'final_pf_s': env.last_post_flow.elapsed_s if env.last_post_flow else None,
                   'policy_or_controller_s': policy_seconds + env.last_control_seconds, 'pure_policy_s': policy_seconds,
                   'screen_s': env.last_screen_seconds, 'dsg_pf_calls': info['dsg_pf_calls'], 'screen_pf_calls': info['screen_pf_calls'],
                   'screen_pf_failures': info['screen_pf_failures'], 'candidate': info['candidate'],
                   'capability_limited': info['capability_limited'],
                   'screen_voltage_feasible': info['screen_voltage_feasible'], 'intervention_reason': info['intervention_reason'],
                   'screen_attempts': json.dumps(info['screen_attempts'], ensure_ascii=False), 'reward': reward,
                   'next_precontrol_failed': info['next_precontrol_failed'], 'termination_reason': info['termination_reason'], 'error': info['failure_reason']}
            episode_rows.append(row); steps.append(row); total += reward
            for j, bus in enumerate(grid['pv']['buses']):
                actions.append({'scenario': f'seed_{seed}', 'method': label, 'step': info['step_index'], 'pv_bus': bus,
                                'policy_action': float(action[j]), 'dsg_q_mvar': info['dsg_q_mvar'][j] if info['dsg_q_mvar'] else None,
                                'raw_q_mvar': info['raw_q_mvar'][j], 'proposed_q_mvar': info['proposed_q_mvar'][j],
                                'executed_q_mvar': float(q[j]) if q is not None else None,
                                'qmax_mvar': info['qmax_mvar'][j], 'intervention_reason': info['intervention_reason']})
            signature.update(before.astype(np.float32).tobytes())
            signature.update(np.asarray(action, dtype=np.float32).tobytes())
            signature.update(json.dumps([reward, info['executed_q_mvar'], info['post_voltage_pu'], terminated, truncated], allow_nan=False).encode())
            previous = q.copy() if q is not None and valid else None
            if terminated or truncated:
                break
        summary = {'scenario': f'seed_{seed}', 'method': label,
                   **aggregate(episode_rows, env.horizon, grid['scenario']['step_minutes']),
                   'episode_complete': info['episode_complete'], 'episode_return': total,
                   'next_precontrol_failures': sum(x['next_precontrol_failed'] for x in episode_rows),
                   'enhanced_failures': sum(x['termination_reason'] == 'enhanced_control_failure' for x in episode_rows),
                   'screen_interventions': sum(x['candidate'] not in (None, 'proposed') for x in episode_rows),
                   'preexisting_violation_accepts': sum(x['candidate'] is not None and x['screen_voltage_feasible'] is False for x in episode_rows),
                   'mean_pure_policy_inference_s': float(np.mean([x['pure_policy_s'] for x in episode_rows])),
                   'trajectory_sha256': signature.hexdigest()}
        summaries.append(summary); signatures[f'{seed}/{label}'] = signature.hexdigest()
        env.close()
    return {'summaries': summaries, 'raw_steps': steps, 'raw_actions': actions, 'signatures': signatures,
            'step_minutes': grid['scenario']['step_minutes']}


class EnhancedValidation(BaseCallback):
    def __init__(self, grid, settings, baselines, enhanced, config, output, shield):
        super().__init__(verbose=0)
        self.grid, self.settings, self.baselines, self.enhanced = grid, settings, baselines, enhanced
        self.config, self.output, self.shield = config, output, shield
        self.validation_seconds = 0.
        self.validation_rows, self.episode_rows = [], []
        self.best_score, self.best_signatures, self.best_step, self._last_print_step = -np.inf, None, None, 0

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                self.episode_rows.append({'timesteps': self.num_timesteps, 'scenario_seed': info['scenario_seed'],
                    'episode_return': info['episode']['r'], 'length': info['episode']['l'],
                    'complete_episode': info['episode_complete'], 'failure_stage': info['failure_stage']})
        if self.num_timesteps % self.config['run']['validation_interval'] == 0:
            start = perf_counter()
            result = evaluate_enhanced(self.grid, self.settings, self.baselines, self.enhanced,
                                       self.config['scenarios']['validation_seeds'], self.model, mode='residual', shield=self.shield)
            score = float(np.mean([x['episode_return'] for x in result['summaries']]))
            row = {'timesteps': self.num_timesteps, 'updates_before_this_step_training': self.model._n_updates,
                   'mean_validation_return': score,
                   'complete_episodes': sum(x['episode_complete'] for x in result['summaries']),
                   'successful_steps': sum(x['successful_steps'] for x in result['summaries']),
                   'validation_episodes': len(result['summaries'])}
            previous = next((x for x in self.validation_rows if x['timesteps'] == self.best_step), None)
            rank = lambda x: (x['complete_episodes'], x['successful_steps'], x['mean_validation_return'])
            self.validation_rows.append(row)
            # Failure/coverage cannot be hidden by an attractive partial-episode return.
            # The row is part of the checkpointed callback state, so the same
            # selection rule is restored without an extra unsaved attribute.
            if previous is None or rank(row) > rank(previous):
                self.best_score, self.best_step, self.best_signatures = score, self.num_timesteps, result['signatures']
                self.model.save(self.output / 'best_model.zip')
            self.validation_seconds += perf_counter() - start
        return True


def train(grid, settings, baselines, enhanced, config, output, *, shield, resume=False, pause_at=None):
    output = Path(output)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    source = provenance()
    source['source_sha256']['run_enhanced.ps1'] = digest_file(ROOT / 'run_enhanced.ps1')
    run_data = {'grid': grid, 'environment': settings, 'baselines': baselines, 'enhanced': enhanced,
                'sac': config, 'shield': shield, 'source_sha256': source['source_sha256'], 'dependencies': source['dependencies']}
    digest = hashlib.sha256(json.dumps(run_data, sort_keys=True).encode()).hexdigest()
    if output.exists():
        if not resume:
            raise FileExistsError(f'Refusing to overwrite {output}')
        existing = json.loads((output / 'run_config.json').read_text(encoding='utf8'))
        if existing['run_digest'] != digest:
            raise ValueError('Recovery refused: source, configuration or dependencies changed')
        if (output / 'summary.json').exists():
            return json.loads((output / 'summary.json').read_text(encoding='utf8'))
        if not (output / 'last_checkpoint.zip').exists():
            raise FileNotFoundError('No recovery checkpoint; refusing to restart')
    else:
        if resume:
            raise FileNotFoundError('Resume requires an existing run directory')
        output.mkdir(parents=True)
        json_write(output / 'run_config.json', {'run_digest': digest, **run_data, 'provenance': source})
        with zipfile.ZipFile(output / 'source_snapshot.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in source['source_sha256']:
                archive.write(ROOT / name, name)
            for path in sorted((ROOT / 'configs').glob('*.toml')):
                archive.write(path, str(path.relative_to(ROOT)))
    callback = EnhancedValidation(grid, settings, baselines, enhanced, config, output, shield)
    if resume:
        model, prior = restore_checkpoint(output / 'last_checkpoint.zip', callback, config, digest)
    else:
        env = EnhancedPoolEnv(grid, settings, baselines, enhanced, config['scenarios']['train_seeds'], config['run']['seed'], shield=shield)
        s = config['sac']
        model = MeasuredSAC('MlpPolicy', Monitor(env), seed=config['run']['seed'], device='cpu', verbose=0,
                            learning_rate=s['learning_rate'], buffer_size=s['buffer_size'], learning_starts=s['learning_starts'],
                            batch_size=s['batch_size'], tau=s['tau'], gamma=s['gamma'], train_freq=1, gradient_steps=1,
                            ent_coef='auto', policy_kwargs={'net_arch': s['net_arch']})
        model.update_rows, model.log_every_updates = [], s['log_every_updates']
        prior = 0.
    manager = RecoveryManager(output, callback, config, digest, prior, pause_at)
    model._after_update_hook = manager
    before = actor_digest(model)
    remaining = config['run']['total_timesteps'] - model.num_timesteps
    start = perf_counter()
    try:
        if remaining:
            model.learn(total_timesteps=remaining, reset_num_timesteps=model.num_timesteps == 0,
                        callback=callback, log_interval=None, progress_bar=False)
    except CheckpointPause:
        model.get_env().close()
        return {'status': 'paused_at_checkpoint', 'timesteps': model.num_timesteps}
    except Exception as exc:
        manager.progress(model, 'failed')
        json_write(output / 'failure.json', {'error': repr(exc), 'timesteps': model.num_timesteps})
        model.get_env().close()
        raise
    learn_elapsed = perf_counter() - start
    manager.save(model)
    if callback.best_step is None:
        raise RuntimeError('No validation checkpoint selected')
    best = SAC.load(output / 'best_model.zip', device='cpu')
    verified = evaluate_enhanced(grid, settings, baselines, enhanced, config['scenarios']['validation_seeds'], best, mode='residual', shield=shield)
    if verified['signatures'] != callback.best_signatures:
        raise AssertionError('Best checkpoint reload validation mismatch')
    result = {'status': 'training_complete', 'seed': config['run']['seed'], 'shield': shield,
              'timesteps': model.num_timesteps, 'gradient_updates': model._n_updates, 'replay_size': model.replay_buffer.size(),
              'best_validation_step': callback.best_step, 'best_validation_return': callback.best_score,
              'best_model_sha256': digest_file(output / 'best_model.zip'), 'best_reload_trajectory_exact': True,
              'selection_rule': 'validation complete episodes, then successful steps, then mean return; earlier wins ties',
              'complete_training_episodes': sum(x['complete_episode'] for x in callback.episode_rows),
              'failed_training_episodes': sum(not x['complete_episode'] for x in callback.episode_rows),
              'actor_changed_this_process': actor_digest(model) != before if remaining else None,
              'learn_wall_s_this_process': learn_elapsed, 'active_elapsed_s': prior + perf_counter() - manager.start,
              'validation_seconds': callback.validation_seconds, 'test_performance_evaluated_during_training': False,
              'run_digest': digest}
    json_write(output / 'summary.json', result)
    manager.progress(model, 'training_complete')
    model.get_env().close()
    return result


def write_union(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    write_csv(path, [{k: row.get(k) for k in columns} for row in rows])


def evaluate_comparison(grid, settings, baselines, enhanced, seeds, sac_models, residual_models, output, *, formal_scenarios=None):
    """Same cases/metrics; compare each policy with and without the same screening."""
    if len(sac_models) != len(residual_models) or not sac_models:
        raise ValueError('Provide paired SAC and residual models')
    checked = []
    training_shields = {}
    for sac_path, residual_path in zip(sac_models, residual_models):
        pair = []
        for path in (Path(sac_path), Path(residual_path)):
            meta = json.loads((path.parent / 'summary.json').read_text(encoding='utf8'))
            conf = json.loads((path.parent / 'run_config.json').read_text(encoding='utf8'))
            if (meta['status'] != 'training_complete' or not meta['best_reload_trajectory_exact']
                    or digest_file(path) != meta['best_model_sha256']):
                raise ValueError(f'Model is not the validated best checkpoint: {path}')
            if conf['grid'] != grid or conf['environment'] != settings or conf['baselines'] != baselines:
                raise ValueError(f'Model uses a different grid/environment/baselines: {path}')
            if formal_scenarios is not None and (meta['timesteps'] != enhanced['training']['total_timesteps']
                                                  or conf['sac']['scenarios'] != formal_scenarios):
                raise ValueError(f'Test evaluation requires fully trained models and the fixed scenario split: {path}')
            pair.append((meta, conf))
        if pair[0][0]['seed'] != pair[1][0]['seed']:
            raise ValueError('Pair direct and residual models by the same training seed')
        if pair[1][1]['enhanced'] != enhanced:
            raise ValueError('Residual training and evaluation enhancement settings differ')
        seed = pair[0][0]['seed']
        checked.append(seed)
        training_shields[str(seed)] = pair[1][1]['shield']
    if len(set(checked)) != len(checked):
        raise ValueError('Duplicate model seed')
    if len(set(training_shields.values())) != 1:
        raise ValueError('Residual models must use the same training shield setting')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    all_results = []
    deterministic = evaluate_policies(grid, settings, baselines, seeds, ['unity_pf', 'volt_var', 'dsg'], retain_raw=True)
    all_results.append(('deterministic', deterministic))
    hashes = {'sac': [], 'residual': []}
    for seed, (sac_path, residual_path) in zip(checked, zip(sac_models, residual_models)):
        hashes['sac'].append(digest_file(Path(sac_path)))
        hashes['residual'].append(digest_file(Path(residual_path)))
        direct, residual = SAC.load(sac_path, device='cpu'), SAC.load(residual_path, device='cpu')
        raw = evaluate_policies(grid, settings, baselines, seeds, ['sac'], model=direct, retain_raw=True)
        for collection in ('summaries', 'raw_steps', 'raw_actions'):
            for row in raw[collection]: row['method'] = 'direct_sac_raw'
        all_results.append((str(seed), raw))
        all_results.append((str(seed), evaluate_enhanced(grid, settings, baselines, enhanced, seeds, direct, mode='direct', shield=True)))
        for shield in (False, True):
            all_results.append((str(seed), evaluate_enhanced(grid, settings, baselines, enhanced, seeds, residual, mode='residual', shield=shield)))
    per_case, per_run, step_rows, action_rows = [], [], [], []
    for label, result in all_results:
        for item in result['summaries']:
            training_shield = (training_shields[label] if item['method'].startswith('residual_sac_') else
                               False if item['method'].startswith('direct_sac_') else None)
            per_case.append({'run': label, 'training_shield': training_shield, **item})
        for method in dict.fromkeys(x['method'] for x in result['raw_steps']):
            subset = [x for x in result['raw_steps'] if x['method'] == method]
            cases = [x for x in result['summaries'] if x['method'] == method]
            training_shield = (training_shields[label] if method.startswith('residual_sac_') else
                               False if method.startswith('direct_sac_') else None)
            per_run.append({'run': label, 'method': method, 'training_shield': training_shield,
                            **aggregate(subset, sum(x['planned_steps'] for x in cases), result['step_minutes']),
                            'episode_completion_rate': float(np.mean([x['episode_complete'] for x in cases])),
                            'safe_full_episode_rate': float(np.mean([x['full_horizon_no_failure_no_violation'] for x in cases])),
                            'next_precontrol_failures': sum(x['next_precontrol_failures'] for x in cases)})
        step_rows.extend({'run': label, **x} for x in result['raw_steps'])
        action_rows.extend({'run': label, **x} for x in result['raw_actions'])
    metrics = ('voltage_violation_rate', 'mean_abs_deviation_pu', 'max_abs_deviation_pu', 'mean_line_loss_mw',
               'line_loss_mwh_valid_steps', 'mean_abs_q_mvar', 'mean_abs_delta_q_mvar', 'final_step_failure_rate',
               'valid_sample_coverage', 'episode_completion_rate', 'safe_full_episode_rate', 'pf_attempt_failures',
               'next_precontrol_failures', 'mean_decision_s')
    averages = []
    for method in dict.fromkeys(x['method'] for x in per_run):
        records = [x for x in per_run if x['method'] == method]
        row = {'method': method, 'model_count': len(records) if method not in ('unity_pf','volt_var','dsg') else None}
        for metric in metrics:
            values = [x[metric] for x in records if x[metric] is not None]
            row[metric + '_mean'] = float(np.mean(values)) if values else None
            row[metric + '_seed_std'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        averages.append(row)
    for name, records in (('per_scenario.csv', per_case), ('per_model.csv', per_run), ('aggregate.csv', averages),
                          ('steps.csv', step_rows), ('actions.csv', action_rows)):
        write_union(output / name, records)
    report = {'status': 'complete', 'scenario_seeds': list(seeds), 'sac_models': list(map(str, sac_models)),
              'residual_models': list(map(str, residual_models)), 'training_seeds': checked,
              'residual_training_shield_by_seed': training_shields, 'model_sha256': hashes,
              'per_scenario': per_case, 'per_model': per_run, 'aggregate': averages,
              'std_definition': 'sample SD across supplied paired models, after pooling whole scenarios; baselines evaluated once',
              'screen_rule': enhanced['control'], 'provenance': provenance(),
              'limits': 'Screening can accept preexisting voltage violations without worsening; not a strict safety guarantee.'}
    json_write(output / 'summary.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description='Residual SAC training and matched scenario comparison')
    parser.add_argument('task', choices=('train', 'evaluate'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--shield', choices=('on','off'), default='on')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stop-after', type=int, help='Testing only: pause immediately after this update')
    parser.add_argument('--split', choices=('validation','test'), default='validation')
    parser.add_argument('--sac-model', action='append', default=[])
    parser.add_argument('--residual-model', action='append', default=[])
    args = parser.parse_args()
    if args.stop_after is not None and not args.smoke:
        parser.error('--stop-after is restricted to the smoke workflow')
    if args.seed < 0:
        parser.error('seed must be nonnegative')
    output = Path(args.output).resolve()
    if args.smoke and not output.is_relative_to(ROOT / 'tmp'):
        parser.error('Smoke artifacts must be under tmp/')
    if not args.smoke and not output.is_relative_to(ROOT / 'results'):
        parser.error('Formal artifacts must be under results/')
    grid_config = load_config(ROOT / 'configs/ieee33.toml')
    settings = load_environment_config(ROOT / 'configs/environment.toml')
    baselines = load_baseline_config(ROOT / 'configs/baselines.toml')
    enhanced = load_enhanced_config(ROOT / 'configs/enhanced.toml')
    config = load_sac_config(ROOT / 'configs/sac_formal.toml', formal=True)
    grid, manifest = scenario_manifest(grid_config, config)
    torch.set_num_threads(1)
    if args.task == 'train':
        if args.sac_model or args.residual_model:
            parser.error('Training starts a residual policy, not a loaded evaluation model')
        training = enhanced['training']
        config['run'].update(seed=args.seed, total_timesteps=training['total_timesteps'], validation_interval=training['validation_interval'])
        config['sac'].update({k: training[k] for k in ('learning_starts','buffer_size','batch_size','learning_rate','tau','gamma','net_arch','log_every_updates')})
        config['formal'].update(checkpoint_interval=training['checkpoint_interval'], progress_interval=training['progress_interval'])
        if args.smoke:
            config['run'].update(total_timesteps=32, validation_interval=32)
            config['sac'].update(learning_starts=8, buffer_size=128, batch_size=8, net_arch=[16,16], log_every_updates=8)
            config['formal'].update(checkpoint_interval=16, progress_interval=8)
            config['scenarios']['validation_seeds'] = [2001]
        result = train(grid, settings, baselines, enhanced, config, output, shield=args.shield == 'on', resume=args.resume, pause_at=args.stop_after)
        print(json.dumps({'output': str(output), **result}, ensure_ascii=False, indent=2))
    else:
        if args.resume or args.stop_after is not None:
            parser.error('evaluate does not accept resume or stop-after')
        if not args.sac_model or len(args.sac_model) != len(args.residual_model):
            parser.error('provide equal numbers of --sac-model and --residual-model')
        if args.split == 'test' and len(args.sac_model) < 3:
            parser.error('Independent test comparison requires at least three paired training seeds')
        seeds = config['scenarios'][f'{args.split}_seeds']
        if args.smoke:
            if args.split != 'validation': parser.error('Smoke evaluation may use validation only')
            seeds = seeds[:1]
        result = evaluate_comparison(grid, settings, baselines, enhanced, seeds, args.sac_model, args.residual_model, output,
                                     formal_scenarios=config['scenarios'] if args.split == 'test' else None)
        print(json.dumps({'output': str(output), 'status': result['status'], 'split': args.split}, ensure_ascii=False))


if __name__ == '__main__':
    main()
