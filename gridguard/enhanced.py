"""DSG residual and bounded current-state AC screening, without safety guarantees."""
from copy import deepcopy
from time import perf_counter
import tomllib

import numpy as np

from .controllers import control
from .environment import GridGuardEnv
from .network import action_to_q


def load_enhanced_config(path):
    with open(path, 'rb') as stream:
        config = tomllib.load(stream)
    c = config['control']
    if not np.isfinite(c['residual_scale']) or not 0 <= c['residual_scale'] <= 1:
        raise ValueError('residual_scale must be in [0, 1]')
    if type(c['max_candidate_powerflows']) is not int or not 1 <= c['max_candidate_powerflows'] <= 5:
        raise ValueError('max_candidate_powerflows must be an integer in [1, 5]')
    if not np.isfinite(c['voltage_tolerance_pu']) or c['voltage_tolerance_pu'] < 0:
        raise ValueError('Invalid voltage tolerance')
    if type(c['allow_nonworsening_when_infeasible']) is not bool:
        raise ValueError('allow_nonworsening_when_infeasible must be boolean')
    t = config['training']
    counts = ('total_timesteps', 'validation_interval', 'checkpoint_interval', 'progress_interval',
              'learning_starts', 'buffer_size', 'batch_size', 'log_every_updates')
    if any(type(t[k]) is not int or t[k] < 1 for k in counts):
        raise ValueError('Training counts must be positive integers')
    if (t['learning_starts'] >= t['total_timesteps'] or t['batch_size'] > min(t['buffer_size'], t['learning_starts'])
            or t['total_timesteps'] % t['validation_interval'] or t['checkpoint_interval'] <= t['learning_starts']):
        raise ValueError('Invalid training schedule or replay buffer')
    if (not np.isfinite([t['learning_rate'], t['tau'], t['gamma']]).all() or t['learning_rate'] <= 0
            or not 0 < t['tau'] <= 1 or not 0 < t['gamma'] <= 1
            or not t['net_arch'] or any(type(x) is not int or x < 1 for x in t['net_arch'])):
        raise ValueError('Invalid SAC hyperparameters')
    return config


def severity(model, flow):
    violation = model.voltage_violation(flow.voltage_pu)
    return float(np.max(violation)), float(np.sum(violation))


def screen_candidates(model, inputs, reference_flow, candidates, settings):
    """Each candidate, including previous-action fallback, gets a fresh current-state solve."""
    start = perf_counter()
    reference = severity(model, reference_flow)
    tol = settings['voltage_tolerance_pu']
    attempts, failures = [], 0
    for label, q in candidates[:settings['max_candidate_powerflows']]:
        q = np.clip(np.asarray(q, dtype=float), -inputs['qmax_mvar'], inputs['qmax_mvar'])
        model.set_operating_point(inputs['load_scale'], inputs['p_mw'], q)
        flow = model.solve()
        if not flow.converged:
            failures += 1
            attempts.append({'candidate': label, 'q_mvar': q.tolist(), 'converged': False, 'accepted': False, 'error': flow.error})
            continue
        max_v, sum_v = severity(model, flow)
        feasible = max_v <= tol
        nonworsening = (settings['allow_nonworsening_when_infeasible'] and reference[0] > tol
                        and max_v <= reference[0] + tol and sum_v <= reference[1] + tol)
        accepted = feasible or nonworsening
        attempts.append({'candidate': label, 'q_mvar': q.tolist(), 'converged': True, 'accepted': bool(accepted),
                         'max_violation_pu': max_v, 'sum_violation_pu': sum_v, 'voltage_feasible': bool(feasible)})
        if accepted:
            return {'q': q, 'flow': flow, 'candidate': label, 'voltage_feasible': bool(feasible),
                    'attempts': attempts, 'pf_failures': failures, 'elapsed_s': perf_counter() - start,
                    'reference_max_violation_pu': reference[0], 'reference_sum_violation_pu': reference[1]}
    return {'q': None, 'flow': None, 'candidate': None, 'voltage_feasible': False,
            'attempts': attempts, 'pf_failures': failures, 'elapsed_s': perf_counter() - start,
            'reference_max_violation_pu': reference[0], 'reference_sum_violation_pu': reference[1]}


class EnhancedEnv(GridGuardEnv):
    def __init__(self, grid, settings, baselines, enhanced, *, mode='residual', shield=True):
        super().__init__(grid, settings)
        if mode not in ('residual', 'direct'):
            raise ValueError('mode must be residual or direct')
        self.baselines = deepcopy(baselines)
        self.enhanced = deepcopy(enhanced)
        self.mode, self.shield = mode, bool(shield)
        self.last_control_seconds = self.last_screen_seconds = 0.

    def step(self, action):
        if self._needs_reset:
            raise RuntimeError('Call reset before step or after episode completion')
        if np.iscomplexobj(action):
            raise ValueError('Actions must be real-valued')
        raw = np.asarray(action, dtype=float)
        inputs = self.current_inputs
        # Validate before touching the grid.
        direct_q = action_to_q(raw, inputs['qmax_mvar'])
        start = perf_counter()
        chosen = None
        baseline_q = np.zeros(self.n_pv)
        if self.mode == 'residual':
            chosen = control('dsg', self.model, inputs['load_scale'], inputs['p_mw'], self.baselines)
            baseline_q = chosen.q_mvar.copy()
            raw_q = baseline_q + self.enhanced['control']['residual_scale'] * direct_q
            proposal = np.clip(raw_q, -inputs['qmax_mvar'], inputs['qmax_mvar'])
        else:
            raw_q = proposal = direct_q
        capability_limited = bool(np.any(np.abs(raw_q - proposal) > 1e-12))
        base_failed = chosen is not None and (not chosen.flow.converged or not chosen.algorithm_converged)
        screening = None
        if self.shield and not base_failed:
            fallback = np.clip(self.previous_q, -inputs['qmax_mvar'], inputs['qmax_mvar'])
            # Raw, two bounded corrections toward DSG (or zero for direct SAC),
            # baseline, then previous executed action; all are individually checked.
            candidates = [('proposed', proposal), ('half_residual', .5 * (proposal + baseline_q)),
                          ('quarter_residual', .25 * proposal + .75 * baseline_q),
                          ('baseline', baseline_q), ('previous', fallback)]
            screening = screen_candidates(self.model, inputs, self.pre_flow, candidates, self.enhanced['control'])
        executed = None if base_failed or (screening is not None and screening['q'] is None) else (
            screening['q'] if screening is not None else proposal)
        self.last_screen_seconds = screening['elapsed_s'] if screening else 0.
        self.last_control_seconds = perf_counter() - start
        extra = {'policy_action': raw.tolist(), 'dsg_q_mvar': baseline_q.tolist() if chosen else None,
                 'residual_scale': self.enhanced['control']['residual_scale'] if chosen else None,
                 'raw_q_mvar': raw_q.tolist(), 'proposed_q_mvar': proposal.tolist(),
                 'capability_limited': capability_limited, 'shield_enabled': self.shield,
                 'dsg_status': chosen.status if chosen else None,
                 'dsg_pf_calls': chosen.pf_calls if chosen else 0,
                 'dsg_pf_failures': chosen.pf_failures if chosen else 0,
                 'screen_pf_calls': len(screening['attempts']) if screening else 0,
                 'screen_pf_failures': screening['pf_failures'] if screening else 0,
                 'screen_attempts': screening['attempts'] if screening else [],
                 'candidate': screening['candidate'] if screening else None,
                 'screen_voltage_feasible': screening['voltage_feasible'] if screening else None,
                 'reference_max_violation_pu': screening['reference_max_violation_pu'] if screening else None,
                 'intervention_reason': ('dsg_failure' if base_failed else
                     'all_candidates_rejected' if executed is None else
                     'candidate_rejected_or_corrected' if screening and screening['candidate'] != 'proposed' else
                     'preexisting_voltage_violation' if screening and not screening['voltage_feasible'] else
                     'capability_limit' if capability_limited else None)}
        if executed is None:
            # No invented executed Q or physical result. Explicit terminal failure.
            self.elapsed_steps += 1
            self._needs_reset = True
            self.last_post_flow = None
            penalty = -float(self.settings['reward']['failure_penalty'])
            self.episode_return += penalty
            info = {'step_index': self.t, 'hour': float(self.day['hour'][self.t]), 'scenario_seed': self.scenario_seed,
                    'load_scale': inputs['load_scale'], 'pv_total_mw': float(inputs['p_mw'].sum()),
                    'raw_action': raw.tolist(), 'executed_q_mvar': None, 'qmax_mvar': inputs['qmax_mvar'].tolist(),
                    'previous_q_mvar': self.previous_q.tolist(), 'postcontrol_converged': False,
                    'post_voltage_pu': None, 'line_loss_mw': None, 'balance_error_mw': None,
                    'next_precontrol_failed': False, 'failure_stage': 'enhanced_control',
                    'failure_reason': extra['intervention_reason'], 'observation_valid': False,
                    'episode_complete': False, 'termination_reason': 'enhanced_control_failure',
                    'reward_raw': {k: None for k in ('voltage','loss','violation','movement')},
                    'reward_contributions': {'voltage': 0., 'loss': 0., 'violation': 0., 'movement': 0., 'failure': penalty},
                    'controllable': bool(np.any(inputs['qmax_mvar'] > 0)), **extra}
            return self._absorbing_observation(), penalty, True, False, info
        normalized = np.divide(executed, inputs['qmax_mvar'], out=np.zeros_like(executed), where=inputs['qmax_mvar'] > 0)
        obs, reward, terminated, truncated, info = super().step(np.clip(normalized, -1, 1))
        info.update(extra)
        info['raw_action'] = raw.tolist()
        return obs, reward, terminated, truncated, info


class EnhancedPoolEnv(EnhancedEnv):
    def __init__(self, grid, settings, baselines, enhanced, seeds, seed, *, shield):
        super().__init__(grid, settings, baselines, enhanced, shield=shield)
        self.pool, self.sampler, self.order = tuple(seeds), np.random.default_rng(seed), []

    def reset(self, *, seed=None, options=None):
        if options and 'scenario_seed' in options:
            raise ValueError('Training scenes must use the configured training pool')
        if seed is not None:
            self.sampler, self.order = np.random.default_rng(seed), []
        if not self.order:
            self.order = self.sampler.permutation(self.pool).tolist()
        return super().reset(seed=seed, options={**(options or {}), 'scenario_seed': self.order.pop()})
