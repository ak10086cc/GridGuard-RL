"""Small, reproducible inference demo on a reserved synthetic development day."""
import hashlib
import json
from time import perf_counter

import torch
from stable_baselines3 import SAC

from .learning import scenario_manifest
from .policy_evaluation import evaluate_policies


def run_demo(grid_config, environment, baselines, config, model_path, output):
    if not model_path.is_file():
        raise FileNotFoundError(f'Demo model missing: {model_path}')
    grid, _ = scenario_manifest(grid_config, config)
    if 42 not in config['scenarios']['reserved_development_seeds']:
        raise ValueError('Demo seed 42 must remain outside training, validation, and test pools')
    torch.set_num_threads(1)
    start = perf_counter()
    model = SAC.load(model_path, device='cpu')
    report = evaluate_policies(grid, environment, baselines, [42], ['unity_pf', 'sac'], model, output, retain_raw=True)
    results = report['summaries']
    result = {'scope': 'One 96-step synthetic development day; illustration only, not the held-out test result',
              'scenario_seed': 42, 'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'evaluation_s': perf_counter() - start, 'summaries': results}
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    if not all(x['episode_complete'] for x in results):
        raise RuntimeError('Demo episode incomplete; inspect summary and evaluation_steps.csv')
    return result
