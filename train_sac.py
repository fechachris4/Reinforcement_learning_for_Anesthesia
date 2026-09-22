"""
Train SAC (or PID + SAC with --residual). Each episode is a new random patient.
Validation uses the PID tuning patients; the test patients are only used in benchmark.py.

    python train_sac.py --seed 0 --steps 500000 [--residual]

With --n-envs N, N patients are simulated in parallel and the network gets
--grad-steps updates per N environment steps. The simulator is cheap, so
fewer, batched updates are what make training fast.
"""

import argparse, json, os
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from AnesthesiaEnv import AnesthesiaEnv
from evaluate import run_episode, episode_metrics
from patients import tuning_patients
from residual import ResidualEnv, ResidualPolicy


class ValidationCallback(BaseCallback):
    def __init__(self, every, patients, log_path, best_path, residual=False):
        super().__init__()
        self.residual = residual
        self.every, self.patients, self.log_path, self.best_path = every, patients, log_path, best_path
        self.log = []
        self.best = -1.0
        self.next_eval = every

    def _on_step(self):
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.every
            ctrl = ResidualPolicy(self.model) if self.residual else self.model
            ms = [episode_metrics(run_episode(ctrl, p, 500 + i)) for i, p in enumerate(self.patients)]
            row = {'step': self.num_timesteps,
                   'time_in_target': float(np.mean([m['time_in_target'] for m in ms])),
                   'MDAPE': float(np.mean([m['MDAPE'] for m in ms]))}
            self.log.append(row)
            # keep the best checkpoint on validation
            if row['time_in_target'] > self.best:
                self.best = row['time_in_target']
                self.model.save(self.best_path)
            print(f"[{self.num_timesteps:>7}] val TiT={row['time_in_target']:.1f}%  MDAPE={row['MDAPE']:.1f}%", flush=True)
            with open(self.log_path, 'w') as f:
                json.dump(self.log, f)
        return True


def train(seed, steps, n_envs=1, grad_steps=1, out_dir='models', residual=False):
    torch.set_num_threads(1)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs('results', exist_ok=True)
    env = make_vec_env(ResidualEnv if residual else AnesthesiaEnv, n_envs=n_envs, seed=10_000 + 100 * seed)
    name = f"{'residual' if residual else 'sac'}_seed{seed}"
    # low exploration noise for the residual agent, otherwise early corrections swamp the PID
    extra = dict(ent_coef='auto_0.01', target_entropy=-4.0, learning_starts=2_000) if residual else dict(learning_starts=5_000)
    model = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=200_000, batch_size=256,
                gamma=0.995, tau=0.005, train_freq=1,
                gradient_steps=grad_steps, seed=seed, verbose=0, **extra)
    cb = ValidationCallback(20_000, tuning_patients(20), f'results/learning_curve_{name}.json',
                            f'{out_dir}/{name}', residual)
    model.learn(total_timesteps=steps, callback=cb)
    model.save(f'{out_dir}/{name}_final')
    return model


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=500_000)
    ap.add_argument('--n-envs', type=int, default=8)
    ap.add_argument('--grad-steps', type=int, default=2)
    ap.add_argument('--out', default='models')
    ap.add_argument('--residual', action='store_true', help='learn corrections on top of the PID')
    a = ap.parse_args()
    train(a.seed, a.steps, a.n_envs, a.grad_steps, a.out, a.residual)
