"""Grid search for the PID gains on the tuning patients."""

import itertools, json, os
import numpy as np

from pid_baseline import PIDController
from evaluate import run_episode, episode_metrics
from patients import tuning_patients


def score(ctrl, pats):
    ms = [episode_metrics(run_episode(ctrl, p, 500 + i)) for i, p in enumerate(pats)]
    return np.mean([m['time_in_target'] for m in ms]), np.mean([m['MDAPE'] for m in ms])


if __name__ == '__main__':
    pats = tuning_patients(20)
    grid = itertools.product([0.15, 0.2, 0.25, 0.3, 0.4, 0.5], [0.01, 0.02, 0.04, 0.06, 0.08, 0.12], [0.0, 0.002, 0.005, 0.01, 0.02])
    best = None
    for u0, kp, ki in grid:
        tit, mdape = score(PIDController(u0, kp, ki), pats)
        key = (tit, -mdape)
        if best is None or key > best[0]:
            best = (key, {'u0': u0, 'kp': kp, 'ki': ki})
            print(f"new best u0={u0} kp={kp} ki={ki}: TiT={tit:.1f}% MDAPE={mdape:.1f}%")
    os.makedirs('results', exist_ok=True)
    with open('results/pid_gains.json', 'w') as f:
        json.dump(best[1], f, indent=2)
    print('saved', best[1])
