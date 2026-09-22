"""
Benchmark three controllers on 30 held-out patients:
  PID          tuned clinical-style baseline
  SAC          pure reinforcement learning (3 seeds)
  PID + SAC    residual RL: SAC learns corrections on top of the PID (3 seeds)

    python benchmark.py
Writes results/benchmark.json, results/benchmark.md and figures in media/.
"""

import glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from evaluate import evaluate, run_episode, TEST_NOISE_OFFSET
from patients import test_patients, describe
from pid_baseline import load_pid
from residual import ResidualPolicy

from style import COLORS as C, TARGET as C_TARGET
MUTED = '#6B7280'
ROWS = [('time_in_target', 'Time in target, BIS 40-60 (%)'),
        ('MDAPE', 'MDAPE, inaccuracy (%)'),
        ('MDPE', 'MDPE, bias (%)'),
        ('wobble', 'Wobble (%)'),
        ('time_below_40', 'Time too deep, BIS < 40 (%)'),
        ('min_bis', 'Lowest BIS reached'),
        ('induction_min', 'Induction time (min)'),
        ('propofol_mgkgh', 'Maintenance propofol (mg/kg/h)')]


def load(kind, path):
    m = SAC.load(path, device='cpu')
    return ResidualPolicy(m) if kind == 'residual' else m


def seeds_of(kind):
    return sorted(int(p.split('seed')[1].split('.')[0]) for p in glob.glob(f'models/{kind}_seed[0-9].zip'))


def main():
    os.makedirs('results', exist_ok=True); os.makedirs('media', exist_ok=True)
    res = {'PID': {0: evaluate(load_pid())}}
    for label, kind in (('SAC', 'sac'), ('PID + SAC', 'residual')):
        res[label] = {s: evaluate(load(kind, f'models/{kind}_seed{s}.zip')) for s in seeds_of(kind)}
        for s, (summ, _) in res[label].items():
            print(f'{label} seed {s}: time in target {summ["time_in_target"]:.1f}%')

    names = [n for n in ('PID', 'SAC', 'PID + SAC') if res[n]]
    head = '| Metric | ' + ' | '.join(f'{n} ({len(res[n])} seed{"s" if len(res[n]) > 1 else ""})' if n != 'PID' else 'PID (tuned)' for n in names) + ' |'
    lines = [head, '|' + '---|' * (len(names) + 1)]
    for k, label in ROWS:
        cells = []
        for n in names:
            v = np.array([res[n][s][0][k] for s in res[n]])
            cells.append(f'{v.mean():.1f}' if len(v) == 1 else f'{v.mean():.1f} ± {v.std():.1f}')
        lines.append(f'| {label} | ' + ' | '.join(cells) + ' |')

    per_patient = {n: np.mean([[m['time_in_target'] for m in res[n][s][1]] for s in res[n]], axis=0) for n in names}
    paired = {}
    for n in names[1:]:
        d = per_patient[n] - per_patient['PID']
        paired[n] = {'better': int(np.sum(d > 1)), 'worse': int(np.sum(d < -1)), 'n': len(d),
                     'mean_diff': float(d.mean())}
        lines.append('')
        lines.append(f'{n} vs PID, per patient (time in target, averaged over seeds): better on {paired[n]["better"]}, '
                     f'worse on {paired[n]["worse"]}, within 1 point on {len(d) - paired[n]["better"] - paired[n]["worse"]} of {len(d)}.')

    out = {n: {str(s): res[n][s][0] for s in res[n]} for n in names}
    out['paired_vs_pid'] = paired
    # the model shown in the video is chosen on validation score, not test score
    best = {}
    for kind in ('sac', 'residual'):
        scores = {}
        for s in seeds_of(kind):
            log = json.load(open(f'results/learning_curve_{kind}_seed{s}.json'))
            scores[s] = max(r['time_in_target'] for r in log)
        if scores:
            best[kind] = max(scores, key=scores.get)
    out['video_models'] = best
    json.dump(out, open('results/benchmark.json', 'w'), indent=2)
    open('results/benchmark.md', 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    figure_learning_curves(res['PID'][0][0]['time_in_target'])
    figure_paired(per_patient)
    if 'residual' in best:
        figure_traces(load('residual', f'models/residual_seed{best["residual"]}.zip'))


def style(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.grid(axis='y', color='#E5E7EB')


def figure_learning_curves(pid_test):
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=130)
    for kind, label in (('sac', 'SAC'), ('residual', 'PID + SAC')):
        for i, path in enumerate(sorted(glob.glob(f'results/learning_curve_{kind}_seed*.json'))):
            log = json.load(open(path))
            ax.plot([r['step'] / 1000 for r in log], [r['time_in_target'] for r in log], color=C[label],
                    lw=1.6, alpha=0.85, label=label if i == 0 else None)
    ax.set_xlabel('training steps (thousands)'); ax.set_ylabel('time in target (%)')
    ax.set_ylim(0, 100); style(ax); ax.legend(frameon=False, loc='lower right')
    fig.tight_layout(); fig.savefig('media/learning_curves.png'); plt.close(fig)


def figure_paired(pp):
    fig, ax = plt.subplots(figsize=(4.8, 4.6), dpi=130)
    ax.plot([0, 100], [0, 100], color='#D1D5DB', lw=1)
    for n in ('SAC', 'PID + SAC'):
        if n in pp:
            ax.scatter(pp['PID'], pp[n], s=30, color=C[n], alpha=0.85, label=n,
                       marker='o' if n == 'SAC' else '^')
    ax.set_xlabel('PID: time in target (%)'); ax.set_ylabel('RL controller: time in target (%)')
    ax.text(4, 93, 'above the line: RL better', color=MUTED, fontsize=9)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); style(ax); ax.legend(frameon=False, loc='lower right')
    fig.tight_layout(); fig.savefig('media/paired_patients.png'); plt.close(fig)


def figure_traces(model, idx=(0, 3, 8, 11, 19, 26)):
    pats = test_patients(30); pid = load_pid()
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6), dpi=120, sharex=True, sharey=True)
    for ax, i in zip(axes.flat, idx):
        for ctrl, c, lab in ((pid, C['PID'], 'PID'), (model, C['PID + SAC'], 'PID + SAC')):
            tr = run_episode(ctrl, pats[i], TEST_NOISE_OFFSET + i)
            ax.plot([s['time_min'] for s in tr], [s['bis'] for s in tr], color=c, lw=1.5, label=lab)
        ax.axhspan(40, 60, color=C_TARGET, alpha=0.1, lw=0)
        ax.text(0.98, 0.97, describe(pats[i]), transform=ax.transAxes, fontsize=10, va='top', ha='right'); ax.set_ylim(0, 100); style(ax)
    axes[0, 0].legend(frameon=False, loc='center right')
    for ax in axes[:, 0]: ax.set_ylabel('BIS')
    for ax in axes[1]: ax.set_xlabel('time (min)')
    fig.tight_layout(); fig.savefig('media/test_traces.png'); plt.close(fig)


if __name__ == '__main__':
    main()
