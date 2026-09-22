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
ROWS = [('time_in_target', 'Maintenance: time in 40-60 (%)'),
        ('MDAPE', 'Maintenance: MDAPE (%)'),
        ('MDPE', 'Maintenance: MDPE (%)'),
        ('wobble', 'Maintenance: wobble (%)'),
        ('propofol_mgkgh', 'Maintenance: propofol (mg/kg/h)'),
        ('case_below_40', 'Whole case: time below 40 (%)'),
        ('case_above_60', 'Whole case: time above 60 after 1 min (%)'),
        ('reached_below_20', 'Patients reaching BIS < 20 (%)')]


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
    head = '| | ' + ' | '.join(names) + ' |'
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

    figure_paired(per_patient)
    if 'residual' in best and 'sac' in best:
        figure_traces({'PID': load_pid(),
                       'SAC': load('sac', f'models/sac_seed{best["sac"]}.zip'),
                       'PID + SAC': load('residual', f'models/residual_seed{best["residual"]}.zip')})


def style(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=11)


def figure_paired(pp):
    fig, ax = plt.subplots(figsize=(4.4, 4.2), dpi=150)
    ax.plot([0, 100], [0, 100], color='#9CA3AF', lw=1, ls='--')
    for n, mk in (('SAC', 'o'), ('PID + SAC', '^')):
        if n in pp:
            ax.scatter(pp['PID'], pp[n], s=34, color=C[n], alpha=0.85, label=n, marker=mk, lw=0)
    ax.set_xlabel('PID, time in 40-60 (%)', fontsize=12)
    ax.set_ylabel('RL, time in 40-60 (%)', fontsize=12)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); style(ax)
    ax.legend(frameon=False, loc='lower right', fontsize=11, handletextpad=0.2)
    fig.tight_layout(); fig.savefig('media/paired_patients.png'); plt.close(fig)


def figure_traces(models, idx=(0, 3, 8, 11, 19, 26)):
    pats = test_patients(30)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5.6), dpi=120, sharex=True, sharey=True)
    for ax, i in zip(axes.flat, idx):
        for lab, ctrl in models.items():
            tr = run_episode(ctrl, pats[i], TEST_NOISE_OFFSET + i)
            ax.plot([s['time_min'] for s in tr], [s['bis'] for s in tr], color=C[lab], lw=1.4, label=lab)
        ax.axhspan(40, 60, color=C_TARGET, alpha=0.1, lw=0)
        ax.text(0.98, 0.97, describe(pats[i]), transform=ax.transAxes, fontsize=11, va='top', ha='right')
        ax.set_ylim(0, 100); ax.set_xlim(0, 40); ax.set_xticks([0, 10, 20, 30, 40]); style(ax)
    for ax in axes[:, 0]: ax.set_ylabel('BIS', fontsize=12)
    for ax in axes[1]: ax.set_xlabel('Time (min)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=3, frameon=False, fontsize=12)
    fig.savefig('media/test_traces.png'); plt.close(fig)


if __name__ == '__main__':
    main()
