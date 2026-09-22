"""
Render a side-by-side simulation video: SAC vs PID on one held-out patient.

    python make_video.py --patient 3 --model models/sac_seed0.zip
Outputs media/sac_vs_pid.mp4 and media/sac_vs_pid.gif
"""

import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter

from evaluate import run_episode, TEST_NOISE_OFFSET
from patients import test_patients, describe
from pid_baseline import load_pid
from AnesthesiaEnv import EPISODE_MIN, stimulation_schedule

from style import COLORS, TARGET as C_TARGET
C_PID = COLORS['PID']
INK, MUTED, GRID = '#111827', '#6B7280', '#E5E7EB'


def load_controller(path):
    from stable_baselines3 import SAC
    m = SAC.load(path, device='cpu')
    if 'residual' in os.path.basename(path):
        from residual import ResidualPolicy
        return ResidualPolicy(m)
    return m


def arrays(trace):
    g = lambda k: np.array([s[k] for s in trace])
    return {'t': g('time_min'), 'bis': g('bis'), 'meas': g('bis_measured'),
            'inf': g('infusion_mgkgh'), 'dist': g('disturbance'), 'phase': g('phase'),
            'bol': g('bolus_mgkg')}


def zone(b):
    if b > 60: return 'too light', '#D97706'
    if b < 40: return 'too deep', '#7C3AED'
    return 'in target', C_TARGET


def render(sac_tr, pid_tr, patient, out_mp4, out_gif, rl_name='SAC', rl_long='SAC', stride_mp4=1, stride_gif=3):
    S, P = arrays(sac_tr), arrays(pid_tr)
    C_SAC = COLORS[rl_name]
    n = len(S['t'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.edgecolor': '#9CA3AF',
                         'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor='white')
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.2], left=0.08, right=0.97, top=0.92, bottom=0.10, hspace=0.12)
    axb = fig.add_subplot(gs[0])
    axi = fig.add_subplot(gs[1], sharex=axb)

    clock = fig.text(0.97, 0.965, '', fontsize=15, color=INK, ha='right', va='center', family='DejaVu Sans Mono')

    # BIS panel
    axb.axhspan(40, 60, color=C_TARGET, alpha=0.12, lw=0)
    stim = S['dist'] > 0.5
    if stim.any():
        edges = np.flatnonzero(np.diff(np.r_[0, stim.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            axb.axvspan(S['t'][a], S['t'][min(b, n - 1)], color='#9CA3AF', alpha=0.15, lw=0)
            axb.text(S['t'][a] + 0.3, 2, 'stimulation', fontsize=13, color='#4B5563', va='bottom')
    axb.set_xlim(0, EPISODE_MIN); axb.set_ylim(0, 100); axb.set_yticks([0, 20, 40, 60, 80, 100])
    axb.set_ylabel('BIS')
    (ms,) = axb.plot([], [], '.', color=C_SAC, ms=4, alpha=0.45)
    (mp,) = axb.plot([], [], '.', color=C_PID, ms=4, alpha=0.45)
    (ls,) = axb.plot([], [], color=C_SAC, lw=2.6)
    (lp,) = axb.plot([], [], color=C_PID, lw=2.2)
    leg = axb.legend([ls, lp], [rl_long, 'PID'], loc='upper right', bbox_to_anchor=(1.0, 0.99),
                     frameon=False, fontsize=14, handlelength=1.5)
    for s in ('top', 'right'): axb.spines[s].set_visible(False)
    plt.setp(axb.get_xticklabels(), visible=False)

    # infusion panel
    axi.set_ylim(0, 21); axi.set_yticks([0, 10, 20]); axi.set_ylabel('Propofol\n(mg/kg/h)')
    axi.set_xlabel('Time (min)'); axi.set_xticks([0, 10, 20, 30, 40])
    (is_,) = axi.step([], [], color=C_SAC, lw=1.8, where='post')
    (ip,) = axi.step([], [], color=C_PID, lw=1.6, where='post')
    for s in ('top', 'right'): axi.spines[s].set_visible(False)

    # induction boluses don't fit on the infusion axis: mark them with a triangle and label
    bolus_marks = []
    for k, (D, color) in enumerate(((S, C_SAC), (P, C_PID))):
        total = D['bol'].sum()
        if total >= 0.1:
            t0 = D['t'][np.argmax(D['bol'] > 0)]
            y = 17 - 7 * k
            mk = axi.plot([t0 + 0.15], [y], marker='v', color=color, ms=9, ls='none')[0]
            tx = axi.text(t0 + 1.8, y, f'bolus {total:.1f} mg/kg', color=color, fontsize=13, va='center')
            for a in (mk, tx): a.set_visible(False)
            bolus_marks += [(t0, mk), (t0, tx)]

    def in_range(D, i):
        # from 5 min on, same window as the results table
        bb = D['bis'][:i + 1][D['t'][:i + 1] >= 5.0]
        return f'{np.mean((bb >= 40) & (bb <= 60)) * 100:.0f}%' if len(bb) else '-'

    def update(i):
        m, s = divmod(int(round(S['t'][i] * 60)), 60)
        clock.set_text(f't = {m:02d}:{s:02d}')
        for D, line, dots, inf in ((S, ls, ms, is_), (P, lp, mp, ip)):
            line.set_data(D['t'][:i + 1], D['bis'][:i + 1])
            dots.set_data(D['t'][:i + 1], D['meas'][:i + 1])
            inf.set_data(D['t'][:i + 1], D['inf'][:i + 1])
        leg.get_texts()[0].set_text(f'{rl_long}: {in_range(S, i)} in 40-60 since 5 min')
        leg.get_texts()[1].set_text(f'PID: {in_range(P, i)} in 40-60 since 5 min')
        for t0, mk in bolus_marks:
            mk.set_visible(S['t'][i] >= t0)
        return []

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    w = FFMpegWriter(fps=16, bitrate=2400)
    with w.saving(fig, out_mp4, dpi=100):
        for i in list(range(0, n, stride_mp4)) + [n - 1] * 32:
            update(i); w.grab_frame()
    if out_gif:
        # same layout as the MP4, just a lower resolution, so nothing gets cropped
        w = PillowWriter(fps=10)
        with w.saving(fig, out_gif, dpi=64):
            for i in list(range(0, n, stride_gif)) + [n - 1] * 15:
                update(i); w.grab_frame()
    plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--patient', type=int, default=0)
    ap.add_argument('--model', default='models/sac_seed0.zip')
    ap.add_argument('--out', default='media/sac_vs_pid.mp4')
    ap.add_argument('--no-gif', action='store_true')
    a = ap.parse_args()
    patient = test_patients(30)[a.patient]
    seed = TEST_NOISE_OFFSET + a.patient
    pid = load_pid()
    rl = load_controller(a.model)
    residual = 'residual' in os.path.basename(a.model)
    render(run_episode(rl, patient, seed), run_episode(pid, patient, seed), patient,
           a.out, None if a.no_gif else a.out.replace('.mp4', '.gif'),
           rl_name='PID + SAC' if residual else 'SAC',
           rl_long='PID + SAC' if residual else 'SAC')
    print('wrote', a.out)
