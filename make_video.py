"""
Render a simulation video: an RL controller vs the PID on one held-out patient.

    python make_video.py --patient 0 --model models/residual_seed4.zip
Outputs media/sac_vs_pid.mp4 and media/sac_vs_pid.gif
"""

import argparse, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.patches import Rectangle

from evaluate import run_episode, TEST_NOISE_OFFSET
from patients import test_patients
from pid_baseline import load_pid
from AnesthesiaEnv import EPISODE_MIN

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


def render(sac_tr, pid_tr, patient, out_mp4, out_gif, rl_name='SAC', rl_long='SAC', stride_mp4=1, stride_gif=3):
    S, P = arrays(sac_tr), arrays(pid_tr)
    C_SAC = COLORS[rl_name]
    n = len(S['t'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.edgecolor': '#9CA3AF',
                         'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK})

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor='white')
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.25], left=0.08, right=0.97, top=0.92, bottom=0.10, hspace=0.12)
    axb = fig.add_subplot(gs[0])
    axi = fig.add_subplot(gs[1], sharex=axb)

    clock = fig.text(0.97, 0.965, '', fontsize=15, color=INK, ha='right', va='center', family='DejaVu Sans Mono')

    # BIS panel
    axb.axhspan(40, 60, color=C_TARGET, alpha=0.12, lw=0)
    stim = S['dist'] > 0.5
    stim_marks = []   # shown once each event starts
    if stim.any():
        edges = np.flatnonzero(np.diff(np.r_[0, stim.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            span = Rectangle((S['t'][a], 0), 0, 1, transform=axb.get_xaxis_transform(), color='#9CA3AF', alpha=0.15, lw=0)
            axb.add_patch(span)
            label = axb.text(S['t'][a] + 0.3, 2, 'stimulation', fontsize=13, color='#4B5563', va='bottom')
            stim_marks.append((S['t'][a], S['t'][min(b, n - 1)], span, label))
    axb.set_xlim(0, EPISODE_MIN); axb.set_ylim(0, 100); axb.set_yticks([0, 20, 40, 60, 80, 100])
    axb.set_ylabel('BIS')
    (ms,) = axb.plot([], [], '.', color=C_SAC, ms=3, alpha=0.3)
    (mp,) = axb.plot([], [], '.', color=C_PID, ms=3, alpha=0.3)
    (ls,) = axb.plot([], [], color=C_SAC, lw=2.6)
    (lp,) = axb.plot([], [], color=C_PID, lw=2.2)
    fig.legend([ls, lp], [rl_long, 'PID'], loc='center left', bbox_to_anchor=(0.07, 0.965), ncol=2,
                     frameon=False, fontsize=14, handlelength=1.5, columnspacing=2.5)
    for s in ('top', 'right'): axb.spines[s].set_visible(False)
    plt.setp(axb.get_xticklabels(), visible=False)
    # running time in 40-60, from 5 min on (same window as the results table)
    rhead = axb.text(0.995, 0.99, 'in 40-60 since 5 min', transform=axb.transAxes, ha='right', va='top',
                     fontsize=12, color=MUTED)
    rvals = [axb.text(0.995, 0.92 - 0.075 * k, '', transform=axb.transAxes, ha='right', va='top',
                      fontsize=14, color=c, family='DejaVu Sans Mono') for k, c in enumerate((C_SAC, C_PID))]

    # infusion panel
    axi.set_ylim(0, 21); axi.set_yticks([0, 10, 20]); axi.set_ylabel('Infusion\n(mg/kg/h)')
    axi.set_xlabel('Time (min)'); axi.set_xticks([0, 10, 20, 30, 40])
    (is_,) = axi.step([], [], color=C_SAC, lw=1.8, where='post')
    (ip,) = axi.step([], [], color=C_PID, lw=1.6, where='post')
    for s in ('top', 'right'): axi.spines[s].set_visible(False)

    # induction boluses don't fit on the infusion axis, so state them in a line of text
    # induction boluses don't fit on the infusion axis, so state them in a line of text, one colour per controller
    given = [(name, D, c) for name, D, c in ((rl_long, S, C_SAC), ('PID', P, C_PID)) if D['bol'].sum() >= 0.1]
    t_bolus = max([D['t'][np.argmax(D['bol'] > 0)] for _, D, _ in given], default=0)
    bolus_texts = [axi.text(0.03, 0.97, 'Bolus, first minute:', transform=axi.transAxes, fontsize=13, color=INK, va='top')]
    fig.canvas.draw()
    for name, D, c in given:
        x = axi.transAxes.inverted().transform(bolus_texts[-1].get_window_extent().corners()[-1])[0] + 0.03
        bolus_texts.append(axi.text(x, 0.97, f'{name} {D["bol"].sum():.1f} mg/kg', transform=axi.transAxes,
                                    fontsize=13, color=c, va='top'))
        fig.canvas.draw()
    for t in bolus_texts: t.set_visible(False)

    def in_range(D, i):
        # from 5 min on, same window as the results table
        bb = D['bis'][:i + 1][D['t'][:i + 1] >= 5.0]
        return f'{np.mean((bb >= 40) & (bb <= 60)) * 100:.0f}%' if len(bb) else None

    def update(i):
        m, s = divmod(int(round(S['t'][i] * 60)), 60)
        clock.set_text(f't = {m:02d}:{s:02d}')
        for D, line, dots, inf in ((S, ls, ms, is_), (P, lp, mp, ip)):
            line.set_data(D['t'][:i + 1], D['bis'][:i + 1])
            dots.set_data(D['t'][:i + 1], D['meas'][:i + 1])
            inf.set_data(D['t'][:i + 1], D['inf'][:i + 1])
        for t0, t1, span, label in stim_marks:   # grow each event as it happens
            span.set_width(np.clip(S['t'][i] - t0, 0, t1 - t0)); label.set_visible(S['t'][i] >= t0)
        rhead.set_visible(S['t'][i] >= 5.0)
        for txt, name, D in zip(rvals, (rl_long, 'PID'), (S, P)):
            r = in_range(D, i)
            txt.set_text('' if r is None else f'{name} {r:>4}')
        for t in bolus_texts: t.set_visible(S['t'][i] >= t_bolus)
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
