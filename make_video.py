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
from matplotlib.patches import Rectangle

from evaluate import run_episode, TEST_NOISE_OFFSET
from patients import test_patients, describe
from pid_baseline import load_pid
from AnesthesiaEnv import EPISODE_MIN, stimulation_schedule

C_SAC, C_PID = '#2563EB', '#EA580C'
C_TARGET = '#16A34A'
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
            'inf': g('infusion_mgkgh'), 'dist': g('disturbance'), 'phase': g('phase')}


def zone(b):
    if b > 60: return 'too light', '#D97706'
    if b < 40: return 'too deep', '#7C3AED'
    return 'in target', C_TARGET


def render(sac_tr, pid_tr, patient, out_mp4, out_gif, rl_name='SAC', rl_long='SAC (reinforcement learning)', stride_mp4=1, stride_gif=3):
    S, P = arrays(sac_tr), arrays(pid_tr)
    n = len(S['t'])
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.edgecolor': GRID,
                         'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor='white')
    gs = fig.add_gridspec(3, 4, height_ratios=[0.42, 3.0, 1.3], width_ratios=[1, 1, 1, 0.75],
                          left=0.07, right=0.97, top=0.965, bottom=0.08, hspace=0.28, wspace=0.25)
    head = fig.add_subplot(gs[0, :]); head.axis('off')
    axb = fig.add_subplot(gs[1, :3])
    axi = fig.add_subplot(gs[2, :3], sharex=axb)
    axg = fig.add_subplot(gs[1:, 3]); axg.axis('off')

    head.text(0, 0.95, f'Closed-loop propofol anaesthesia: {rl_name} vs PID',
              fontsize=16, fontweight='bold', color=INK, va='top')
    head.text(0, 0.0, f'Test patient never seen in training: {describe(patient)}   ·   '
              f'noisy, 20 s-delayed BIS monitor   ·   identical conditions for both',
              fontsize=11, color=MUTED, va='bottom')
    clock = head.text(1.0, 0.95, '', fontsize=16, color=INK, ha='right', va='top', family='DejaVu Sans Mono')

    # BIS panel
    axb.axhspan(40, 60, color=C_TARGET, alpha=0.10, lw=0)
    axb.axhline(50, color=C_TARGET, lw=1, ls='--', alpha=0.7)
    axb.text(0.3, 58.5, 'target 40-60', color=C_TARGET, fontsize=10, va='top')
    stim = S['dist'] > 0.5
    if stim.any():
        edges = np.flatnonzero(np.diff(np.r_[0, stim.astype(int), 0]))
        for k, (a, b) in enumerate(zip(edges[::2], edges[1::2])):
            axb.axvspan(S['t'][a], S['t'][min(b, n - 1)], color='#9CA3AF', alpha=0.13, lw=0)
            axb.text(S['t'][a] + 0.2, 97, 'surgical stimulation', fontsize=9, color=MUTED, va='top')
    axb.set_xlim(0, EPISODE_MIN); axb.set_ylim(0, 100)
    axb.set_ylabel('BIS (depth of anaesthesia)')
    axb.grid(axis='y', color=GRID, lw=0.8)
    axb.text(39.7, 95, 'awake ≈ 90+', fontsize=9, color=MUTED, ha='right', va='top')
    (ls,) = axb.plot([], [], color=C_SAC, lw=2.4, label=rl_long)
    (lp,) = axb.plot([], [], color=C_PID, lw=2.0, label='PID (tuned baseline)')
    (ms,) = axb.plot([], [], '.', color=C_SAC, ms=2.5, alpha=0.35)
    (mp,) = axb.plot([], [], '.', color=C_PID, ms=2.5, alpha=0.35)
    axb.legend(loc='lower right', frameon=False, fontsize=10)
    for s in ('top', 'right'): axb.spines[s].set_visible(False)
    plt.setp(axb.get_xticklabels(), visible=False)

    # infusion panel
    axi.set_ylim(0, 21); axi.set_ylabel('propofol\nmg/kg/h')
    axi.set_xlabel('time (min)')
    axi.grid(axis='y', color=GRID, lw=0.8)
    (is_,) = axi.step([], [], color=C_SAC, lw=1.8, where='post')
    (ip,) = axi.step([], [], color=C_PID, lw=1.5, where='post')
    for s in ('top', 'right'): axi.spines[s].set_visible(False)

    # gauges
    def gauge(x, name, color):
        axg.add_patch(Rectangle((x, 0.10), 0.28, 0.75, fc='#F3F4F6', ec='none'))
        for lo, hi, c in ((0, 40, '#7C3AED'), (40, 60, C_TARGET), (60, 100, '#D97706')):
            axg.add_patch(Rectangle((x, 0.10 + 0.75 * lo / 100), 0.28, 0.75 * (hi - lo) / 100, fc=c, alpha=0.18, ec='none'))
        axg.text(x + 0.14, 0.90, name, ha='center', fontsize=12, fontweight='bold', color=color)
        (marker,) = axg.plot([x - 0.02, x + 0.30], [0.5, 0.5], color=color, lw=4, solid_capstyle='butt')
        val = axg.text(x + 0.14, 0.03, '', ha='center', fontsize=11, color=INK)
        stat = axg.text(x + 0.14, -0.04, '', ha='center', fontsize=10)
        tit = axg.text(x + 0.14, -0.11, '', ha='center', fontsize=10, color=MUTED)
        return marker, val, stat, tit

    axg.set_xlim(0, 1); axg.set_ylim(-0.15, 1)
    for yv, lab in ((0.10 + 0.75 * 0.2, 'too deep'), (0.10 + 0.75 * 0.5, 'target'), (0.10 + 0.75 * 0.8, 'too light')):
        axg.text(0.5, yv, lab, ha='center', va='center', fontsize=9, color=MUTED, rotation=0)
    gS = gauge(0.05, rl_name, C_SAC)
    gP = gauge(0.67, 'PID', C_PID)

    def update(i):
        m, s = divmod(int(round(S['t'][i] * 60)), 60)
        clock.set_text(f'{m:02d}:{s:02d} / 40:00')
        for D, line, dots, inf in ((S, ls, ms, is_), (P, lp, mp, ip)):
            line.set_data(D['t'][:i + 1], D['bis'][:i + 1])
            dots.set_data(D['t'][:i + 1], D['meas'][:i + 1])
            inf.set_data(D['t'][:i + 1], D['inf'][:i + 1])
        for D, (marker, val, stat, tit) in ((S, gS), (P, gP)):
            b = D['bis'][i]
            y = 0.10 + 0.75 * b / 100
            marker.set_ydata([y, y])
            val.set_text(f'BIS {b:4.0f}')
            z, c = zone(b)
            stat.set_text(z); stat.set_color(c)
            mt = D['phase'][:i + 1] == 'maintenance'
            if mt.any():
                bb = D['bis'][:i + 1][mt]
                tit.set_text(f'{np.mean((bb >= 40) & (bb <= 60)) * 100:.0f}% in target')
        return []

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    w = FFMpegWriter(fps=16, bitrate=2400)
    with w.saving(fig, out_mp4, dpi=100):
        for i in list(range(0, n, stride_mp4)) + [n - 1] * 32:
            update(i); w.grab_frame()
    if out_gif:
        fig.set_size_inches(9.6, 5.4)
        w = PillowWriter(fps=10)
        with w.saving(fig, out_gif, dpi=80):
            for i in list(range(0, n, stride_gif)) + [n - 1] * 15:
                update(i); w.grab_frame()
    # poster frame
    update(n - 1)
    fig.set_size_inches(12.8, 7.2)
    fig.savefig(out_mp4.replace('.mp4', '_final.png'), dpi=100)
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
           rl_long='PID + SAC (RL corrections on top of PID)' if residual else 'SAC (pure reinforcement learning)')
    print('wrote', a.out)
