"""
Plotting functions for anaesthesia control evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# plot style
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
})

# colors
BLUE = '#1f77b4'
ORANGE = '#ff7f0e'
GREEN = '#2ca02c'
RED = '#d62728'


def plot_bis_trajectory(trajectories, save_path=None, title='BIS Control'):
    """Plot BIS over time for multiple episodes."""
    n = len(trajectories)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows), squeeze=False)

    for i, traj in enumerate(trajectories):
        ax = axes[i // cols, i % cols]

        # target zone
        ax.axhspan(40, 60, alpha=0.2, color=GREEN)
        ax.axhline(50, color=GREEN, linestyle='--', alpha=0.7)

        # BIS trajectory
        ax.plot(traj['time'], traj['bis'], color=BLUE, linewidth=1.5)

        ax.set_xlabel('Time (min)')
        ax.set_ylabel('BIS')
        ax.set_ylim(0, 100)
        ax.set_title(f'Episode {i+1}')

    # hide empty subplots
    for i in range(n, rows * cols):
        axes[i // cols, i % cols].set_visible(False)

    plt.suptitle(title, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_episode_detail(trajectory, save_path=None):
    """Detailed view of one episode: BIS and drug infusion."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    time = trajectory['time']
    bis = trajectory['bis']
    action = trajectory['action']

    # BIS plot
    ax1.axhspan(40, 60, alpha=0.2, color=GREEN)
    ax1.axhline(50, color=GREEN, linestyle='--', alpha=0.7)
    ax1.plot(time, bis, color=BLUE, linewidth=2)
    ax1.set_ylabel('BIS')
    ax1.set_ylim(0, 100)
    ax1.set_title('BIS Trajectory')

    # action plot (scaled to mg/min)
    infusion = np.array(action) * 12  # max 12 mg/min
    ax2.fill_between(time, 0, infusion, alpha=0.5, color=ORANGE)
    ax2.plot(time, infusion, color=ORANGE, linewidth=1)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Infusion (mg/min)')
    ax2.set_ylim(0, 12)
    ax2.set_title('Drug Infusion Rate')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_bis_distribution(trajectories, save_path=None):
    """Histogram of BIS values across all episodes."""
    all_bis = np.concatenate([t['bis'] for t in trajectories])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(all_bis, bins=50, alpha=0.7, color=BLUE, edgecolor='black', density=True)

    # target markers
    ax.axvline(50, color=GREEN, linestyle='--', linewidth=2, label='Target')
    ax.axvspan(40, 60, alpha=0.1, color=GREEN, label='Target zone')

    mean_bis = np.mean(all_bis)
    ax.axvline(mean_bis, color=RED, linestyle=':', linewidth=2, label=f'Mean: {mean_bis:.1f}')

    time_in_target = np.mean((all_bis >= 40) & (all_bis <= 60)) * 100

    ax.set_xlabel('BIS')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 100)
    ax.set_title(f'BIS Distribution (Time in Target: {time_in_target:.1f}%)')
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_comparison(sac_traj, pid_traj, save_path=None):
    """Side-by-side comparison of SAC vs PID."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    sac = sac_traj[0]
    pid = pid_traj[0]

    # top left: BIS trajectories
    ax = axes[0, 0]
    ax.axhspan(40, 60, alpha=0.2, color=GREEN)
    ax.axhline(50, color=GREEN, linestyle='--', alpha=0.5)
    ax.plot(sac['time'], sac['bis'], color=BLUE, linewidth=1.5, label='SAC')
    ax.plot(pid['time'], pid['bis'], color=ORANGE, linewidth=1.5, label='PID', alpha=0.8)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('BIS')
    ax.set_title('BIS Trajectory')
    ax.legend()
    ax.set_ylim(0, 100)

    # top right: actions
    ax = axes[0, 1]
    ax.plot(sac['time'], np.array(sac['action']) * 12, color=BLUE, linewidth=1.5, label='SAC')
    ax.plot(pid['time'], np.array(pid['action']) * 12, color=ORANGE, linewidth=1.5, label='PID', alpha=0.8)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('Infusion (mg/min)')
    ax.set_title('Drug Infusion')
    ax.legend()
    ax.set_ylim(0, 12)

    # bottom left: BIS distribution
    ax = axes[1, 0]
    sac_bis = np.concatenate([t['bis'] for t in sac_traj])
    pid_bis = np.concatenate([t['bis'] for t in pid_traj])

    ax.hist(sac_bis, bins=50, alpha=0.6, color=BLUE, label=f'SAC (mean={np.mean(sac_bis):.1f})', density=True)
    ax.hist(pid_bis, bins=50, alpha=0.6, color=ORANGE, label=f'PID (mean={np.mean(pid_bis):.1f})', density=True)
    ax.axvline(50, color=GREEN, linestyle='--', linewidth=2)
    ax.axvspan(40, 60, alpha=0.1, color=GREEN)
    ax.set_xlabel('BIS')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 100)
    ax.set_title('BIS Distribution')
    ax.legend(fontsize=9)

    # bottom right: time in target comparison
    ax = axes[1, 1]
    sac_40_60 = np.mean((sac_bis >= 40) & (sac_bis <= 60)) * 100
    pid_40_60 = np.mean((pid_bis >= 40) & (pid_bis <= 60)) * 100
    sac_45_55 = np.mean((sac_bis >= 45) & (sac_bis <= 55)) * 100
    pid_45_55 = np.mean((pid_bis >= 45) & (pid_bis <= 55)) * 100

    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [sac_40_60, sac_45_55], width, label='SAC', color=BLUE)
    ax.bar(x + width/2, [pid_40_60, pid_45_55], width, label='PID', color=ORANGE)
    ax.set_ylabel('Time in Target (%)')
    ax.set_title('Target Adherence')
    ax.set_xticks(x)
    ax.set_xticklabels(['BIS 40-60', 'BIS 45-55'])
    ax.legend()
    ax.set_ylim(0, 100)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def plot_metrics_bar(sac_metrics, pid_metrics, save_path=None):
    """Bar chart comparing key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # panel 1: bias and inaccuracy
    ax = axes[0]
    x = np.arange(2)
    width = 0.35

    sac_vals = [abs(sac_metrics['bias']), sac_metrics['inaccuracy']]
    pid_vals = [abs(pid_metrics['bias']), pid_metrics['inaccuracy']]

    ax.bar(x - width/2, sac_vals, width, label='SAC', color=BLUE)
    ax.bar(x + width/2, pid_vals, width, label='PID', color=ORANGE)
    ax.axhline(20, color='gray', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_ylabel('Error (%)')
    ax.set_title('Bias & Inaccuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(['|Bias|', 'Inaccuracy'])
    ax.legend()

    # panel 2: time in target
    ax = axes[1]
    sac_vals = [sac_metrics['time_in_target_40_60'], sac_metrics['time_in_target_45_55']]
    pid_vals = [pid_metrics['time_in_target_40_60'], pid_metrics['time_in_target_45_55']]

    ax.bar(x - width/2, sac_vals, width, label='SAC', color=BLUE)
    ax.bar(x + width/2, pid_vals, width, label='PID', color=ORANGE)
    ax.set_ylabel('Time in Target (%)')
    ax.set_title('Target Adherence')
    ax.set_xticks(x)
    ax.set_xticklabels(['40-60', '45-55'])
    ax.legend()
    ax.set_ylim(0, 100)

    # panel 3: variability and terminations
    ax = axes[2]
    sac_vals = [sac_metrics['variability'], sac_metrics['termination_rate']]
    pid_vals = [pid_metrics['variability'], pid_metrics['termination_rate']]

    ax.bar(x - width/2, sac_vals, width, label='SAC', color=BLUE)
    ax.bar(x + width/2, pid_vals, width, label='PID', color=ORANGE)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Stability & Safety')
    ax.set_xticks(x)
    ax.set_xticklabels(['Variability', 'Terminations'])
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    return fig


def generate_all_plots(sac_traj, sac_metrics, pid_traj=None, pid_metrics=None, save_dir='./figures'):
    """Generate all evaluation figures."""
    os.makedirs(save_dir, exist_ok=True)

    print("  - BIS trajectories")
    plot_bis_trajectory(sac_traj[:6], f'{save_dir}/bis_trajectories.png')
    plt.close()

    print("  - Episode detail")
    plot_episode_detail(sac_traj[0], f'{save_dir}/episode_detail.png')
    plt.close()

    print("  - BIS distribution")
    plot_bis_distribution(sac_traj, f'{save_dir}/bis_distribution.png')
    plt.close()

    if pid_traj and pid_metrics:
        print("  - SAC vs PID comparison")
        plot_comparison(sac_traj, pid_traj, f'{save_dir}/sac_vs_pid.png')
        plt.close()

        print("  - Metrics comparison")
        plot_metrics_bar(sac_metrics, pid_metrics, f'{save_dir}/metrics_comparison.png')
        plt.close()

    print(f"  Saved to {save_dir}/")
