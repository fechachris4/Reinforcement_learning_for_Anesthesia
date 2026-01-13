"""
Generate Animations (GIFs) for Anaesthesia Control
===================================================

Creates side-by-side comparison GIFs showing untrained vs trained agent behavior.

Usage:
    python generate_animations.py                    # Generate all animations
    python generate_animations.py --trained-only     # Only trained agent GIF
"""

import os
import sys

# Ensure we can import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import io

from stable_baselines3 import SAC
from AnesthesiaEnv import AnesthesiaEnv, EvaluationEnv, max_infusion_rate, max_bolus_rate, bolus_budget

# Constants
BIS_TARGET = 50
BIS_ACCEPTABLE = (40, 60)
PROPOFOL_MAX = max_infusion_rate  # mg/min (from AnesthesiaEnv)
BOLUS_MAX = max_bolus_rate        # mg/kg/min
BOLUS_BUDGET = bolus_budget       # mg/kg total

# Colors
BLUE = '#2563EB'
AMBER = '#F59E0B'
GREEN = '#10B981'
RED = '#EF4444'
CYAN = '#00D4FF'


def run_episode_for_gif(env, model=None, max_steps=2400):
    """Run episode and return data for GIF frames."""
    obs, info = env.reset()

    # Get dt from environment (1/60 min = 1 sec per step)
    dt = getattr(env, 'dt', 1/60)

    data = {
        'time': [],
        'bis': [],
        'applied_infusion': [],    # mg/min (what env actually used)
        'applied_bolus': [],       # mg/kg/min (what env actually used)
        'bolus_remaining': [],
        'phase': [],
        'reward': [],
        'cumulative_reward': [],
        'ce_prop': [],
    }

    cumulative = 0

    for step in range(max_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        cumulative += reward
        data['time'].append(step * dt)
        data['bis'].append(info.get('bis', 50))
        # Use APPLIED values from info, not raw agent output
        data['applied_infusion'].append(info.get('applied_infusion_mg_min', 0))
        data['applied_bolus'].append(info.get('applied_bolus_mg_kg_min', 0))
        data['bolus_remaining'].append(info.get('bolus_remaining', 0))
        data['phase'].append(info.get('phase', 'maintenance'))
        data['reward'].append(reward)
        data['cumulative_reward'].append(cumulative)
        data['ce_prop'].append(info.get('ce_prop', 0))

        if terminated or truncated:
            break

    return data


def create_frame(data, step_idx, title, max_time=None, figsize=(14, 4), dpi=100):
    """Create a single frame for the GIF with two-phase visualization."""
    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Lock x-axis to prevent jitter across frames
    if max_time is None:
        max_time = max(12, data['time'][-1] + 0.5)

    time = data['time'][:step_idx+1]
    bis = data['bis'][:step_idx+1]
    applied_infusion = data['applied_infusion'][:step_idx+1]
    applied_bolus = data['applied_bolus'][:step_idx+1]
    bolus_remaining = data['bolus_remaining'][:step_idx+1]
    phases = data['phase'][:step_idx+1]
    current_phase = phases[-1] if phases else 'induction'

    current_bis = bis[-1]
    current_bolus_left = bolus_remaining[-1]
    cumulative_r = data['cumulative_reward'][step_idx]

    # Calculate time in target so far
    bis_arr = np.array(bis)
    tit = np.mean((bis_arr >= 40) & (bis_arr <= 60)) * 100

    phase_label = "INDUCTION" if current_phase == 'induction' else "MAINTENANCE"
    fig.suptitle(f"{title} | {phase_label} | Time: {time[-1]:.1f} min | Reward: {cumulative_r:.0f} | TiT: {tit:.0f}%",
                 fontsize=11, fontweight='bold')

    # Panel 1: BIS Gauge
    ax = axes[0]
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.add_patch(Rectangle((0, 0), 20, 1, color=RED, alpha=0.7))
    ax.add_patch(Rectangle((20, 0), 20, 1, color=AMBER, alpha=0.7))
    ax.add_patch(Rectangle((40, 0), 20, 1, color=GREEN, alpha=0.7))
    ax.add_patch(Rectangle((60, 0), 20, 1, color=AMBER, alpha=0.7))
    ax.add_patch(Rectangle((80, 0), 20, 1, color=RED, alpha=0.7))
    ax.axvline(50, color='black', linewidth=2, linestyle='--')
    ax.axvline(current_bis, color=CYAN, linewidth=4)
    ax.set_title(f'BIS: {current_bis:.1f}', fontweight='bold')
    ax.set_yticks([])
    ax.set_xlabel('BIS Value')

    # Panel 2: BIS History with phase shading
    ax = axes[1]
    # Add phase background shading
    _add_phase_shading(ax, time, phases, max_time)
    ax.axhspan(*BIS_ACCEPTABLE, alpha=0.15, color=GREEN, zorder=1)
    ax.axhline(BIS_TARGET, color='black', linestyle='--', alpha=0.5, zorder=2)
    ax.plot(time, bis, color=BLUE, linewidth=2, zorder=3)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max_time)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('BIS')
    ax.set_title('BIS History', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel 3: Bolus Rate (applied, not raw action)
    # Panel 3: Ce Concentration + Bolus Gauge Inset
    ax = axes[2]
    ce_prop = data['ce_prop'][:step_idx+1]
    
    # Main Plot: Effect-Site Concentration
    ax.plot(time, ce_prop, color='purple', linewidth=2, label='Ce Propofol')
    ax.fill_between(time, ce_prop, alpha=0.1, color='purple')
    ax.axhspan(3.0, 5.0, alpha=0.1, color='gray', label='Window')
    
    current_ce = ce_prop[-1]
    ax.set_ylim(0, 8.0)
    ax.set_xlim(0, max_time)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel(r'Concentration ($\mu$g/mL)')
    ax.set_title(f'Effect-Site ($Ce$): {current_ce:.2f}', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Inset Plot: Bolus Usage Gauge
    # Position: [x, y, width, height] in normalized axes coordinates
    ax_inset = ax.inset_axes([0.85, 0.55, 0.12, 0.4])
    bolus_used = BOLUS_BUDGET - bolus_remaining[-1]
    
    # Bar shows absolute mg/kg used
    ax_inset.bar([0], [bolus_used], color=RED, alpha=0.8, width=0.8)
    ax_inset.set_ylim(0, BOLUS_BUDGET)
    ax_inset.set_xticks([])
    ax_inset.set_ylabel('mg/kg', fontsize=8)
    ax_inset.set_title('Bolus', fontsize=9, fontweight='bold')
    
    # Add text label: "Used: 2.5 mg/kg"
    ax_inset.text(0, bolus_used/2 if bolus_used > 0.5 else 0.2, 
                  f"Used:\n{bolus_used:.2f}", 
                  ha='center', va='center', fontsize=8, color='black', fontweight='bold')
    
    ax_inset.patch.set_alpha(0.8)

    # Panel 4: Infusion Rate (applied, not raw action)
    ax = axes[3]
    ax.fill_between(time, applied_infusion, alpha=0.7, color=AMBER, label='Infusion')
    ax.set_ylim(0, PROPOFOL_MAX * 1.1)
    ax.set_xlim(0, max_time)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('mg/min')
    ax.set_title('Infusion Rate (applied)', fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Convert to PIL Image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)

    return img


def _add_phase_shading(ax, time, phases, max_time):
    """Add background shading to indicate induction vs maintenance phase."""
    if not time:
        return

    # Find phase transition point
    transition_time = None
    for i, p in enumerate(phases):
        if p == 'maintenance':
            transition_time = time[i]
            break

    # Shade induction phase (light red), maintenance (light green)
    if transition_time is not None:
        ax.axvspan(0, transition_time, alpha=0.08, color=RED, zorder=0)
        ax.axvspan(transition_time, max_time, alpha=0.08, color=GREEN, zorder=0)
    else:
        # Still in induction
        ax.axvspan(0, max_time, alpha=0.08, color=RED, zorder=0)


def generate_gif(data, title, output_path, frame_interval=3, fps=12):
    """Generate GIF from episode data."""
    frames = []
    total_steps = len(data['time'])

    # Pre-compute max_time to lock axis limits (prevents jitter)
    max_time = max(12, data['time'][-1] + 0.5)

    print(f"  Creating frames for {title}...")
    for step in range(0, total_steps, frame_interval):
        frame = create_frame(data, step, title, max_time=max_time)
        frames.append(frame)

        if step % 100 == 0:
            print(f"    Step {step}/{total_steps}")

    # Add final frame
    if (total_steps - 1) % frame_interval != 0:
        frames.append(create_frame(data, total_steps - 1, title, max_time=max_time))

    # Save GIF
    duration = int(1000 / fps)
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0)
    print(f"  Saved: {output_path} ({len(frames)} frames)")


def print_episode_stats(data, title):
    """Print a nice summary table for the animated episode."""
    bis = np.array(data['bis'])
    time_min = np.array(data['time'])
    
    # Metrics
    mean_bis = np.mean(bis)
    std_bis = np.std(bis)
    tit_40_60 = np.mean((bis >= 40) & (bis <= 60)) * 100
    tit_45_55 = np.mean((bis >= 45) & (bis <= 55)) * 100
    
    # Safety
    deep = np.mean(bis < 30) * 100
    light = np.mean(bis > 70) * 100
    
    # Phase transition
    phases = data['phase']
    try:
        m_idx = phases.index('maintenance')
        induction_time = time_min[m_idx]
    except ValueError:
        induction_time = time_min[-1]  # Never entered maintenance
        
    print(f"""
    STATS FOR: {title}
    --------------------------------------------------
    Duration:        {time_min[-1]:.1f} min
    Induction Time:  {induction_time:.1f} min
    
    Mean BIS:        {mean_bis:.1f} (SD: {std_bis:.1f})
    TiT (40-60):     {tit_40_60:.1f} %
    TiT (45-55):     {tit_45_55:.1f} %
    
    Deep (<30):      {deep:.1f} %
    Light (>70):     {light:.1f} %
    Total Reward:    {data['cumulative_reward'][-1]:.0f}
    --------------------------------------------------
    """)


def main():
    parser = argparse.ArgumentParser(description="Generate animation GIFs")
    parser.add_argument("--model", type=str, default="./models/sac_model",
                        help="Path to trained SAC model")
    parser.add_argument("--output", type=str, default="./figures",
                        help="Output directory")
    parser.add_argument("--steps", type=int, default=2400,
                        help="Max steps per episode (2400 = 40 min)")
    parser.add_argument("--trained-only", action="store_true",
                        help="Only generate trained agent GIF")
    parser.add_argument("--fps", type=int, default=12,
                        help="Frames per second")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("="*60)
    print("Generating Animation GIFs")
    print("="*60)

    # Load trained model
    print(f"\nLoading model from {args.model}")
    trained_model = SAC.load(args.model)

    env = EvaluationEnv()

    # Generate trained agent GIF
    print("\n--- Trained Agent ---")
    env.reset(seed=42)
    trained_data = run_episode_for_gif(env, model=trained_model, max_steps=args.steps)
    generate_gif(trained_data, "Trained SAC Agent",
                 os.path.join(args.output, "trained_agent.gif"), fps=args.fps)
    print_episode_stats(trained_data, "Trained Agent")

    # Generate untrained agent GIF
    if not args.trained_only:
        print("\n--- Untrained Agent (Random) ---")
        env.reset(seed=42)
        untrained_data = run_episode_for_gif(env, model=None, max_steps=args.steps)
        generate_gif(untrained_data, "Untrained Agent (Random)",
                     os.path.join(args.output, "untrained_agent.gif"), fps=args.fps)
        print_episode_stats(untrained_data, "Untrained Agent")

    env.close()

    print("\n" + "="*60)
    print("Animation Generation Complete!")
    print("="*60)
    print(f"\nGIFs saved to {args.output}/")
    print("  - trained_agent.gif")
    if not args.trained_only:
        print("  - untrained_agent.gif")


if __name__ == "__main__":
    main()
