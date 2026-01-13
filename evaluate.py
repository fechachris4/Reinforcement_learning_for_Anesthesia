"""
Evaluation metrics for anaesthesia controllers.
Implements clinical performance measures from Varvel et al. (1992).
"""

import numpy as np
import json
import os

from AnesthesiaEnv import AnesthesiaEnv


# --- Clinical metrics (Varvel et al. 1992) ---

def compute_bias(bis_values, target=50):
    """
    Median percentage error.
    Positive = underdosing (BIS too high), negative = overdosing.
    Good performance: within ±10-20%
    """
    errors = (bis_values - target) / target * 100
    return float(np.median(errors))


def compute_inaccuracy(bis_values, target=50):
    """
    Median absolute percentage error.
    Measures overall error regardless of direction.
    Good performance: below 20%
    """
    errors = (bis_values - target) / target * 100
    return float(np.median(np.abs(errors)))


def compute_variability(bis_values, target=50):
    """
    Median deviation from the bias (wobble).
    High = oscillating control, low = steady control.
    Good performance: below 15%
    """
    errors = (bis_values - target) / target * 100
    bias = np.median(errors)
    return float(np.median(np.abs(errors - bias)))


def compute_drift(bis_values, time_minutes):
    """
    How error changes over time (slope).
    Positive = getting worse, negative = improving.
    """
    if len(bis_values) < 10:
        return 0.0

    errors = np.abs(bis_values - 50)
    slope, _ = np.polyfit(time_minutes, errors, 1)
    return float(slope)


def time_in_range(bis_values, low, high):
    """Percentage of time BIS stays within [low, high]."""
    return float(np.mean((bis_values >= low) & (bis_values <= high)) * 100)


# --- Main evaluation function ---

def evaluate_controller(controller, env, n_episodes=20, verbose=True):
    """
    Run episodes and compute all performance metrics.

    Returns dict with:
    - bias, inaccuracy, variability, drift (Varvel metrics)
    - time_in_target_40_60, time_in_target_45_55
    - mean_bis, std_bis
    - termination_rate
    - mean_reward, std_reward
    """
    all_bis = []
    episode_rewards = []
    episode_lengths = []
    terminations = []
    episode_drifts = []

    for ep in range(n_episodes):
        # reset controller if it has state (like PID)
        if hasattr(controller, 'reset'):
            controller.reset()

        obs, info = env.reset()
        ep_reward = 0
        ep_bis = []
        ep_times = []

        done = False
        while not done:
            action, _ = controller.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            all_bis.append(info['bis'])
            ep_bis.append(info['bis'])
            ep_times.append(info['time_min'])
            ep_reward += reward
            done = terminated or truncated

        episode_rewards.append(ep_reward)
        episode_lengths.append(len(ep_bis))
        terminations.append(terminated)

        # per-episode drift
        ep_drift = compute_drift(np.array(ep_bis), np.array(ep_times))
        episode_drifts.append(ep_drift)

        if verbose and (ep + 1) % 5 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: reward={ep_reward:.0f}, "
                  f"steps={len(ep_bis)}, final BIS={info['bis']:.1f}")

    bis = np.array(all_bis)

    return {
        # Varvel metrics
        'bias': compute_bias(bis),
        'inaccuracy': compute_inaccuracy(bis),
        'variability': compute_variability(bis),
        'drift': np.mean(episode_drifts),

        # target adherence
        'time_in_target_40_60': time_in_range(bis, 40, 60),
        'time_in_target_45_55': time_in_range(bis, 45, 55),

        # BIS stats
        'mean_bis': float(np.mean(bis)),
        'std_bis': float(np.std(bis)),

        # safety
        'time_below_30': time_in_range(bis, 0, 30),
        'time_above_70': time_in_range(bis, 70, 100),
        'termination_rate': float(np.mean(terminations) * 100),

        # rewards
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'n_episodes': n_episodes,
    }


def collect_trajectories(controller, env, n_episodes=5):
    """
    Run episodes and collect full trajectory data for plotting.
    Returns list of dicts with 'time', 'bis', 'action', 'reward' arrays.
    """
    episodes = []

    for ep in range(n_episodes):
        if hasattr(controller, 'reset'):
            controller.reset()

        obs, info = env.reset()
        data = {'time': [], 'bis': [], 'action': [], 'reward': []}

        done = False
        while not done:
            action, _ = controller.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            data['time'].append(info['time_min'])
            data['bis'].append(info['bis'])
            data['action'].append(float(action[0]))
            data['reward'].append(reward)
            done = terminated or truncated

        # convert to arrays
        for key in data:
            data[key] = np.array(data[key])

        episodes.append(data)

    return episodes


# --- Reporting ---

def print_report(metrics, name):
    """Print a readable summary of controller performance."""

    # interpret bias
    if abs(metrics['bias']) <= 10:
        bias_rating = "excellent"
    elif abs(metrics['bias']) <= 20:
        bias_rating = "acceptable"
    else:
        bias_rating = "poor"

    # interpret inaccuracy
    if metrics['inaccuracy'] <= 20:
        inac_rating = "excellent"
    elif metrics['inaccuracy'] <= 30:
        inac_rating = "acceptable"
    else:
        inac_rating = "poor"

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"\n  Clinical Metrics (Varvel)")
    print(f"  -------------------------")
    print(f"  Bias:        {metrics['bias']:+.1f}%  ({bias_rating})")
    print(f"  Inaccuracy:  {metrics['inaccuracy']:.1f}%  ({inac_rating})")
    print(f"  Variability: {metrics['variability']:.1f}%")
    print(f"  Drift:       {metrics['drift']:.4f} %/min")

    print(f"\n  Target Adherence")
    print(f"  -------------------------")
    print(f"  Time in 40-60: {metrics['time_in_target_40_60']:.1f}%")
    print(f"  Time in 45-55: {metrics['time_in_target_45_55']:.1f}%")

    print(f"\n  Safety")
    print(f"  -------------------------")
    print(f"  BIS < 30:      {metrics['time_below_30']:.1f}%")
    print(f"  BIS > 70:      {metrics['time_above_70']:.1f}%")
    print(f"  Terminations:  {metrics['termination_rate']:.1f}%")

    print(f"\n  Summary")
    print(f"  -------------------------")
    print(f"  Mean BIS:    {metrics['mean_bis']:.1f} ± {metrics['std_bis']:.1f}")
    print(f"  Mean Reward: {metrics['mean_reward']:.0f} ± {metrics['std_reward']:.0f}")
    print(f"  Episodes:    {metrics['n_episodes']}")
    print(f"{'='*50}\n")


def save_metrics(metrics, name, save_dir='./results'):
    """Save metrics to JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{name}_metrics.json')

    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved: {path}")
