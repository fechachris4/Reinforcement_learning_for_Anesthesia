"""
Full evaluation: SAC vs PID comparison with plots.

Usage:
    python run_evaluation.py
    python run_evaluation.py --model ./models/sac_model --episodes 20
"""

import argparse
import os
import pickle
from stable_baselines3 import SAC

from AnesthesiaEnv import EvaluationEnv
from evaluate import evaluate_controller, collect_trajectories, print_report
from pid_baseline import InductionPIDController
from visualize import generate_all_plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./models/sac_model')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--output', default='./figures')
    args = parser.parse_args()

    print("=" * 60)
    print("SAC vs PID Evaluation")
    print("=" * 60)

    # separate envs for fair comparison (same patient sequence)
    env_sac = EvaluationEnv()
    env_pid = EvaluationEnv()

    # load SAC model
    print(f"\nLoading model: {args.model}")
    sac = SAC.load(args.model)
    pid = InductionPIDController()

    # evaluate both controllers
    print(f"\nEvaluating SAC ({args.episodes} episodes)...")
    sac_metrics = evaluate_controller(sac, env_sac, n_episodes=args.episodes)
    print_report(sac_metrics, "SAC Agent")

    print(f"\nEvaluating PID ({args.episodes} episodes)...")
    pid_metrics = evaluate_controller(pid, env_pid, n_episodes=args.episodes)
    print_report(pid_metrics, "PID Controller")

    # collect trajectories for plotting
    print("\nCollecting trajectories for plots...")
    env_sac.reset(seed=42)
    env_pid.reset(seed=42)
    sac_traj = collect_trajectories(sac, EvaluationEnv(), n_episodes=6)
    pid_traj = collect_trajectories(pid, EvaluationEnv(), n_episodes=6)

    # save data for later
    os.makedirs('./results', exist_ok=True)
    with open('./results/evaluation_data.pkl', 'wb') as f:
        pickle.dump({
            'sac_trajectories': sac_traj,
            'pid_trajectories': pid_traj,
            'sac_metrics': sac_metrics,
            'pid_metrics': pid_metrics,
        }, f)
    print("Saved: ./results/evaluation_data.pkl")

    # generate plots
    print("\nGenerating plots...")
    generate_all_plots(sac_traj, sac_metrics, pid_traj, pid_metrics, args.output)

    # comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<20} {'SAC':>12} {'PID':>12}")
    print("-" * 50)
    print(f"{'Bias':<20} {sac_metrics['bias']:>+11.1f}% {pid_metrics['bias']:>+11.1f}%")
    print(f"{'Inaccuracy':<20} {sac_metrics['inaccuracy']:>11.1f}% {pid_metrics['inaccuracy']:>11.1f}%")
    print(f"{'Variability':<20} {sac_metrics['variability']:>11.1f}% {pid_metrics['variability']:>11.1f}%")
    print(f"{'TiT (40-60)':<20} {sac_metrics['time_in_target_40_60']:>11.1f}% {pid_metrics['time_in_target_40_60']:>11.1f}%")
    print(f"{'Terminations':<20} {sac_metrics['termination_rate']:>11.1f}% {pid_metrics['termination_rate']:>11.1f}%")
    print(f"{'Mean Reward':<20} {sac_metrics['mean_reward']:>12.0f} {pid_metrics['mean_reward']:>12.0f}")
    print("-" * 50)

    # determine winner
    sac_wins = sum([
        abs(sac_metrics['bias']) < abs(pid_metrics['bias']),
        sac_metrics['inaccuracy'] < pid_metrics['inaccuracy'],
        sac_metrics['time_in_target_40_60'] > pid_metrics['time_in_target_40_60'],
        sac_metrics['termination_rate'] < pid_metrics['termination_rate'],
    ])

    if sac_wins >= 3:
        print(f"SAC wins ({sac_wins}/4 metrics)")
    elif sac_wins <= 1:
        print(f"PID wins ({4-sac_wins}/4 metrics)")
    else:
        print("Close match!")

    print(f"\nPlots saved to: {args.output}/")

    env_sac.close()
    env_pid.close()


if __name__ == '__main__':
    main()
