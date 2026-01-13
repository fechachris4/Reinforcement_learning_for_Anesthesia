"""
SAC Training Script for Closed-Loop Anaesthesia Control
========================================================

Trains a Soft Actor-Critic agent to control BIS via two-phase dosing:
- Induction: Bolus delivery
- Maintenance: Continuous infusion

Usage:
    python train_sac.py                          # Default 100k steps
    python train_sac.py --total_timesteps 500000 # Longer training
"""

import os
import argparse
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from AnesthesiaEnv import AnesthesiaEnv, EvaluationEnv

# =============================================================================
# HYPERPARAMETERS
# =============================================================================

LEARNING_RATE = 3e-4
BUFFER_SIZE = 100_000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
EVAL_FREQ = 5000
N_EVAL_EPISODES = 10

# Paths
SAVE_DIR = "./models"
FIGURES_DIR = "./figures"

# Entropy estimation
DEFAULT_INITIAL_ENTROPY = 2.0  # Approximate entropy before replay buffer fills


class TrainingCallback(BaseCallback):
    """Callback to log all data needed for publication figures."""

    def __init__(self, eval_env, eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES,
                 checkpoint_steps=None, save_dir=SAVE_DIR, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.checkpoint_steps = checkpoint_steps or []
        self.save_dir = save_dir

        # Data storage
        self.timesteps = []
        self.mean_rewards = []
        self.std_rewards = []
        self.time_in_target = []
        self.entropy_coefs = []     # α values
        self.policy_entropies = []  # H(π) values

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._log_evaluation()

        if self.num_timesteps in self.checkpoint_steps:
            self._save_checkpoint()

        return True

    def _log_evaluation(self):
        """Run evaluation and log all metrics."""
        entropy_coef = torch.exp(self.model.log_ent_coef).item()
        policy_entropy = self._compute_policy_entropy()
        rewards, time_in_target_pcts = self._evaluate_episodes()

        self.timesteps.append(self.num_timesteps)
        self.mean_rewards.append(np.mean(rewards))
        self.std_rewards.append(np.std(rewards))
        self.time_in_target.append(np.mean(time_in_target_pcts))
        self.entropy_coefs.append(entropy_coef)
        self.policy_entropies.append(policy_entropy)

        if self.verbose:
            print(f"  [{self.num_timesteps:>7,}] "
                  f"Reward: {np.mean(rewards):>8.0f} +/- {np.std(rewards):>5.0f} | "
                  f"TiT: {np.mean(time_in_target_pcts):>4.0f}% | "
                  f"α: {entropy_coef:.3f} | "
                  f"H(π): {policy_entropy:.2f}")

    def _evaluate_episodes(self):
        """Run evaluation episodes, return (rewards, time_in_target_pcts)."""
        rewards, time_in_target_pcts = [], []
        for _ in range(self.n_eval_episodes):
            reward, time_in_target_pct = self._run_eval_episode()
            rewards.append(reward)
            time_in_target_pcts.append(time_in_target_pct)
        return rewards, time_in_target_pcts

    def _run_eval_episode(self):
        """Run single eval episode, return (total_reward, time_in_target_pct)."""
        obs, _ = self.eval_env.reset()
        total_reward, bis_values = 0, []

        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            total_reward += reward
            bis_values.append(info['bis'])
            done = terminated or truncated

        time_in_target_pct = np.mean([(40 <= b <= 60) for b in bis_values]) * 100
        return total_reward, time_in_target_pct

    def _save_checkpoint(self):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_{self.num_timesteps}")
        self.model.save(checkpoint_path)
        if self.verbose:
            print(f"  >> Checkpoint saved: {checkpoint_path}")

    def _compute_policy_entropy(self, n_samples=100):
        """Estimate policy entropy by sampling from replay buffer."""
        if self.model.replay_buffer.size() < n_samples:
            return DEFAULT_INITIAL_ENTROPY

        replay_data = self.model.replay_buffer.sample(n_samples)
        obs = replay_data.observations

        with torch.no_grad():
            _, log_std, _ = self.model.actor.get_action_dist_params(obs)
            # Entropy of Gaussian: 0.5 + 0.5*log(2π) + log(σ)
            entropy = 0.5 + 0.5 * np.log(2 * np.pi) + log_std
            entropy = entropy.mean().item()

        return entropy

    def get_data(self):
        """Return all logged data for figure generation."""
        return {
            'timesteps': np.array(self.timesteps),
            'mean_rewards': np.array(self.mean_rewards),
            'std_rewards': np.array(self.std_rewards),
            'time_in_target': np.array(self.time_in_target),
            'ent_coef': np.array(self.entropy_coefs),
            'entropy': np.array(self.policy_entropies),
        }


def train(total_timesteps=100_000, seed=42, eval_freq=EVAL_FREQ):
    """Train SAC agent with full logging for figures."""
    np.random.seed(seed)
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    _print_header(total_timesteps, seed)

    # Environments
    train_env = Monitor(AnesthesiaEnv())
    train_env.reset(seed=seed)
    eval_env = EvaluationEnv()

    # Checkpoints at 10%, 50%, 100% for action distribution figure
    checkpoint_steps = [
        int(total_timesteps * 0.1),
        int(total_timesteps * 0.5),
        total_timesteps,
    ]

    callback = TrainingCallback(
        eval_env=eval_env,
        eval_freq=eval_freq,
        checkpoint_steps=checkpoint_steps,
        verbose=1
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=LEARNING_RATE,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        tau=TAU,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        verbose=0,
        seed=seed,
    )

    # Train
    print("Training progress:")
    print(f"{'Step':>9} | {'Reward':>16} | {'TiT':>5} | {'α':>6} | {'H(π)':>6}")
    print("-" * 60)

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    _save_results(model, callback, checkpoint_steps)

    train_env.close()
    eval_env.close()

    return model


def _print_header(total_timesteps: int, seed: int):
    """Print training header."""
    print(f"\n{'='*60}")
    print(f"SAC Training | {total_timesteps:,} steps | seed={seed}")
    print(f"{'='*60}")
    print(f"Logging: rewards, α, H(π), checkpoints")
    print(f"{'='*60}\n")


def _save_results(model, callback, checkpoint_steps):
    """Save model and training logs."""
    model.save(os.path.join(SAVE_DIR, "sac_model"))
    print(f"\nModel saved to {SAVE_DIR}/sac_model")

    data = callback.get_data()
    data['checkpoint_steps'] = np.array(checkpoint_steps)
    np.savez(os.path.join(SAVE_DIR, "training_logs.npz"), **data)
    print(f"Training logs saved to {SAVE_DIR}/training_logs.npz")

    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Checkpoints at steps: {checkpoint_steps}")
    print(f"\nTo generate figures: python generate_training_figures.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAC for anaesthesia control")
    parser.add_argument("--total_timesteps", type=int, default=100_000,
                        help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--eval_freq", type=int, default=EVAL_FREQ,
                        help="Evaluation frequency")
    args = parser.parse_args()

    train(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        eval_freq=args.eval_freq
    )
