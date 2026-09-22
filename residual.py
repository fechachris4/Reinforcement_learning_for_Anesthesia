"""
Residual reinforcement learning: PID runs underneath, SAC learns corrections.

At every step the PID proposes an action. The agent sees the normal
observation plus the PID's proposed infusion, and outputs two bounded
corrections:
  - maintenance: infusion = PID infusion + 0.15 * a[0]  (about ±3 mg/kg/h)
  - induction:   bolus size = PID bolus x (1 + 0.4 * a[1]) (0.6x to 1.4x)

Corrections carry a small quadratic cost, so the agent only departs from the
PID where that pays off.

With a = 0 the controller is exactly the tuned PID.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from AnesthesiaEnv import AnesthesiaEnv, OBS_DIM, OBS_MAINTENANCE
from pid_baseline import load_pid

INF_SCALE = 0.15
BOLUS_SCALE = 0.4
CORRECTION_COST = 0.05


def combine(pid, base, obs, delta):
    """Apply the agent's correction to the PID's proposed action."""
    a = np.array(base, dtype=np.float32)
    delta = np.clip(np.asarray(delta, dtype=np.float32), -1, 1)
    if obs[OBS_MAINTENANCE] > 0.5:
        a[0] = np.clip(base[0] + INF_SCALE * delta[0], 0.0, 1.0)
    else:                                             # induction
        target = pid.bolus_target * (1.0 + BOLUS_SCALE * delta[1])
        a[1] = 1.0 if pid.bolus_given < target - 1e-6 else 0.0
    return a


def augment(obs, base):
    return np.append(obs, np.float32(base[0])).astype(np.float32)


class ResidualEnv(gym.Env):
    def __init__(self, patient=None, noise_seed=None):
        super().__init__()
        self.env = AnesthesiaEnv(patient, noise_seed)
        self.pid = load_pid()
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM + 1,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.pid.reset()
        self.obs, info = self.env.reset(seed=seed)
        self.base, _ = self.pid.predict(self.obs)
        return augment(self.obs, self.base), info

    def step(self, delta):
        a = combine(self.pid, self.base, self.obs, delta)
        in_maintenance = self.obs[OBS_MAINTENANCE] > 0.5
        self.obs, r, term, trunc, info = self.env.step(a)
        d = np.clip(np.asarray(delta, dtype=np.float64), -1, 1)
        r -= CORRECTION_COST * (d[0] ** 2 if in_maintenance else d[1] ** 2)
        self.base, _ = self.pid.predict(self.obs)
        return augment(self.obs, self.base), r, term, trunc, info


class ResidualPolicy:
    """Wraps a trained residual SAC so it can run on the plain environment."""

    def __init__(self, model):
        self.model = model
        self.pid = load_pid()

    def reset(self):
        self.pid.reset()

    def predict(self, obs, deterministic=True):
        base, _ = self.pid.predict(obs)
        delta, _ = self.model.predict(augment(obs, base), deterministic=deterministic)
        return combine(self.pid, base, obs, delta), None
