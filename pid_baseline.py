"""
PID controller baseline for BIS regulation.
Used for comparison against the SAC agent.
"""

import numpy as np


# PID gains (tuned for BIS control)
KP = 0.015
KI = 0.001
KD = 0.005
INTEGRAL_MAX = 50.0
BIS_TARGET = 50.0


class PIDController:
    """Basic PID controller for BIS regulation."""

    def __init__(self, target=BIS_TARGET):
        self.target = target
        self.integral = 0.0
        self.last_error = 0.0
        self.filtered_derivative = 0.0

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.filtered_derivative = 0.0

    def predict(self, obs, deterministic=True):
        """
        SB3-compatible interface.
        obs[0] = (BIS - 50) / 100, so BIS = obs[0] * 100 + 50
        """
        bis = obs[0] * 100 + 50
        error = bis - self.target
        dt = 1/60  # 1 second per step

        # PID terms
        p_term = KP * error

        self.integral += error * dt
        self.integral = np.clip(self.integral, -INTEGRAL_MAX, INTEGRAL_MAX)
        i_term = KI * self.integral

        raw_derivative = (error - self.last_error) / dt
        self.filtered_derivative = 0.8 * self.filtered_derivative + 0.2 * raw_derivative
        d_term = KD * self.filtered_derivative
        self.last_error = error

        # convert to action [0, 1]
        pid_output = p_term + i_term + d_term
        action = np.clip(0.3 + pid_output, 0.0, 1.0)

        return np.array([action, 0.0], dtype=np.float32), None


class InductionPIDController(PIDController):
    """
    Two-phase PID: bolus during induction, PID infusion during maintenance.
    Matches the environment's two-phase structure.
    """

    def __init__(self, target=BIS_TARGET):
        super().__init__(target)
        self.step_count = 0

    def reset(self):
        super().reset()
        self.step_count = 0

    def predict(self, obs, deterministic=True):
        self.step_count += 1
        dt = 1/60

        # check phase from observation (index 5 = in_maintenance flag)
        in_maintenance = len(obs) > 5 and obs[5] > 0.5

        if not in_maintenance:
            # induction phase: give full bolus, reset PID state
            self.integral = 0.0
            self.filtered_derivative = 0.0
            bis = obs[0] * 100 + 50
            self.last_error = bis - self.target
            return np.array([0.0, 1.0], dtype=np.float32), None

        # maintenance phase: use PID for infusion
        bis = obs[0] * 100 + 50
        error = bis - self.target

        p_term = KP * error

        self.integral += error * dt
        self.integral = np.clip(self.integral, -INTEGRAL_MAX, INTEGRAL_MAX)
        i_term = KI * self.integral

        raw_derivative = (error - self.last_error) / dt
        self.filtered_derivative = 0.8 * self.filtered_derivative + 0.2 * raw_derivative
        d_term = KD * self.filtered_derivative
        self.last_error = error

        action = np.clip(0.25 + p_term + i_term + d_term, 0.0, 1.0)
        return np.array([action, 0.0], dtype=np.float32), None
