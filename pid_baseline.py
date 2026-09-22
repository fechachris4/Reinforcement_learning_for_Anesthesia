"""
PID baseline, built the way a clinical closed-loop system would be:

- Induction: age-adjusted bolus (2.0 mg/kg under 55 years, 1.5 mg/kg from 55),
  given at the maximum bolus rate, then stop.
- Maintenance: PI control on the filtered BIS, with a weight-scaled
  feed-forward infusion and anti-windup.

Gains are tuned by grid search on a separate tuning population
(patients.tuning_patients), never on the test patients. See tune_pid.py.
"""

import numpy as np

from AnesthesiaEnv import BOLUS_BUDGET, DT

# defaults are overwritten by results/pid_gains.json when present
DEFAULT_GAINS = {'u0': 0.45, 'kp': 0.02, 'ki': 0.004}


class PIDController:
    def __init__(self, u0=None, kp=None, ki=None):
        g = dict(DEFAULT_GAINS)
        g.update({k: v for k, v in {'u0': u0, 'kp': kp, 'ki': ki}.items() if v is not None})
        self.u0, self.kp, self.ki = g['u0'], g['kp'], g['ki']
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.bolus_target = None
        self.bolus_given = 0.0

    def predict(self, obs, deterministic=True):
        error = obs[0] * 50.0                 # filtered BIS - 50
        in_maintenance = obs[6] > 0.5
        age = obs[7] * 100.0

        if not in_maintenance:
            if self.bolus_target is None:
                self.bolus_target = 2.0 if age < 55 else 1.5
            bolus_left = obs[5] * BOLUS_BUDGET
            self.bolus_given = BOLUS_BUDGET - bolus_left
            give = 1.0 if self.bolus_given < self.bolus_target - 1e-6 else 0.0
            return np.array([0.0, give], dtype=np.float32), None

        u_unsat = self.u0 + self.kp * error + self.ki * self.integral
        u = float(np.clip(u_unsat, 0.0, 1.0))
        # anti-windup: only integrate when not pushing further into saturation
        if (0.0 < u_unsat < 1.0) or (u_unsat >= 1.0 and error < 0) or (u_unsat <= 0.0 and error > 0):
            self.integral += error * DT
        return np.array([u, 0.0], dtype=np.float32), None


def load_pid(path='results/pid_gains.json') -> PIDController:
    import json, os
    if os.path.exists(path):
        with open(path) as f:
            return PIDController(**json.load(f))
    return PIDController()
