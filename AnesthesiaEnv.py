"""
Anaesthesia RL Environment

Controls BIS (depth of anaesthesia) using propofol.
Two phases: induction (bolus) then maintenance (infusion).
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from EleveldPatient import EleveldPatient


# config
max_bolus_rate = 6.0       # mg/kg/min
max_infusion_rate = 12.0   # mg/min
bolus_budget = 3.0         # mg/kg total

bis_target = 50.0
bis_induction_threshold = 60.0
bis_danger_low = 20.0
bis_danger_high = 95.0

# reward weights
r_bis_scale = 4.0
r_bis_width = 10.0
r_bolus_cost = 0.8
r_smoothness = 0.8
r_transition_penalty = 1.0
r_safety_penalty = 100.0

# timing
dt = 1/60                  # 1 second
max_steps = 2400           # 40 min episode
grace_period = 120         # 2 min before safety checks


# patient pool (used for both training and evaluation)
patients = [
    {'age': 40, 'weight': 70, 'height': 170, 'gender': 'm'},
    {'age': 25, 'weight': 60, 'height': 165, 'gender': 'f'},
    {'age': 70, 'weight': 80, 'height': 175, 'gender': 'm'},
    {'age': 45, 'weight': 100, 'height': 185, 'gender': 'm'},
    {'age': 35, 'weight': 55, 'height': 160, 'gender': 'f'},
]


class AnesthesiaEnv(gym.Env):

    def __init__(self, fixed_patient=None, cycle_patients=True):
        super().__init__()
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.fixed_patient = fixed_patient
        self.cycle_patients = cycle_patients and (fixed_patient is None)

        self.patient = None
        self.step_count = 0
        self.prev_bis = 93.0
        self.bis_delta = 0.0              # [FIX #2] store bis trend for observation
        self.bolus_left = bolus_budget
        self.in_maintenance = False
        self.first_maintenance_step = False  # [FIX #1] track first maintenance step
        self.last_infusion_action = 0.0   # [FIX #1] track infusion actions separately
        self._patient_idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # select patient
        if self.fixed_patient:
            p = self.fixed_patient
        elif self.cycle_patients:
            p = patients[self._patient_idx]
            self._patient_idx = (self._patient_idx + 1) % len(patients)
        else:
            p = patients[0]

        self.patient = EleveldPatient(**p)
        self.step_count = 0
        self.prev_bis = 93.0
        self.bis_delta = 0.0              # [FIX #2] reset bis trend
        self.bolus_left = bolus_budget
        self.in_maintenance = False
        self.first_maintenance_step = False  # [FIX #1] reset flag
        self.last_infusion_action = 0.0   # [FIX #1] reset infusion tracker

        return self._get_obs(), {'bis': 93.0, 'phase': 'induction'}

    def step(self, action):
        action = np.clip(action, 0, 1)
        act_inf = action[0]
        act_bol = action[1]

        # check phase transition
        transition_pen = 0.0
        if not self.in_maintenance:
            if self.prev_bis <= bis_induction_threshold or self.bolus_left <= 0:
                self.in_maintenance = True
                self.first_maintenance_step = True  # [FIX #1] mark first maintenance step
                # penalty if bolus ran out but bis still high
                if self.prev_bis > bis_induction_threshold:
                    transition_pen = (self.prev_bis - bis_induction_threshold) * r_transition_penalty

        # execute action based on phase
        if not self.in_maintenance:
            # induction: use bolus only
            bol_rate = act_bol * max_bolus_rate
            bol_used = min(bol_rate * dt, self.bolus_left)
            self.bolus_left -= bol_used
            self.patient.add_bolus(bol_used * self.patient.weight)
            bis, map_val = self.patient.step(dt, 0, 0)
            inf_used = 0.0
            phase = 'induction'
            r_smooth = 0.0
        else:
            # maintenance: use infusion only
            inf_used = act_inf * max_infusion_rate
            bis, map_val = self.patient.step(dt, inf_used, 0)
            bol_used = 0.0
            phase = 'maintenance'

            # [FIX #1] smoothness penalty: zero on first maintenance step,
            # otherwise compare to previous infusion action
            if self.first_maintenance_step:
                r_smooth = 0.0
                self.first_maintenance_step = False
            else:
                delta = act_inf - self.last_infusion_action
                r_smooth = r_smoothness * (delta ** 2)

            self.last_infusion_action = act_inf

        # reward calculation
        r_bis = r_bis_scale * np.exp(-((bis - bis_target)**2) / (2 * r_bis_width**2))
        r_bol = r_bolus_cost * bol_used

        # [FIX #2] compute bis delta BEFORE updating prev_bis
        self.bis_delta = bis - self.prev_bis

        # [FIX #4] safety check: accumulate penalties for multiple violations
        # [FIX #5] grace_period >= means first 120 steps (0-119) are safe, checks start at step 120
        r_safe = 0.0
        terminated = False
        if self.step_count >= grace_period:
            if bis < bis_danger_low or bis > bis_danger_high:
                r_safe += r_safety_penalty  # [FIX #4] use += to accumulate
                terminated = True
            if map_val < 50 or map_val > 110:
                r_safe += r_safety_penalty  # [FIX #4] use += to accumulate
                terminated = True

        reward = r_bis - r_bol - r_smooth - transition_pen - r_safe

        # update state
        self.step_count += 1
        self.prev_bis = bis  # [FIX #2] update AFTER computing bis_delta
        truncated = self.step_count >= max_steps

        ce, _ = self.patient.get_effect_site_concentrations()

        info = {
            'bis': bis,
            'map': map_val,
            'phase': phase,
            'bolus_remaining': self.bolus_left,
            'applied_infusion_mg_min': inf_used,
            'applied_bolus_mg_kg_min': bol_used / dt if bol_used > 0 else 0,
            'ce_prop': ce,
            'time_min': self.step_count * dt,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        ce, _ = self.patient.get_effect_site_concentrations()
        bis = self.patient.get_bis(ce, 0)

        return np.array([
            (bis - bis_target) / 100.0,          # bis error
            self.bis_delta / 100.0,              # [FIX #2] use stored delta, not recomputed
            self.last_infusion_action,           # [FIX #1] use infusion action tracker
            ce / 5.0,                            # drug concentration
            self.bolus_left / bolus_budget,      # bolus remaining
            float(self.in_maintenance),          # phase flag
        ], dtype=np.float32)


class EvaluationEnv(AnesthesiaEnv):
    """Same env but cycles through patients deterministically for evaluation."""

    # [FIX #3] removed separate test_patients list - now uses same patients as training

    def __init__(self):
        # [FIX #6] don't call super().__init__() which would set up patient cycling
        # instead, initialize directly to avoid creating patient twice
        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)

        self.fixed_patient = None
        self.cycle_patients = False  # we handle cycling ourselves

        self.patient = None
        self.step_count = 0
        self.prev_bis = 93.0
        self.bis_delta = 0.0
        self.bolus_left = bolus_budget
        self.in_maintenance = False
        self.first_maintenance_step = False
        self.last_infusion_action = 0.0
        self._patient_idx = 0
        self._eval_idx = 0

    def reset(self, seed=None, options=None):
        # [FIX #6] create patient once, using shared patients list
        gym.Env.reset(self, seed=seed)

        # [FIX #3] use same patients list as training environment
        p = patients[self._eval_idx]
        self._eval_idx = (self._eval_idx + 1) % len(patients)

        self.patient = EleveldPatient(**p)
        self.step_count = 0
        self.prev_bis = 93.0
        self.bis_delta = 0.0
        self.bolus_left = bolus_budget
        self.in_maintenance = False
        self.first_maintenance_step = False
        self.last_infusion_action = 0.0

        return self._get_obs(), {'bis': 93.0, 'phase': 'induction'}


if __name__ == "__main__":
    env = AnesthesiaEnv()
    obs, info = env.reset()

    print("testing environment...")
    print(f"initial bis: {info['bis']}")

    # induction
    for i in range(30):
        obs, r, done, trunc, info = env.step([0.5, 1.0])
    print(f"after induction: bis={info['bis']:.1f}, bolus_left={info['bolus_remaining']:.2f}")

    # maintenance
    for i in range(300):
        obs, r, done, trunc, info = env.step([0.6, 0.0])
        if done:
            break
    print(f"final: bis={info['bis']:.1f}, phase={info['phase']}")

    # test bug fixes
    print("\n--- Bug Fix Verification ---")

    # Test #1: Smoothness penalty
    env2 = AnesthesiaEnv()
    env2.reset()
    env2.in_maintenance = True
    env2.first_maintenance_step = True
    env2.last_infusion_action = 0.0

    # First maintenance step should have zero smoothness penalty
    obs, r1, _, _, _ = env2.step([0.5, 0.0])
    env2.reset()
    env2.in_maintenance = True
    env2.first_maintenance_step = False
    env2.last_infusion_action = 0.0
    obs, r2, _, _, _ = env2.step([0.5, 0.0])
    print(f"Smoothness test: first_step_reward={r1:.3f}, normal_step_reward={r2:.3f}")
    print(f"  (normal step should have lower reward due to smoothness penalty)")

    # Test #2: BIS trend
    env3 = AnesthesiaEnv()
    env3.reset()
    for _ in range(10):
        obs, _, _, _, info = env3.step([0.0, 1.0])
    print(f"BIS trend obs[1]: {obs[1]:.4f} (should be non-zero when BIS is changing)")
