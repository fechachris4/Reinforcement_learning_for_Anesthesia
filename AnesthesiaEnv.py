"""
Closed-loop propofol anaesthesia environment (Gymnasium).

The controller only sees what an anaesthetist sees: a delayed, noisy BIS
signal, the pump history and basic patient covariates. It does not see drug
concentrations or the patient's hidden PK/PD parameters.

Episode (40 min, 5 s control interval):
  1. Induction: bolus from awake (BIS ~93) until measured BIS < 60,
     the bolus budget runs out, or 3 min pass.
  2. Maintenance: continuous infusion, target BIS 50 (clinical range 40-60),
     while surgical stimulation pushes BIS up at random times.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from EleveldPatient import EleveldPatient
from patients import sample_patient

# timing
DT_S = 5.0
DT = DT_S / 60.0                  # minutes
EPISODE_MIN = 40.0
MAX_STEPS = int(EPISODE_MIN * 60 / DT_S)

# dosing limits (weight-scaled)
MAX_INFUSION = 1 / 3              # mg/kg/min  (20 mg/kg/h)
MAX_BOLUS_RATE = 4.0              # mg/kg/min
BOLUS_BUDGET = 2.5                # mg/kg
MIN_BOLUS = 1.0                   # mg/kg; induction always gives at least this
INDUCTION_TIMEOUT_MIN = 3.0       # after the minimum bolus, switch to maintenance by 3 min

# BIS monitor model
BIS_TARGET = 50.0
BIS_INDUCTION_DONE = 60.0
BIS_DELAY_S = 20.0
BIS_NOISE_SD = 3.0
FILTER_ALPHA = 0.3
DELAY_STEPS = int(BIS_DELAY_S / DT_S)
BIS_UNSAFE = 25.0                 # below this: deep overdose (burst suppression risk)

# reward
TRACK_WIDTH = 10.0
OVERDOSE_W = 0.03                 # per BIS point below 40
LIGHT_W = 0.03                    # per BIS point above 60 (awake or too light)
SMOOTH_W = 0.5
DRUG_W = 0.05
BOLUS_W = 0.1
UNSAFE_PENALTY = 0.5              # per step spent below BIS_UNSAFE

OBS_DIM = 10
# indices into the observation vector (see _obs)
OBS_BIS_FILTERED, OBS_BOLUS_LEFT, OBS_MAINTENANCE, OBS_AGE = 0, 5, 6, 7


def stimulation_schedule(rng: np.random.Generator) -> list:
    """Surgical stimulation events: (start_min, rise_min, hold_min, fall_min, amplitude)."""
    events = [(rng.uniform(8, 12), 1.0, rng.uniform(3, 8), 2.0, rng.uniform(8, 15))]
    if rng.random() < 0.7:
        events.append((rng.uniform(20, 30), 1.0, rng.uniform(2, 6), 2.0, rng.uniform(5, 12)))
    return events


def disturbance_at(t: float, events: list) -> float:
    d = 0.0
    for start, rise, hold, fall, amp in events:
        x = t - start
        if x < 0:
            continue
        if x < rise:
            d += amp * x / rise
        elif x < rise + hold:
            d += amp
        elif x < rise + hold + fall:
            d += amp * (1 - (x - rise - hold) / fall)
    return d


class AnesthesiaEnv(gym.Env):
    """
    Args:
        patient: fixed patient dict (from patients.py). If None, a new patient
            is sampled at every reset (training).
        noise_seed: fixes monitor noise and stimulation timing, so different
            controllers can be compared on identical conditions.
    """

    metadata = {'render_modes': []}

    def __init__(self, patient: dict = None, noise_seed: int = None):
        super().__init__()
        self.action_space = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)  # [infusion, bolus]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self.fixed_patient = patient
        self.noise_seed = noise_seed
        self.patient_rng = np.random.default_rng()

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.patient_rng = np.random.default_rng(seed)

        self.patient_spec = self.fixed_patient or sample_patient(self.patient_rng)
        self.patient = EleveldPatient(opioid_switch=True, **self.patient_spec)
        self.weight = self.patient.weight

        noise_rng_seed = self.noise_seed if self.noise_seed is not None else int(self.patient_rng.integers(1 << 31))
        self.noise_rng = np.random.default_rng(noise_rng_seed)
        self.events = stimulation_schedule(self.noise_rng)

        self.step_count = 0
        self.in_maintenance = False
        self.bolus_left = BOLUS_BUDGET
        self.true_bis = self.patient.E0
        self.bis_history = [self.true_bis] * (DELAY_STEPS + 1)
        self.measured = self._measure()
        self.filtered = self.measured
        self.filtered_hist = [self.filtered] * 7
        self.last_infusion = 0.0
        self.infusion_hist = [0.0] * int(300 / DT_S)
        return self._obs(), self._info(0.0, 0.0)

    # ------------------------------------------------------------------- step
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0)
        t = self.step_count * DT

        # phase switch is decided on the measured signal, as a clinician would
        given = BOLUS_BUDGET - self.bolus_left
        if (not self.in_maintenance and given >= MIN_BOLUS - 1e-9
                and (self.filtered <= BIS_INDUCTION_DONE or self.bolus_left <= 1e-9 or t >= INDUCTION_TIMEOUT_MIN)):
            self.in_maintenance = True

        bolus_mgkg = 0.0
        infusion = 0.0
        if self.in_maintenance:
            infusion = float(a[0])
            smooth = SMOOTH_W * (infusion - self.last_infusion) ** 2
            self.last_infusion = infusion
            u_prop = infusion * MAX_INFUSION * self.weight          # mg/min
            self.patient.step(DT, u_prop, 0.0)
        else:
            smooth = 0.0
            requested = float(a[1]) * MAX_BOLUS_RATE * DT
            if given < MIN_BOLUS:   # no skipping induction: the first 1 mg/kg always goes in
                requested = max(requested, min(MAX_BOLUS_RATE * DT, MIN_BOLUS - given))
            bolus_mgkg = min(requested, self.bolus_left)
            self.bolus_left -= bolus_mgkg
            self.patient.add_bolus(bolus_mgkg * self.weight)
            self.patient.step(DT, 0.0, 0.0)

        ce, _ = self.patient.get_effect_site_concentrations()
        drug_bis = float(self.patient.get_bis(ce, 0.0))
        self.true_bis = float(np.clip(drug_bis + disturbance_at(t + DT, self.events), 0, 100))

        # monitor: delay + noise + light filtering
        self.bis_history.append(self.true_bis)
        self.bis_history.pop(0)
        self.measured = self._measure()
        self.filtered = FILTER_ALPHA * self.measured + (1 - FILTER_ALPHA) * self.filtered
        self.filtered_hist.append(self.filtered)
        self.filtered_hist.pop(0)
        self.infusion_hist.append(infusion)
        self.infusion_hist.pop(0)

        # reward uses the true BIS (available in simulation only, never observed)
        err = self.true_bis - BIS_TARGET
        reward = (np.exp(-err ** 2 / (2 * TRACK_WIDTH ** 2))
                  - OVERDOSE_W * max(0.0, 40.0 - self.true_bis)
                  - LIGHT_W * max(0.0, self.true_bis - 60.0)
                  - smooth
                  - DRUG_W * infusion
                  - BOLUS_W * bolus_mgkg)

        if self.true_bis < BIS_UNSAFE:
            reward -= UNSAFE_PENALTY

        self.step_count += 1
        terminated = False     # episodes always run the full 40 min so every controller is scored on the same window
        truncated = self.step_count >= MAX_STEPS
        return self._obs(), float(reward), bool(terminated), bool(truncated), self._info(infusion, bolus_mgkg)

    # ---------------------------------------------------------------- helpers
    def _measure(self) -> float:
        return float(np.clip(self.bis_history[0] + self.noise_rng.normal(0, BIS_NOISE_SD), 0, 100))

    def _obs(self) -> np.ndarray:
        trend = (self.filtered_hist[-1] - self.filtered_hist[0]) / (6 * DT)   # BIS per min over 30 s
        return np.array([
            (self.filtered - BIS_TARGET) / 50.0,
            (self.measured - BIS_TARGET) / 50.0,
            np.clip(trend / 20.0, -3, 3),
            self.last_infusion,
            float(np.mean(self.infusion_hist)),
            self.bolus_left / BOLUS_BUDGET,
            float(self.in_maintenance),
            self.patient.age / 100.0,
            self.weight / 100.0,
            self.step_count / MAX_STEPS,
        ], dtype=np.float32)

    def _info(self, infusion: float, bolus_mgkg: float) -> dict:
        ce, _ = self.patient.get_effect_site_concentrations()
        return {
            'time_min': self.step_count * DT,
            'bis': self.true_bis,
            'bis_measured': self.measured,
            'bis_filtered': self.filtered,
            'disturbance': disturbance_at(self.step_count * DT, self.events),
            'infusion_mgkgh': infusion * MAX_INFUSION * 60.0,
            'bolus_mgkg': bolus_mgkg,
            'ce': float(ce),
            'phase': 'maintenance' if self.in_maintenance else 'induction',
        }


if __name__ == '__main__':
    env = AnesthesiaEnv()
    obs, info = env.reset(seed=0)
    total = 0.0
    for _ in range(MAX_STEPS):
        obs, r, term, trunc, info = env.step([0.45, 1.0])
        total += r
        if term or trunc:
            break
    print(f"sanity run: steps={env.step_count} final BIS={info['bis']:.1f} return={total:.1f}")
