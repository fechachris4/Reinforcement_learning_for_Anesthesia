"""
Evaluation on held-out patients.

Every controller runs on the same test patients with the same monitor noise
and the same surgical-stimulation timing (common random numbers), so
differences come from the controller, not from luck.

Metrics (maintenance phase, after induction):
  time_in_target  % of time true BIS in [40, 60]
  MDPE            median performance error, bias (Varvel et al. 1992)
  MDAPE           median absolute performance error, inaccuracy
  wobble          median |PE - MDPE|, variability
  time_below_40   % of time too deep
  induction_min   time until measured BIS < 60
  propofol        mean maintenance dose, mg/kg/h
"""

import numpy as np

from AnesthesiaEnv import AnesthesiaEnv, MAX_STEPS, BIS_TARGET
from patients import test_patients

TEST_NOISE_OFFSET = 1000


def run_episode(controller, patient, noise_seed):
    env = AnesthesiaEnv(patient=patient, noise_seed=noise_seed)
    if hasattr(controller, 'reset'):
        controller.reset()
    obs, info = env.reset()
    trace = [info]
    for _ in range(MAX_STEPS):
        action, _ = controller.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        trace.append(info)
        if term or trunc:
            break
    return trace


def episode_metrics(trace):
    t = np.array([s['time_min'] for s in trace])
    bis = np.array([s['bis'] for s in trace])
    maint = np.array([s['phase'] == 'maintenance' for s in trace])
    ind_idx = int(np.argmax(maint)) if maint.any() else len(trace) - 1
    b = bis[maint]
    pe = (b - BIS_TARGET) / BIS_TARGET * 100
    mdpe = float(np.median(pe))
    inf = np.array([s['infusion_mgkgh'] for s in trace])[maint]
    return {
        'time_in_target': float(np.mean((b >= 40) & (b <= 60)) * 100),
        'MDPE': mdpe,
        'MDAPE': float(np.median(np.abs(pe))),
        'wobble': float(np.median(np.abs(pe - mdpe))),
        'time_below_40': float(np.mean(b < 40) * 100),
        'min_bis': float(bis.min()),
        'induction_min': float(t[ind_idx]),
        'propofol_mgkgh': float(inf.mean()),
        # whole case, induction included
        'case_below_40': float(np.mean(bis < 40) * 100),
        'case_above_60': float(np.mean(bis[t > 1.0] > 60) * 100),   # ignore the first minute (awake at start)
        'reached_below_20': float(bis.min() < 20) * 100,
    }


def evaluate(controller, n_patients=30):
    per_patient = []
    for i, p in enumerate(test_patients(n_patients)):
        per_patient.append(episode_metrics(run_episode(controller, p, TEST_NOISE_OFFSET + i)))
    keys = per_patient[0].keys()
    summary = {k: float(np.mean([m[k] for m in per_patient])) for k in keys}
    return summary, per_patient
