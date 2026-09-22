"""
Run controllers on the test patients. Every controller gets the same monitor
noise and stimulation timing for a given patient (common random numbers).

Metrics use true BIS over 5-40 min, except bolus_mgkg, time_to_60_min,
reached_below_20 and min_bis, which cover the whole case.
PE/MDPE/MDAPE/wobble as in Varvel et al. 1992.
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


WINDOW = (5.0, 40.0)   # min; every controller is scored over the same stretch


def episode_metrics(trace):
    t = np.array([s['time_min'] for s in trace])
    bis = np.array([s['bis'] for s in trace])
    w = (t >= WINDOW[0]) & (t <= WINDOW[1])
    b = bis[w]
    pe = (b - BIS_TARGET) / BIS_TARGET * 100
    mdpe = float(np.median(pe))
    maint = np.array([s['phase'] == 'maintenance' for s in trace])
    below60 = np.flatnonzero(bis < 60)
    return {
        'time_in_target': float(np.mean((b >= 40) & (b <= 60)) * 100),
        'MDPE': mdpe,
        'MDAPE': float(np.median(np.abs(pe))),
        'wobble': float(np.median(np.abs(pe - mdpe))),
        'time_below_40': float(np.mean(b < 40) * 100),
        'time_above_60': float(np.mean(b > 60) * 100),
        'propofol_mgkgh': float(np.mean([s['infusion_mgkgh'] for s in trace][int(np.argmax(maint)):])),
        'bolus_mgkg': float(sum(s['bolus_mgkg'] for s in trace)),
        'time_to_60_min': float(t[below60[0]]) if len(below60) else float('nan'),
        'reached_below_20': float(bis.min() < 20) * 100,
        'min_bis': float(bis.min()),
    }


def evaluate(controller, n_patients=30):
    per_patient = []
    for i, p in enumerate(test_patients(n_patients)):
        per_patient.append(episode_metrics(run_episode(controller, p, TEST_NOISE_OFFSET + i)))
    keys = per_patient[0].keys()
    summary = {k: float(np.mean([m[k] for m in per_patient])) for k in keys}
    return summary, per_patient
