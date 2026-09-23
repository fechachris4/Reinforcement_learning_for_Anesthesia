"""Fast checks that need no trained model: patient model, environment, PID baseline.

Run from the repo root:  python -m unittest discover tests
"""

import json
import os
import sys
import unittest

import numpy as np
from scipy.integrate import solve_ivp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from AnesthesiaEnv import AnesthesiaEnv  # noqa: E402
from EleveldPatient import EleveldPatient  # noqa: E402
from evaluate import evaluate  # noqa: E402
from patients import test_patients, tuning_patients  # noqa: E402
from pid_baseline import load_pid  # noqa: E402


class PatientModelTest(unittest.TestCase):
    def test_matrix_exponential_step_matches_ode_solver(self):
        # 2 mg/kg bolus, then 8 mg/kg/h from 3 min, 5 s steps over 40 min
        p = EleveldPatient(60, 70, 'm', 170)
        p.add_bolus(140)
        y, dt, worst = p.state.copy(), 5 / 60, 0.0
        for k in range(480):
            u = 0.0 if k < 36 else 8 * 70 / 60
            y = solve_ivp(p.get_derivatives, (0, dt), y, args=(u, 0.0),
                          rtol=1e-6, atol=1e-9).y[:, -1]
            bis = p.step(dt, u, 0.0)
            worst = max(worst, abs(bis - p.get_bis(y[3], y[7])))
        self.assertLess(worst, 1e-8)  # measured 9.7e-10; README quotes 1e-9

    def test_test_and_tuning_patients_do_not_overlap(self):
        def key(p):
            return (p['age'], p['weight'], p['height'])
        test = {key(p) for p in test_patients(30)}
        tune = {key(p) for p in tuning_patients(20)}
        self.assertEqual(len(test), 30)
        self.assertFalse(test & tune)


class EnvironmentTest(unittest.TestCase):
    def test_reset_and_step_respect_spaces(self):
        env = AnesthesiaEnv(patient=test_patients(1)[0], noise_seed=0)
        obs, _ = env.reset()
        self.assertTrue(env.observation_space.contains(obs))
        for _ in range(20):
            obs, reward, term, trunc, info = env.step(env.action_space.sample())
            self.assertTrue(np.isfinite(reward))
            self.assertTrue(np.all(np.isfinite(obs)))
            self.assertIn('bis', info)
            if term or trunc:
                break


class PidBenchmarkTest(unittest.TestCase):
    def test_pid_reproduces_published_benchmark(self):
        """The PID column of results/benchmark.md, from the saved gains."""
        with open('results/benchmark.json') as f:
            published = json.load(f)['PID']['0']
        summary, per_patient = evaluate(load_pid())
        self.assertEqual(len(per_patient), 30)
        for k in ('time_in_target', 'MDAPE', 'bolus_mgkg', 'reached_below_20'):
            self.assertAlmostEqual(summary[k], published[k], places=6, msg=k)


if __name__ == '__main__':
    unittest.main()
