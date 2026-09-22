"""
Virtual patients: sampled age, sex, height and weight, plus log-normal
variability on the main PK/PD parameters. Training, tuning and test patients
come from separate random streams, so the test set is never seen in training
or tuning.

Variances are half of Eleveld 2018's inter-individual variances. At full
variance some patients can't be brought into range within the infusion limit.
"""

import numpy as np

TEST_SEED = 20_260_922       # held-out test patients
TUNING_SEED = 7              # PID tuning and RL checkpoint selection

# log-normal variances (omega^2) for multiplicative factors
OMEGA2 = {
    'V1': 0.61, 'V2': 0.565, 'V3': 0.597,
    'Cl': 0.265, 'Q2': 0.346, 'Q3': 0.209,
    'ke0': 0.30,
    'Ce50': 0.242,
}
VARIANCE_SCALE = 0.5
E0_SD = 3.0      # additive, BIS units


def sample_patient(rng: np.random.Generator) -> dict:
    """Draw one adult surgical patient."""
    sex = 'm' if rng.random() < 0.5 else 'f'
    age = float(rng.uniform(18, 80))
    height = float(np.clip(rng.normal(178 if sex == 'm' else 165, 7), 150, 200))
    bmi = float(np.clip(rng.normal(26, 4), 18, 40))
    weight = float(np.clip(bmi * (height / 100) ** 2, 45, 150))

    variability = {}
    for name, w2 in OMEGA2.items():
        sd = np.sqrt(w2 * VARIANCE_SCALE)
        variability[name] = float(np.exp(np.clip(rng.normal(0, sd), -2 * sd, 2 * sd)))
    variability['E0'] = float(np.clip(rng.normal(0, E0_SD), -6, 4))

    return {'age': age, 'weight': weight, 'height': height, 'gender': sex,
            'variability': variability}


def patient_set(seed: int, n: int) -> list:
    rng = np.random.default_rng(seed)
    return [sample_patient(rng) for _ in range(n)]


def test_patients(n: int = 30) -> list:
    return patient_set(TEST_SEED, n)


def tuning_patients(n: int = 20) -> list:
    return patient_set(TUNING_SEED, n)


def describe(p: dict) -> str:
    return f"{p['gender'].upper()}, {p['age']:.0f} y, {p['weight']:.0f} kg, {p['height']:.0f} cm"
