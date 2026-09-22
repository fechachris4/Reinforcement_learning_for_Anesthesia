"""
Eleveld 2018 propofol PK/PD model, plus the Eleveld 2017 remifentanil PK kept
from the coursework version (the environment always gives 0 remifentanil).

Christian Akabueze, BIOE70077 coursework, Imperial College London
"""

import numpy as np
from scipy.linalg import expm
from math import exp, pow


class EleveldPatient:
    """
    State: [A1, A2, A3, Ce] for propofol (mg, Ce in ug/mL), then the same
    for remifentanil (ug, Ce in ng/mL). Time is in minutes.
    """

    def __init__(self, age, weight, gender, height, opioid_switch=True, variability=None):
        """
        age in years, weight in kg, height in cm, gender 'm' or 'f'.
        variability: multiplicative factors per patient, e.g. {'V1': 1.2, 'Cl': 0.8,
        'ke0': 1.1, 'Ce50': 0.9}, plus an additive 'E0'. None = population-typical.
        """
        self.age = age
        self.weight = weight
        self.gender = gender.lower()
        self.height = height
        self.opioid_switch = opioid_switch
        self._validate_covariates()

        self.state = np.zeros(8)
        self._calc_propofol_coefficients()
        self._calc_remifentanil_coefficients()
        self._calc_pd_parameters()
        self.variability = dict(variability or {})
        self._apply_variability()
        self._discrete_cache = {}

    def _validate_covariates(self):
        if not 0 <= self.age <= 100:
            raise ValueError(f"Age {self.age} outside valid range [0, 100]")
        if not 20 <= self.weight <= 200:
            raise ValueError(f"Weight {self.weight} outside valid range [20, 200]")
        if not 100 <= self.height <= 250:
            raise ValueError(f"Height {self.height} outside valid range [100, 250]")
        if self.gender not in ('m', 'f'):
            raise ValueError(f"Gender must be 'm' or 'f', got {self.gender}")

    def _alsallami(self, age, height, weight, sex):
        """Fat-free mass (kg), Al-Sallami et al. 2015."""
        bmi = weight / ((height / 100) ** 2)
        if sex == "m":
            lbm = (9270 * weight) / (6680 + 216 * bmi)   # Janmahasatian lean body mass
            return (0.88 + (0.12 / (1 + (age / 13.4) ** -12.7))) * lbm
        lbm = (9270 * weight) / (8780 + 244 * bmi)
        return (1.11 + ((1 - 1.11) / (1 + (age / 7.1) ** -1.1))) * lbm

    def _calc_propofol_coefficients(self):
        # theta from Eleveld 2018, Table 2 (arterial sampling)
        theta = {
            1: 6.28,      # V1 ref (L)
            2: 25.5,      # V2 ref (L)
            3: 273,       # V3 ref (L)
            4: 1.79,      # CL male (L/min)
            5: 1.75,      # Q2 (L/min)
            6: 1.11,      # Q3 (L/min)
            8: 42.3,      # CL maturation E50 (weeks PMA)
            9: 9.06,      # CL maturation slope
            10: -0.0156,  # V2 ageing
            11: -0.00286, # CL with opioids
            12: 33.6,     # V1 weight E50 (kg)
            13: -0.0138,  # V3 with opioids
            14: 68.3,     # Q3 maturation E50 (weeks PMA)
            15: 2.10,     # CL female (L/min)
            16: 1.30,     # Q2 immature
        }
        age, wgt, hgt, sex = self.age, self.weight, self.height, self.gender

        pma = age * 52 + 40          # post-menstrual age, weeks
        pmaref = 35 * 52 + 40        # reference patient: 35 y, 70 kg, 170 cm, male

        def ageing(x, a):
            return exp(x * (a - 35))

        def sigmoid(x, e50, y):
            return (x**y) / (x**y + e50**y)

        clmat = sigmoid(pma, theta[8], theta[9])
        clmatref = sigmoid(pmaref, theta[8], theta[9])
        q3mat = sigmoid(pma, theta[14], 1)
        q3matref = sigmoid(pmaref, theta[14], 1)
        ffm = self._alsallami(age, hgt, wgt, sex)
        ffmref = self._alsallami(35, 170, 70, "m")

        # volumes (L)
        self.V1_p = theta[1] * (sigmoid(wgt, theta[12], 1) / sigmoid(70, theta[12], 1))
        self.V2_p = theta[2] * (wgt / 70) * ageing(theta[10], age)
        self.V3_p = theta[3] * (ffm / ffmref)

        # clearances (L/min)
        cl_ref = theta[4] if sex == "m" else theta[15]
        self.Cl_p = cl_ref * (wgt / 70)**0.75 * (clmat / clmatref)
        self.Q2_p = theta[5] * (self.V2_p / theta[2])**0.75 * (1 + theta[16] * (1 - q3mat / q3matref))
        self.Q3_p = theta[6] * (self.V3_p / theta[3])**0.75 * (q3mat / q3matref)

        if self.opioid_switch:
            self.V3_p *= exp(theta[13] * age)
            self.Cl_p *= exp(theta[11] * age)

        # plasma to effect-site equilibration (1/min); smaller ke0 = longer lag
        self.ke0_p = 0.146 * (wgt / 70)**-0.25
        self._rate_constants()

    def _rate_constants(self):
        self.k10_p = self.Cl_p / self.V1_p
        self.k12_p = self.Q2_p / self.V1_p
        self.k21_p = self.Q2_p / self.V2_p
        self.k13_p = self.Q3_p / self.V1_p
        self.k31_p = self.Q3_p / self.V3_p

    def _apply_variability(self):
        """Scale PK/PD parameters by the patient's factors, then rebuild the rate constants."""
        v = self.variability
        self.V1_p *= v.get('V1', 1.0)
        self.V2_p *= v.get('V2', 1.0)
        self.V3_p *= v.get('V3', 1.0)
        self.Cl_p *= v.get('Cl', 1.0)
        self.Q2_p *= v.get('Q2', 1.0)
        self.Q3_p *= v.get('Q3', 1.0)
        self.ke0_p *= v.get('ke0', 1.0)
        self.Ce50_prop_base *= v.get('Ce50', 1.0)
        self.E0 += v.get('E0', 0.0)
        self._rate_constants()

    def _system_matrices(self):
        """dx/dt = A x + B u for the 8-state model, u = [propofol mg/min, remifentanil ug/min]."""
        A = np.zeros((8, 8))
        p = [self.k10_p, self.k12_p, self.k21_p, self.k13_p, self.k31_p, self.ke0_p, self.V1_p]
        r = [self.k10_r, self.k12_r, self.k21_r, self.k13_r, self.k31_r, self.ke0_r, self.V1_r]
        for o, (k10, k12, k21, k13, k31, ke0, V1) in zip((0, 4), (p, r)):
            A[o, o] = -(k10 + k12 + k13)
            A[o, o + 1] = k21
            A[o, o + 2] = k31
            A[o + 1, o] = k12
            A[o + 1, o + 1] = -k21
            A[o + 2, o] = k13
            A[o + 2, o + 2] = -k31
            A[o + 3, o] = ke0 / V1
            A[o + 3, o + 3] = -ke0
        B = np.zeros((8, 2))
        B[0, 0] = 1.0
        B[4, 1] = 1.0
        return A, B

    def _discrete(self, dt):
        """Exact zero-order-hold discretisation, cached per step size."""
        if dt not in self._discrete_cache:
            A, B = self._system_matrices()
            M = np.zeros((10, 10))
            M[:8, :8] = A
            M[:8, 8:] = B
            E = expm(M * dt)
            self._discrete_cache[dt] = (E[:8, :8], E[:8, 8:])
        return self._discrete_cache[dt]

    def _calc_remifentanil_coefficients(self):
        # Eleveld 2017. Only here because the coursework version dosed remifentanil too.
        v1ref, v2ref, v3ref = 5.81, 8.82, 5.03
        clref, q2ref, q3ref = 2.58, 1.72, 0.124
        age, wgt, sex = self.age, self.weight, self.gender

        def ageing(i, a):
            return exp(i * (a - 35))

        def sigmoid(x, e50, y):
            return (x**y) / (x**y + e50**y)

        fsize = self._alsallami(age, self.height, wgt, sex) / self._alsallami(35, 170, 70, "m")
        fmat = sigmoid(wgt, 2.88, 2) / sigmoid(70, 2.88, 2)
        fsex = 1 if sex == "m" else 1 + 0.470 * sigmoid(age, 12, 6) * (1 - sigmoid(age, 45, 6))

        self.V1_r = v1ref * fsize * ageing(-0.00554, age)
        self.V2_r = v2ref * fsize * ageing(-0.00327, age) * fsex
        self.V3_r = v3ref * fsize * ageing(-0.0315, age) * exp(-0.0260 * (wgt - 70))
        self.Cl_r = clref * fsize**0.75 * fmat * fsex * ageing(-0.00327, age)
        self.Q2_r = q2ref * (self.V2_r / v2ref)**0.75 * ageing(-0.00554, age) * fsex
        self.Q3_r = q3ref * (self.V3_r / v3ref)**0.75 * ageing(-0.00554, age)
        self.ke0_r = 1.09 * ageing(-0.0289, age)

        self.k10_r = self.Cl_r / self.V1_r
        self.k12_r = self.Q2_r / self.V1_r
        self.k21_r = self.Q2_r / self.V2_r
        self.k13_r = self.Q3_r / self.V1_r
        self.k31_r = self.Q3_r / self.V3_r

    def _calc_pd_parameters(self):
        """BIS = E0 * (1 - Ce^g / (Ce^g + Ce50^g))."""
        self.E0 = 93.0
        self.Ce50_prop_base = 3.08 * pow(10, -0.00635 * (self.age - 35))   # ug/mL
        self.gamma_prop = 2.0
        # rough value, not fitted; has no effect while remifentanil is 0
        self.C50_remi_interaction = 10.0  # ng/mL

    def get_derivatives(self, t, y, u_prop, u_remi):
        """Right-hand side for solve_ivp. Only used to check step() against an ODE solver."""
        A1_p, A2_p, A3_p, Ce_p, A1_r, A2_r, A3_r, Ce_r = y
        dA1_p = (u_prop - (self.k10_p + self.k12_p + self.k13_p) * A1_p
                 + self.k21_p * A2_p + self.k31_p * A3_p)
        dA2_p = self.k12_p * A1_p - self.k21_p * A2_p
        dA3_p = self.k13_p * A1_p - self.k31_p * A3_p
        dCe_p = self.ke0_p * (A1_p / self.V1_p - Ce_p)

        dA1_r = (u_remi - (self.k10_r + self.k12_r + self.k13_r) * A1_r
                 + self.k21_r * A2_r + self.k31_r * A3_r)
        dA2_r = self.k12_r * A1_r - self.k21_r * A2_r
        dA3_r = self.k13_r * A1_r - self.k31_r * A3_r
        dCe_r = self.ke0_r * (A1_r / self.V1_r - Ce_r)
        return [dA1_p, dA2_p, dA3_p, dCe_p, dA1_r, dA2_r, dA3_r, dCe_r]

    def step(self, dt, u_prop, u_remi):
        """Advance dt minutes at constant infusion (propofol mg/min, remifentanil ug/min). Returns BIS."""
        u = np.array([max(0, u_prop), max(0, u_remi)])
        # the PK model is linear, so the zero-order-hold update is exact
        # and about 100x faster than an ODE solver
        Ad, Bd = self._discrete(dt)
        self.state = np.maximum(Ad @ self.state + Bd @ u, 0)
        return self.get_bis(self.state[3], self.state[7])

    def get_bis(self, Ce_prop, Ce_remi):
        # remifentanil lowers the propofol Ce50, by at most 80%
        shift = 0.8 * Ce_remi / (Ce_remi + self.C50_remi_interaction)
        ce50 = max(self.Ce50_prop_base * (1 - shift), 1e-6)
        r = (Ce_prop / ce50) ** self.gamma_prop
        return np.clip(self.E0 * (1 - r / (1 + r)), 0, 100)

    def get_effect_site_concentrations(self):
        return self.state[3], self.state[7]

    def add_bolus(self, bolus_mg):
        """Bolus goes straight into the central compartment."""
        self.state[0] += bolus_mg


if __name__ == '__main__':
    # check the matrix-exponential step against an ODE solver: 2 mg/kg bolus, then 8 mg/kg/h
    import time
    from scipy.integrate import solve_ivp
    p = EleveldPatient(60, 70, 'm', 170)
    p.add_bolus(140)
    y, dt, worst, t_ode, t_exp = p.state.copy(), 5 / 60, 0.0, 0.0, 0.0
    for k in range(480):
        u = 0.0 if k < 36 else 8 * 70 / 60
        t0 = time.perf_counter()
        y = solve_ivp(p.get_derivatives, (0, dt), y, args=(u, 0.0), rtol=1e-6, atol=1e-9).y[:, -1]
        t1 = time.perf_counter()
        bis = p.step(dt, u, 0.0)
        t_ode, t_exp = t_ode + t1 - t0, t_exp + time.perf_counter() - t1
        worst = max(worst, abs(bis - p.get_bis(y[3], y[7])))
    print(f'max BIS difference over 40 min: {worst:.1e}, step is {t_ode / t_exp:.0f}x faster than solve_ivp')
