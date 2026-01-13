"""
Eleveld Patient Model for Propofol and Remifentanil PK/PD Simulation
====================================================================

Implements the Eleveld 2018 Propofol and 2017 Remifentanil pharmacokinetic-
pharmacodynamic models for simulating anaesthesia depth control.

Key References:
    - Eleveld et al. (2018). Pharmacokinetic-pharmacodynamic model for propofol.
      British Journal of Anaesthesia, 120(5):942-959.
    - Eleveld et al. (2017). An Allometric Model of Remifentanil Pharmacokinetics.
      Anesthesiology, 126(6):1005-1018.
    - Al-Sallami et al. (2015). Prediction of Fat-Free Mass in Children.
      Clinical Pharmacokinetics, 54(11):1169-78.

Model Structure:
    - 3-compartment mammillary PK model (central + 2 peripheral)
    - Effect-site compartment linked via ke0
    - Sigmoid Emax PD model for BIS
    - Propofol-Remifentanil interaction model

Author: Christian Akabueze, Imperial College London,BIOE70077 Coursework
"""

import numpy as np
from scipy.integrate import solve_ivp
from math import exp, pow
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class PatientCovariates:
    """Patient demographic covariates for PK/PD calculations."""
    age: float  # years
    weight: float  # kg
    height: float  # cm
    gender: str  # 'm' or 'f'
    opioid_switch: bool = False  # True if opioids co-administered


class EleveldPatient:
    """
    Eleveld Patient Model for Propofol and Remifentanil PK/PD Simulation.
    
    This class implements a physiologically-based pharmacokinetic (PBPK) model
    that simulates drug distribution and effect for closed-loop anaesthesia control.
    
    The model captures:
        1. Three-compartment drug distribution (plasma, fast-equilibrating, slow-equilibrating)
        2. Effect-site concentration dynamics (plasma-effect site equilibration delay)
        3. Pharmacodynamic response (BIS as function of effect-site concentration)
        4. Drug-drug interaction (propofol-remifentanil synergy)
    
    State Vector:
        [A1_p, A2_p, A3_p, Ce_p, A1_r, A2_r, A3_r, Ce_r]
        Where:
            A1_p, A2_p, A3_p = Propofol amounts in compartments 1, 2, 3 (mg)
            Ce_p = Propofol effect-site concentration (ug/mL)
            A1_r, A2_r, A3_r = Remifentanil amounts in compartments 1, 2, 3 (ug)
            Ce_r = Remifentanil effect-site concentration (ng/mL)
    
    Example:
        >>> patient = EleveldPatient(age=45, weight=75, gender='m', height=175)
        >>> bis, map_val = patient.step(dt=1/60, u_prop=100, u_remi=5)
        >>> print(f"BIS: {bis:.1f}, MAP: {map_val:.1f}")
    """

    def __init__(
        self,
        age: float,
        weight: float,
        gender: str,
        height: float,
        opioid_switch: bool = True
    ):
        """
        Initialise patient with demographic covariates.
        
        Args:
            age: Patient age in years (valid range: 18-88)
            weight: Patient weight in kg (valid range: 40-160)
            gender: Patient gender ('m' for male, 'f' for female)
            height: Patient height in cm (valid range: 100-220)
            opioid_switch: Whether opioids are co-administered (affects propofol PK)
        
        Raises:
            ValueError: If covariates are outside validated ranges
        """
        # Store covariates
        self.age = age
        self.weight = weight
        self.gender = gender.lower()
        self.height = height
        self.opioid_switch = opioid_switch
        
        # Validate inputs
        self._validate_covariates()
        
        # Initialise state vector: all drug amounts and concentrations start at zero
        # This represents a drug-naive patient at the start of anaesthesia
        self.state = np.zeros(8, dtype=np.float64)
        
        # Calculate patient-specific PK/PD parameters
        self._calc_propofol_coefficients()
        self._calc_remifentanil_coefficients()
        self._calc_pd_parameters()
        
        # Store previous action for rate-of-change calculations
        self.prev_action = np.zeros(2)

    def _validate_covariates(self) -> None:
        """Validate patient covariates are within model-validated ranges."""
        if not 0 <= self.age <= 100:
            raise ValueError(f"Age {self.age} outside valid range [0, 100]")
        if not 20 <= self.weight <= 200:
            raise ValueError(f"Weight {self.weight} outside valid range [20, 200]")
        if not 100 <= self.height <= 250:
            raise ValueError(f"Height {self.height} outside valid range [100, 250]")
        if self.gender not in ('m', 'f'):
            raise ValueError(f"Gender must be 'm' or 'f', got {self.gender}")

    def _alsallami(
        self,
        age: float,
        height: float,
        weight: float,
        sex: str
    ) -> float:
        """
        Calculate Al-Sallami Fat-Free Mass (FFM).
        
        Fat-free mass is used in allometric scaling of PK parameters as it better
        correlates with drug distribution than total body weight, especially in
        obese patients.
        
        Reference:
            Al-Sallami et al. (2015). Prediction of Fat-Free Mass in Children.
            Clinical Pharmacokinetics, 54(11):1169-78.
        
        Args:
            age: Age in years
            height: Height in cm
            weight: Weight in kg
            sex: 'm' for male, 'f' for female
            
        Returns:
            Fat-free mass in kg
        """
        def bmi(h: float, w: float) -> float:
            """Calculate Body Mass Index (kg/m²)."""
            return w / ((h / 100) ** 2)

        def janmahasation(h: float, w: float, s: str) -> float:
            """
            Calculate Janmahasation Lean Body Mass.
            This is an intermediate calculation for the Al-Sallami FFM formula.
            """
            b = bmi(h, w)
            if s == "m":
                return (9270 * w) / (6680 + 216 * b)
            else:
                return (9270 * w) / (8780 + 244 * b)

        lbm = janmahasation(height, weight, sex)
        
        # Apply age-dependent maturation function
        if sex == "m":
            return (0.88 + (0.12 / (1 + (age / 13.4) ** -12.7))) * lbm
        else:
            return (1.11 + ((1 - 1.11) / (1 + (age / 7.1) ** -1.1))) * lbm

    def _calc_propofol_coefficients(self) -> None:
        """
        Calculate Eleveld 2018 Propofol PK Coefficients.
        
        The Eleveld model uses allometric scaling with fat-free mass and
        age-dependent maturation functions to account for inter-individual
        variability in drug distribution and clearance.
        
        Key model features:
            - Allometric scaling (3/4 power for clearances, linear for volumes)
            - Age-dependent maturation of clearance
            - Sex-specific clearance values
            - Optional opioid adjustment factors
        
        Reference:
            Eleveld et al. (2018). Pharmacokinetic-pharmacodynamic model for propofol.
            British Journal of Anaesthesia, 120(5):942-959.
        """
        # Population fixed effects (theta parameters from Eleveld 2018)
        theta = {
            1: 6.28,    # V1 reference (L)
            2: 25.5,    # V2 reference (L)
            3: 273,     # V3 reference (L)
            4: 1.79,    # CL male reference (L/min)
            5: 1.75,    # Q2 reference
            6: 1.11,    # Q3 reference
            7: 0.191,   # (not used here)
            8: 42.3,    # CL maturation Hill coefficient
            9: 9.06,    # CL maturation exponent
            10: -0.0156,  # V2 age effect
            11: -0.00286, # CL opioid age interaction
            12: 33.6,   # Central volume sigmoid E50
            13: -0.0138,  # V3 opioid age interaction
            14: 68.3,   # Q3 maturation PMA50
            15: 2.10,   # CL female reference (L/min)
            16: 1.30,   # Q2 maturation effect
            17: 1.42,   # ke0 scaling
            18: 0.68    # ke0 reference
        }

        age = self.age
        wgt = self.weight
        hgt = self.height
        sex = self.gender
        
        # Post-menstrual age in weeks (adult approximation)
        pma = age * 52 + 40
        pmaref = 35 * 52 + 40  # Reference: 35-year-old

        # Helper functions for covariate models
        def ageing(x: float, a: float) -> float:
            """Age-dependent exponential scaling."""
            return exp(x * (a - 35))
        
        def sigmoid(x: float, e50: float, y: float) -> float:
            """Sigmoid maturation function."""
            return (x**y) / (x**y + e50**y)
        
        def central(x: float) -> float:
            """Central volume sigmoid scaling."""
            return sigmoid(x, theta[12], 1)

        # Maturation functions
        clmat = sigmoid(pma, theta[8], theta[9])
        clmatref = sigmoid(pmaref, theta[8], theta[9])
        q3mat = sigmoid(pma, theta[14], 1)
        q3matref = sigmoid(pmaref, theta[14], 1)

        # Fat-free mass calculations
        ffm = self._alsallami(age, hgt, wgt, sex)
        ffmref = self._alsallami(35, 170, 70, "m")  # Reference patient

        # === VOLUMES (L) ===
        # V1: Central compartment (blood + highly perfused tissues)
        self.V1_p = theta[1] * (central(wgt) / central(70))
        
        # V2: Shallow peripheral (muscle, viscera)
        self.V2_p = theta[2] * (wgt / 70) * ageing(theta[10], age)
        
        # V3: Deep peripheral (fat, poorly perfused tissues)
        self.V3_p = theta[3] * (ffm / ffmref)

        # === CLEARANCES (L/min) ===
        # CL: Metabolic clearance (primarily hepatic)
        if sex == "m":
            self.Cl_p = theta[4] * (wgt / 70)**0.75 * (clmat / clmatref)
        else:
            self.Cl_p = theta[15] * (wgt / 70)**0.75 * (clmat / clmatref)

        # Q2: Inter-compartmental clearance (central <-> shallow peripheral)
        self.Q2_p = theta[5] * (self.V2_p / theta[2])**0.75 * (1 + theta[16] * (1 - q3mat / q3matref))
        
        # Q3: Inter-compartmental clearance (central <-> deep peripheral)
        self.Q3_p = theta[6] * (self.V3_p / theta[3])**0.75 * (q3mat / q3matref)

        # === OPIOID ADJUSTMENT ===
        # Concurrent opioid administration affects propofol disposition
        if self.opioid_switch:
            opiates_v3_factor = exp(theta[13] * age)
            opiates_cl_factor = exp(theta[11] * age)
            self.V3_p *= opiates_v3_factor
            self.Cl_p *= opiates_cl_factor

        # === EFFECT-SITE EQUILIBRATION ===
        # ke0: Rate constant for plasma-effect site equilibration
        # Lower ke0 = slower equilibration = more delay in drug effect
        self.ke0_p = 0.146 * (wgt / 70)**-0.25

        # === RATE CONSTANTS (1/min) ===
        # Derived from volumes and clearances for ODE system
        self.k10_p = self.Cl_p / self.V1_p      # Elimination from central
        self.k12_p = self.Q2_p / self.V1_p      # Central -> V2
        self.k21_p = self.Q2_p / self.V2_p      # V2 -> Central
        self.k13_p = self.Q3_p / self.V1_p      # Central -> V3
        self.k31_p = self.Q3_p / self.V3_p      # V3 -> Central

    def _calc_remifentanil_coefficients(self) -> None:
        """
        Calculate Eleveld 2017 Remifentanil PK Coefficients.
        
        Similar structure to propofol but with different parameter values
        reflecting remifentanil's unique pharmacokinetics (rapid onset,
        context-insensitive half-time due to ester hydrolysis).
        
        Reference:
            Eleveld et al. (2017). An Allometric Model of Remifentanil 
            Pharmacokinetics and Pharmacodynamics. Anesthesiology, 126(6):1005-1018.
        """
        # Reference volumes and clearances
        v1ref, v2ref, v3ref = 5.81, 8.82, 5.03
        clref, q2ref, q3ref = 2.58, 1.72, 0.124
        
        # Theta parameters for remifentanil
        theta = {
            1: 2.88,    # Maturation sigmoid E50
            2: -0.00554,  # V1 age effect
            3: -0.00327,  # CL age effect / V2 age effect
            4: -0.0315,   # V3 age effect
            5: 0.470,     # Sex effect on V2
            6: -0.0260    # V3 weight effect
        }

        age = self.age
        wgt = self.weight
        hgt = self.height
        sex = self.gender

        def ageing(i: float, a: float) -> float:
            return exp(i * (a - 35))
        
        def sigmoid(x: float, e50: float, y: float) -> float:
            return (x**y) / (x**y + e50**y)

        # Fat-free mass scaling
        ffm = self._alsallami(age, hgt, wgt, sex)
        ffmref = self._alsallami(35, 170, 70, "m")
        Fsize = ffm / ffmref

        # Maturation function
        mat = sigmoid(wgt, theta[1], 2)
        matref = sigmoid(70, theta[1], 2)
        Fmat = mat / matref

        # Sex effect (affects women of certain ages)
        if sex == "m":
            Fsex = 1
        else:
            Fsex = 1 + theta[5] * sigmoid(age, 12, 6) * (1 - sigmoid(age, 45, 6))

        # Volumes
        self.V1_r = v1ref * Fsize * ageing(theta[2], age)
        self.V2_r = v2ref * Fsize * ageing(theta[3], age) * Fsex
        self.V3_r = v3ref * Fsize * ageing(theta[4], age) * exp(theta[6] * (wgt - 70))

        # Clearances
        self.Cl_r = clref * Fsize**0.75 * Fmat * Fsex * ageing(theta[3], age)
        self.Q2_r = q2ref * (self.V2_r / v2ref)**0.75 * ageing(theta[2], age) * Fsex
        self.Q3_r = q3ref * (self.V3_r / v3ref)**0.75 * ageing(theta[2], age)

        # Effect-site equilibration (very fast for remifentanil)
        self.ke0_r = 1.09 * ageing(-0.0289, age)

        # Rate constants
        self.k10_r = self.Cl_r / self.V1_r
        self.k12_r = self.Q2_r / self.V1_r
        self.k21_r = self.Q2_r / self.V2_r
        self.k13_r = self.Q3_r / self.V1_r
        self.k31_r = self.Q3_r / self.V3_r

    def _calc_pd_parameters(self) -> None:
        """
        Calculate pharmacodynamic parameters for BIS response.
        
        The PD model uses a sigmoid Emax relationship between effect-site
        concentration and BIS. Drug interaction is modeled as a shift in
        the propofol Ce50 based on remifentanil concentration.
        
        BIS Model:
            BIS = E0 * (1 - Ce^γ / (Ce^γ + Ce50^γ))
        
        Where:
            E0 = Baseline BIS (awake, ~93)
            Ce = Effect-site concentration
            Ce50 = Concentration producing 50% effect (age-dependent)
            γ = Hill coefficient (steepness of response curve)
        """
        # Baseline BIS (fully awake)
        self.E0 = 93.0
        
        # Minimum BIS (deep anaesthesia/burst suppression)
        self.Emax = 0.0
        
        # Ce50 for propofol: concentration producing 50% BIS reduction
        # Age-dependent: younger patients typically need higher concentrations
        # Formula from Eleveld 2018 supplementary material
        self.Ce50_prop_base = 3.08 * pow(10, -0.00635 * (self.age - 35))
        
        # Hill coefficient (steepness of dose-response curve)
        # Higher gamma = steeper transition between awake and anaesthetised
        self.gamma_prop = 2.0
        
        # Remifentanil interaction parameter
        # Ce50 for remifentanil's contribution to propofol potentiation
        # Based on Bouillon et al. (2004) response surface data
        self.C50_remi_interaction = 10.0  # ng/mL

    def get_derivatives(
        self,
        t: float,
        y: np.ndarray,
        u_prop: float,
        u_remi: float
    ) -> List[float]:
        """
        Compute state derivatives for ODE integration.
        
        This implements the mammillary 3-compartment model ODEs for both
        propofol and remifentanil, plus effect-site dynamics.
        
        Compartment ODEs (mass balance):
            dA1/dt = Input - k10*A1 - k12*A1 - k13*A1 + k21*A2 + k31*A3
            dA2/dt = k12*A1 - k21*A2
            dA3/dt = k13*A1 - k31*A3
        
        Effect-site ODE:
            dCe/dt = ke0 * (Cp - Ce)
            where Cp = A1/V1 (plasma concentration)
        
        Args:
            t: Current time (not used, required by solve_ivp)
            y: State vector [A1_p, A2_p, A3_p, Ce_p, A1_r, A2_r, A3_r, Ce_r]
            u_prop: Propofol infusion rate (mg/min)
            u_remi: Remifentanil infusion rate (ug/min)
            
        Returns:
            List of derivatives for each state variable
        """
        A1_p, A2_p, A3_p, Ce_p, A1_r, A2_r, A3_r, Ce_r = y

        # ===== PROPOFOL DYNAMICS =====
        # Central compartment: input + redistribution - elimination
        dA1_p = (u_prop 
                 - (self.k10_p + self.k12_p + self.k13_p) * A1_p 
                 + self.k21_p * A2_p 
                 + self.k31_p * A3_p)
        
        # Peripheral compartments
        dA2_p = self.k12_p * A1_p - self.k21_p * A2_p
        dA3_p = self.k13_p * A1_p - self.k31_p * A3_p
        
        # Effect-site: equilibrates with plasma concentration
        Cp_p = A1_p / self.V1_p  # Plasma concentration (ug/mL)
        dCe_p = self.ke0_p * (Cp_p - Ce_p)

        # ===== REMIFENTANIL DYNAMICS =====
        dA1_r = (u_remi 
                 - (self.k10_r + self.k12_r + self.k13_r) * A1_r 
                 + self.k21_r * A2_r 
                 + self.k31_r * A3_r)
        dA2_r = self.k12_r * A1_r - self.k21_r * A2_r
        dA3_r = self.k13_r * A1_r - self.k31_r * A3_r
        
        Cp_r = A1_r / self.V1_r  # Plasma concentration (ng/mL)
        dCe_r = self.ke0_r * (Cp_r - Ce_r)

        return [dA1_p, dA2_p, dA3_p, dCe_p, dA1_r, dA2_r, dA3_r, dCe_r]

    def step(
        self,
        dt: float,
        u_prop: float,
        u_remi: float
    ) -> Tuple[float, float]:
        """
        Advance simulation by dt minutes with given infusion rates.
        
        Args:
            dt: Time step in minutes
            u_prop: Propofol infusion rate (mg/min)
            u_remi: Remifentanil infusion rate (ug/min)
            
        Returns:
            Tuple of (BIS, MAP) values after the time step
        """
        # Ensure non-negative infusion rates
        u_prop = max(0, u_prop)
        u_remi = max(0, u_remi)
        
        # Integrate ODEs using LSODA (adaptive step size, handles stiff systems)
        sol = solve_ivp(
            self.get_derivatives,
            [0, dt],
            self.state,
            args=(u_prop, u_remi),
            method='LSODA',
            rtol=1e-6,
            atol=1e-9
        )
        
        # Update state to end of integration
        self.state = sol.y[:, -1]
        
        # Ensure non-negative concentrations (numerical stability)
        self.state = np.maximum(self.state, 0)
        
        # Extract effect-site concentrations
        Ce_prop = self.state[3]  # ug/mL
        Ce_remi = self.state[7]  # ng/mL

        # Calculate physiological outputs
        bis = self.get_bis(Ce_prop, Ce_remi)
        map_val = self.get_map(Ce_prop, Ce_remi)

        return bis, map_val

    def get_bis(self, Ce_prop: float, Ce_remi: float) -> float:
        """
        Calculate BIS using Propofol effect-site concentration with 
        Remifentanil interaction.
        
        The interaction model implements a Ce50 shift: remifentanil reduces
        the propofol concentration required to achieve a given BIS level.
        This captures the clinically observed synergy between hypnotics
        and opioids.
        
        Args:
            Ce_prop: Propofol effect-site concentration (ug/mL)
            Ce_remi: Remifentanil effect-site concentration (ng/mL)
            
        Returns:
            BIS value (0-100 scale)
        """
        # Interaction model: Remi reduces effective Ce50 for propofol
        # Maximum reduction capped at 80% to prevent numerical instability
        max_reduction = 0.8
        interaction_factor = (Ce_remi / (Ce_remi + self.C50_remi_interaction)) * max_reduction
        
        Ce50_shifted = self.Ce50_prop_base * (1 - interaction_factor)
        
        # Prevent division by zero
        if Ce50_shifted < 1e-6:
            Ce50_shifted = 1e-6
        
        # Sigmoid Emax model
        # BIS decreases from E0 (awake) as concentration increases
        prop_ratio = (Ce_prop / Ce50_shifted) ** self.gamma_prop
        effect_normalized = prop_ratio / (1 + prop_ratio)
        
        bis = self.E0 * (1 - effect_normalized)
        
        # Clamp to valid BIS range
        return np.clip(bis, 0, 100)

    def get_map(self, Ce_prop: float, Ce_remi: float) -> float:
        """
        Calculate Mean Arterial Pressure (simplified model).
        
        Both propofol and remifentanil cause dose-dependent hypotension.
        This simplified model captures the qualitative behavior without
        full cardiovascular modeling.
        
        Args:
            Ce_prop: Propofol effect-site concentration (ug/mL)
            Ce_remi: Remifentanil effect-site concentration (ng/mL)
            
        Returns:
            MAP value in mmHg
        """
        baseline = 90.0  # Baseline MAP (mmHg)
        max_drop = 40.0  # Maximum MAP reduction
        
        # Combined hypotensive potency (empirical weighting)
        potency = (Ce_prop / 4.0) + (Ce_remi / 10.0)
        
        # Sigmoid drop in MAP
        drop = max_drop * (potency / (1 + potency))
        
        return np.clip(baseline - drop, 40, 120)

    def get_plasma_concentrations(self) -> Tuple[float, float]:
        """Return current plasma concentrations (Cp_prop, Cp_remi)."""
        Cp_prop = self.state[0] / self.V1_p
        Cp_remi = self.state[4] / self.V1_r
        return Cp_prop, Cp_remi

    def get_effect_site_concentrations(self) -> Tuple[float, float]:
        """Return current effect-site concentrations (Ce_prop, Ce_remi)."""
        return self.state[3], self.state[7]

    def add_bolus(self, bolus_mg: float) -> None:
        """
        Add bolus directly to central compartment A1.
        
        This simulates rapid IV bolus injection during induction.
        The drug is added instantaneously to the central compartment,
        then redistributes according to normal PK dynamics.
        
        Args:
            bolus_mg: Bolus amount in mg (absolute, not per kg)
        """
        self.state[0] += bolus_mg  # A1_p: propofol amount in central compartment (mg)

    def reset(self) -> None:
        """Reset patient to drug-naive state."""
        self.state = np.zeros(8, dtype=np.float64)
        self.prev_action = np.zeros(2)

    def get_patient_info(self) -> dict:
        """Return patient demographic and PK parameter summary."""
        return {
            'age': self.age,
            'weight': self.weight,
            'height': self.height,
            'gender': self.gender,
            'V1_prop': self.V1_p,
            'Cl_prop': self.Cl_p,
            'ke0_prop': self.ke0_p,
            'Ce50_prop': self.Ce50_prop_base,
            'V1_remi': self.V1_r,
            'Cl_remi': self.Cl_r,
            'ke0_remi': self.ke0_r
        }


# Convenience function for creating test patients
def create_test_patient(patient_type: str = 'reference') -> EleveldPatient:
    """
    Create predefined test patients for reproducible experiments.
    
    Args:
        patient_type: One of 'reference', 'elderly', 'obese', 'young_female'
        
    Returns:
        EleveldPatient instance
    """
    patients = {
        'reference': {'age': 40, 'weight': 70, 'height': 170, 'gender': 'm'},
        'elderly': {'age': 75, 'weight': 65, 'height': 165, 'gender': 'm'},
        'obese': {'age': 45, 'weight': 110, 'height': 175, 'gender': 'm'},
        'young_female': {'age': 25, 'weight': 55, 'height': 160, 'gender': 'f'},
        'pediatric': {'age': 12, 'weight': 40, 'height': 150, 'gender': 'm'},
    }
    
    if patient_type not in patients:
        raise ValueError(f"Unknown patient type: {patient_type}. "
                        f"Available: {list(patients.keys())}")
    
    return EleveldPatient(**patients[patient_type], opioid_switch=True)
