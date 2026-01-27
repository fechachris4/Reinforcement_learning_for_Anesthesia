# Closed-Loop Anaesthesia Control with SAC

**Reinforcement learning for BIS-guided propofol delivery using the Eleveld PK/PD model**

This project implements a Soft Actor-Critic (SAC) agent that learns to control anaesthesia by maintaining the Bispectral Index (BIS) at a target of 50. The environment simulates realistic patient pharmacokinetics using the Eleveld 2018 propofol model.

**Course:** BIOE70077 Reinforcement Learning for Bioengineers, Imperial College London

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train SAC agent default is 100k steps
python train_sac.py

# To generate figures
python generate_training_figures.py
python generate_animations.py
```

## Environment Design

### Two-Phase Control

The environment models realistic anaesthesia induction with two distinct phases:

| Phase | Control Signal | Goal |
|-------|---------------|------|
| **Induction** | Bolus (mg/kg/min) | Rapidly lower BIS from 93 to 60 |
| **Maintenance** | Infusion (mg/min) | Hold BIS stable at 50 |

Phase transition occurs when BIS drops below 60 or the bolus budget (3 mg/kg) is depleted.

### State Space (6D)

| Index | Component | Description | Normalization |
|-------|-----------|-------------|---------------|
| 0 | BIS error | `(BIS - 50) / 100` | Centered at target |
| 1 | BIS trend | `(BIS - prev_BIS) / 100` | Rate of change |
| 2 | Last action | Previous infusion action | [0, 1] |
| 3 | Ce propofol | Effect-site concentration | `/ 5.0` |
| 4 | Bolus remaining | Fraction of budget left | [0, 1] |
| 5 | Phase flag | 0 = induction, 1 = maintenance | Binary |

### Action Space (2D, continuous)

| Index | Action | Range | Maps to |
|-------|--------|-------|---------|
| 0 | Infusion | [0, 1] | 0-12 mg/min (maintenance only) |
| 1 | Bolus | [0, 1] | 0-6 mg/kg/min (induction only) |

The environment enforces phase-appropriate actions: bolus is ignored during maintenance, infusion is ignored during induction.

### Reward Function

```
reward = r_bis - r_bolus - r_smooth - r_transition - r_safety
```

| Component | Formula | Purpose |
|-----------|---------|---------|
| `r_bis` | `4.0 * exp(-(BIS-50)² / 200)` | Gaussian tracking reward |
| `r_bolus` | `0.8 * bolus_used` | Penalize excessive bolus |
| `r_smooth` | `0.8 * (action - prev_action)²` | Penalize jerky control |
| `r_transition` | `1.0 * max(0, BIS - 60)` | Penalize failed induction |
| `r_safety` | `100.0` per violation | Terminal penalty |

### Safety Constraints

| Parameter | Safe Range | Action on Violation |
|-----------|------------|---------------------|
| BIS | 20-95 | Episode termination |

Safety checks activate after 120 steps (2 minutes) to allow induction.

## Patient Model

The environment uses the **Eleveld 2018 Propofol PK/PD model**:

### Patient Pool

Training and evaluation use the same 5 patients for fair comparison:

| Patient | Age | Weight | Height | Sex |
|---------|-----|--------|--------|-----|
| Reference | 40 | 70 kg | 170 cm | M |
| Young | 25 | 60 kg | 165 cm | F |
| Elderly | 70 | 80 kg | 175 cm | M |
| Large | 45 | 100 kg | 185 cm | M |
| Small | 35 | 55 kg | 160 cm | F |

## Training


# Default training (100k steps)
python train_sac.py

### SAC Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 3e-4 |
| Buffer size | 100,000 |
| Batch size | 256 |
| Discount (γ) | 0.99 |
| Soft update (τ) | 0.005 |
| Entropy | Auto-tuned (α) |

## Evaluation Metrics

Following Varvel et al. (1992) for anaesthesia performance:

| Metric | Formula | Clinical Target |
|--------|---------|-----------------|
| **Time in Target** | % of steps with BIS in [40, 60] | >80% |

## References

### Algorithm
- Haarnoja, T., et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor." ICML. [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)

### Patient Model
- Eleveld, D.J., et al. (2018). "Pharmacokinetic-pharmacodynamic model for propofol for broad application in anaesthesia and sedation." British Journal of Anaesthesia, 120(5):942-959.

### Metrics
- Varvel, J.R., et al. (1992). "Measuring the predictive performance of computer-controlled infusion pumps." J Pharmacokinet Biopharm, 20(1):63-94.

### Implementation
- Stable-Baselines3: [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)
