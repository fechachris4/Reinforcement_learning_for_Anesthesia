# Closed-loop propofol control: SAC vs a tuned PID

An anaesthetist keeps adjusting the propofol dose during surgery so the patient stays at the right depth. I wanted to know if a reinforcement learning agent (Soft Actor-Critic) could do that job better than a well-tuned PID controller, on patients it had never seen.

This started as coursework in January 2026. The first version trained and tested on the same five patients, had a perfect depth signal, and let the agent see the drug concentration inside the body. In September I rebuilt the evaluation so none of that is true any more.

![SAC vs PID on an unseen patient](media/sac_vs_pid.gif)

*A 72-year-old test patient, 40 min of surgery in 30 s. The PID's induction bolus drops her to BIS 9, and she stays below 15 for 7 min. SAC avoids that by leaving her awake for the first 3 min. [MP4](media/sac_vs_pid.mp4)*

It didn't. Over 30 unseen patients the PID kept BIS in the 40 to 60 range 81% of the time, pure SAC 71%, and SAC learning corrections on top of the PID 79%.

## Setup

Patients are sampled with their own age, size and variation in how they distribute, clear and respond to propofol ([Eleveld 2018](#references) model). The controller sees only what a clinician would: a BIS reading that arrives 20 s late with noise, the pump history, age and weight. Surgical stimulation pushes BIS up at random times.

The PID gives an age-adjusted bolus (2.0 mg/kg, or 1.5 mg/kg from age 55), then runs PI control on the infusion. Its gains were tuned on a separate set of patients. The 30 test patients were never used for training or tuning, and for each one every controller gets the same noise and stimulation. SAC results are the mean ± sd of three training seeds.

## Results

| | PID | SAC | PID + SAC |
|---|---|---|---|
| Maintenance: time in 40-60 (%) | 81.2 | 71.3 ± 1.8 | 78.5 ± 0.8 |
| Maintenance: MDAPE (%) | 8.0 | 11.8 ± 1.1 | 8.4 ± 0.4 |
| Maintenance: MDPE (%) | 1.9 | 0.5 ± 0.3 | 2.1 ± 0.1 |
| Maintenance: wobble (%) | 6.1 | 7.4 ± 1.0 | 6.7 ± 0.2 |
| Maintenance: propofol (mg/kg/h) | 7.1 | 6.8 ± 0.7 | 6.7 ± 0.1 |
| Whole case: time below 40 (%) | 8.8 | 13.9 ± 2.2 | 11.8 ± 0.9 |
| Whole case: time above 60, after the first minute (%) | 11.3 | 15.7 ± 2.9 | 9.9 ± 0.4 |
| Patients that reach BIS < 20 (%) | 23 | 24 ± 10 | 29 ± 4 |

MDPE, MDAPE and wobble are the bias, inaccuracy and variability measures from Varvel et al. (1992).

<img src="media/paired_patients.png" width="45%">

*Time in range per test patient. Points above the diagonal are patients where the RL controller beat the PID: 3 of 30 for SAC, 9 of 30 for PID + SAC.*

![BIS traces on six test patients](media/test_traces.png)

*Six test patients, PID vs PID + SAC. Shaded band: 40 to 60.*

## What went wrong

**SAC found a loophole.** For older patients it skips the bolus, waits about 3 min, then runs the infusion at its maximum. In the video patient that softens the overshoot (lowest BIS 28 vs 9), but it leaves her awake as surgery starts, and across the test set SAC still takes about a quarter of patients below BIS 20, like the PID. Penalising time above 60 did not remove the loophole.

**Residual SAC made the PID worse.** It starts as the PID and only has to learn small corrections, but its corrections lowered its own training reward compared with applying none. My explanation is credit assignment: a dose change reaches the measured BIS 30 to 60 s later, and the PID underneath partly cancels each correction.

**Induction is the weak point for everyone.** Even with the age adjustment, the PID's bolus takes 7 of 30 test patients below BIS 20. The opposite also happens: the 20-year-old test patient (bottom right above) sits above 60 for 20 min with the infusion at its 20 mg/kg/h cap for most of that time.

## Next

A policy with memory (recurrent, or a stacked history of BIS and doses) to handle the delay. Pre-training on the PID's behaviour before RL fine-tuning. MPC on the patient model as a stronger baseline. A patient-specific induction dose.

## Code

| File | |
|---|---|
| `EleveldPatient.py` | 3-compartment PK/PD model, stepped exactly with a matrix exponential (matches an ODE solver to 3×10⁻⁷, about 100× faster) |
| `patients.py` | Patient sampling; training, tuning and test sets use separate seeds |
| `AnesthesiaEnv.py` | Gymnasium env: 5 s steps, 40 min case, delayed noisy BIS, stimulation |
| `pid_baseline.py`, `tune_pid.py` | PID and its grid search |
| `train_sac.py`, `residual.py` | SAC and PID + SAC training |
| `benchmark.py`, `make_video.py` | Results table, figures, video |

```bash
pip install -r requirements.txt
python tune_pid.py
python train_sac.py --seed 0 --steps 500000              # seeds 0-2
python train_sac.py --seed 0 --steps 300000 --residual   # seeds 0-2
python benchmark.py
python make_video.py --patient 0 --model models/sac_seed2.zip
```

All six training runs take about 15 min on an M-series laptop.

## Limitations

Simulation only, with a published population model and half the published inter-patient variance. BIS only: no blood pressure, no opioid dosing. Stimulation is a simple offset on BIS. Not for clinical use.

## References

- Eleveld, D.J. et al. (2018). Pharmacokinetic–pharmacodynamic model for propofol for broad application in anaesthesia and sedation. *Br J Anaesth* 120(5), 942–959.
- Haarnoja, T. et al. (2018). Soft Actor-Critic. *ICML*.
- Varvel, J.R. et al. (1992). Measuring the predictive performance of computer-controlled infusion pumps. *J Pharmacokinet Biopharm* 20(1), 63–94.

*Coursework for BIOE70077 Reinforcement Learning for Bioengineers, Imperial College London.*
