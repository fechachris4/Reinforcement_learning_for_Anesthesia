# Closed-loop propofol control: SAC vs a tuned PID

An anaesthetist keeps adjusting the propofol dose during surgery so the patient stays at the right depth. I wanted to know if reinforcement learning (Soft Actor-Critic) could do that job better than a well-tuned PID controller, on patients it had never seen.

This started as coursework in January 2026. That version trained and tested on the same five patients, had a perfect depth signal, and let the agent see the drug concentration inside the body. In September I rebuilt the evaluation so none of that is true any more.

![PID + SAC vs PID on an unseen patient](media/sac_vs_pid.gif)

*A 72-year-old test patient, 40 min in 30 s. The PID's standard bolus takes her to BIS 9; PID + SAC gives her 1 mg/kg and she bottoms out at 20. She is its biggest win of the 30; overall it ties. Dots are the noisy, 20 s delayed BIS the controllers see. [MP4](media/sac_vs_pid.mp4)*

Short answer: not overall. Across 30 test patients, SAC learning corrections on top of the PID ties with it (85% of the time in the 40 to 60 range for both, with fewer deep overdoses), and pure SAC is clearly worse (70%). The interesting part is what each one learned.

## Setup

Patients are sampled with their own age, size and variation in how they distribute, clear and respond to propofol, using the [Eleveld 2018](#references) model at half its published inter-patient variance (at full variance some patients can't be reached within the infusion limit). The controller sees what a clinician would: a BIS reading that arrives 20 s late with noise, the pump history, age and weight. Surgical stimulation pushes BIS up at random times.

Each case starts with a bolus of at least 1 mg/kg, then switches to a continuous infusion once BIS drops below 60 or 3 min pass. The PID gives an age-adjusted bolus (2.0 mg/kg, or 1.67 mg/kg from age 55), then runs PI control on the infusion, with gains tuned on a separate set of patients. PID + SAC starts from the PID and learns bounded corrections to the bolus size and the infusion.

The 30 test patients were never used for training or tuning, each controller gets the same noise and stimulation for a given patient, and everything is scored over the same 5 to 40 min window. RL results are mean ± sd over five training seeds.

## Results

| | PID | SAC | PID + SAC |
|---|---|---|---|
| Time in 40-60 (%) | 84.9 | 70.0 ± 9.0 | 85.3 ± 0.8 |
| Time below 40 (%) | 5.5 | 19.5 ± 7.6 | 5.0 ± 1.3 |
| Time above 60 (%) | 9.6 | 10.5 ± 2.6 | 9.7 ± 1.2 |
| MDAPE (%) | 7.4 | 14.5 ± 5.5 | 7.5 ± 0.2 |
| Bolus (mg/kg) | 1.9 | 2.5 ± 0.0 | 1.9 ± 0.2 |
| Patients reaching BIS < 20 (%) | 23 | 38 ± 3 | 18 ± 5 |

Compared patient by patient with the PID, PID + SAC is +0.4 points on time in range (95% CI −1.0 to +2.0, bootstrap over seeds and patients) and SAC is −14.9 (−26.0 to −6.3). MDAPE is the median absolute error from Varvel et al. (1992); bias and wobble are in [results/benchmark.md](results/benchmark.md).

<img src="media/paired_patients.png" width="55%">

*One point per test patient, RL averaged over seeds. The grey band is ±1 point. Outside it, PID + SAC beats the PID on 8 patients and loses on 12; SAC beats it on 2 and loses on 25.*

![BIS traces on six test patients](media/test_traces.png)

*True (noise-free) BIS for six test patients spanning the age range, using the seed of each RL controller that scored best on validation. Green: 40 to 60. Grey: surgical stimulation.*

## What each controller learned

**SAC found a loophole, then over-corrected.** In the first version induction could end on a 3 min timeout with no bolus, and SAC learned to give none: patients stayed awake for 3 min, then got the infusion at its limit. Penalising time above 60 didn't stop it, so I made a 1 mg/kg minimum bolus a rule of the environment. Now SAC gives the full 2.5 mg/kg budget every time, and 38% of patients go below BIS 20.

**PID + SAC learned to dose by age, and cut the deep overdoses.** For the 8 test patients aged 55 and over it gave a smaller bolus than the PID (1.47 vs 1.67 mg/kg), and their time in range went from 81% to 84%; younger patients got slightly more (2.10 vs 2.00). Only 18% of patients went below BIS 20, against 23% for the PID. With 8 older patients that is a hint, not a result, and time in range overall didn't move. Its infusion corrections added nothing measurable. My guess, not yet tested, is credit assignment: a dose change reaches the measured BIS 30 to 60 s later, and the PID partly cancels each correction.

**Induction is the weak point for everyone.** Even the PID takes 7 of 30 patients below BIS 20. The 20-year-old (bottom right) is the opposite case: she stays above 60 for 17 min under every controller. She is sampled as 40% less sensitive to propofol than average, and the infusion sits at its 20 mg/kg/h limit for most of that time.

## Next

Test the age effect on more older patients. A policy with memory (recurrent, or a stacked history of BIS and doses) to handle the delay. MPC on the patient model as a stronger baseline. A bolus sized from age and weight, since that is where PID + SAC found its gain.

## Code

| File | |
|---|---|
| `EleveldPatient.py` | 3-compartment PK/PD model, stepped exactly with a matrix exponential (matches an ODE solver to 3×10⁻⁷, about 100× faster) |
| `patients.py` | Patient sampling; training, tuning and test sets use separate seeds |
| `AnesthesiaEnv.py` | Gymnasium env: 5 s steps, 40 min case, delayed noisy BIS, stimulation, minimum bolus |
| `pid_baseline.py`, `tune_pid.py` | PID and its grid search |
| `train_sac.py`, `residual.py` | SAC and PID + SAC training |
| `benchmark.py`, `make_video.py` | Results table, figures, video |

```bash
pip install -r requirements.txt
python tune_pid.py
python train_sac.py --seed 0 --steps 500000              # seeds 0-4
python train_sac.py --seed 0 --steps 500000 --residual   # seeds 0-4
python benchmark.py
python make_video.py --patient 0 --model models/residual_seed4.zip
```

Five seeds of one controller take about 10 min on an Apple M5 laptop, all running at once,, because the patient model steps in closed form and 8 patients are simulated in parallel per run.

## Limitations

Simulation only, with a published population model at reduced variance. BIS only: no blood pressure, no opioid dosing. Stimulation is a simple offset on BIS. Not for clinical use.

## References

- Eleveld, D.J. et al. (2018). Pharmacokinetic–pharmacodynamic model for propofol for broad application in anaesthesia and sedation. *Br J Anaesth* 120(5), 942–959.
- Haarnoja, T. et al. (2018). Soft Actor-Critic. *ICML*.
- Varvel, J.R. et al. (1992). Measuring the predictive performance of computer-controlled infusion pumps. *J Pharmacokinet Biopharm* 20(1), 63–94.

*Coursework for BIOE70077 Reinforcement Learning for Bioengineers, Imperial College London.*
