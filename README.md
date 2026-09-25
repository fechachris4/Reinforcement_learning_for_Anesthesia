# Closed-loop propofol control: SAC vs a tuned PID

[![tests](https://github.com/fechachris4/Reinforcement_learning_for_Anesthesia/actions/workflows/tests.yml/badge.svg)](https://github.com/fechachris4/Reinforcement_learning_for_Anesthesia/actions/workflows/tests.yml)

An anaesthetist keeps adjusting the propofol dose during surgery so the patient stays at the right depth. A PID controller, a simple feedback rule that raises or lowers the dose based on how far the patient is from the target, can automate this. I wanted to know if reinforcement learning (Soft Actor-Critic), where a program learns its own dosing strategy by trial and error, could do the job better, on patients it had never seen.

This started as coursework in January 2026. That version trained and tested on the same five patients, had a perfect depth signal, and let the agent see the drug concentration inside the body. In September I rebuilt the evaluation so none of those shortcuts remain.

![PID + SAC vs PID on an unseen patient](media/sac_vs_pid.gif)

*One test patient the controllers never saw (F, 72 y). Dots: the noisy, delayed depth signal the controllers actually see. This is the patient where the learned corrections help most; across all 30 test patients they make almost no difference. [MP4 version](media/sac_vs_pid.mp4)*

It didn't beat the PID. Across 30 test patients, letting SAC learn corrections on top of the PID ties with the PID alone: both keep patients in the target range about 85% of the time. SAC on its own does clearly worse. Everything here is in simulation.

## Setup

**The depth signal.** Depth of anaesthesia is tracked with BIS (bispectral index), a number from a brain monitor: about 93 when awake, and 40-60 is the target range for surgery. Lower means deeper.

**The patients.** Each simulated patient comes from the [Eleveld 2018](#references) propofol model, which describes how the drug spreads through the body and how strongly it acts, and how that varies between people. I sample patients at half the model's published spread between people, because at the full spread some patients can't be brought into range within the pump's maximum rate.

**What the controllers see.** Only what a clinician would: the BIS reading 20 s late and with noise, the pump history, and the patient's age and weight. At random times, simulated surgical stimulation pushes BIS up.

**How each case runs.** A case starts with a single initial dose (a bolus) of 1 to 2.5 mg/kg to put the patient under, then switches to a continuous infusion once BIS is below 60, the bolus budget is used or 3 min pass. The PID gives an age-adjusted bolus (2.0 mg/kg, or 1.67 from age 55), then adjusts the infusion with proportional and integral terms; adding a derivative term made it worse on the noisy, delayed signal. PID + SAC learns limited corrections to the PID's bolus and infusion. The RL reward is highest at BIS 50 and penalises time outside 40-60, drug use and sudden dose changes.

**Keeping the test fair.** The PID gains and the RL checkpoints (the best of each training run) are chosen on 20 tuning patients. The 30 test patients are used only for the results below, with the same noise and stimulation for every controller. RL is trained with five random seeds, so the results show how much it varies from run to run.

## Results

Adding learned corrections to the PID leaves it where it was: the two tie on every measure within the uncertainty. SAC on its own spends much less time in range, mostly because it goes too deep.

RL columns are mean ± sd over five seeds. The first four rows are scored from 5 to 40 min, the last three over the whole case.

| | PID | SAC | PID + SAC |
|---|---|---|---|
| Time in 40-60 (%) | 84.9 | 70.0 ± 10.1 | 85.3 ± 0.9 |
| Time below 40 (%) | 5.5 | 19.5 ± 8.5 | 5.0 ± 1.4 |
| Time above 60 (%) | 9.6 | 10.5 ± 2.9 | 9.7 ± 1.3 |
| MDAPE (%) | 7.4 | 14.5 ± 6.1 | 7.5 ± 0.2 |
| Maintenance propofol (mg/kg/h) | 7.1 | 6.6 ± 0.6 | 6.9 ± 0.2 |
| Induction bolus (mg/kg) | 1.9 | 2.5 ± 0.0 | 1.9 ± 0.2 |
| Patients reaching BIS < 20 (%) | 23 | 38 ± 3 | 18 ± 6 |

Against the PID, PID + SAC is +0.4 percentage points on time in range (95% confidence interval -1.0 to +2.0), so no real difference. SAC alone is -14.9 (-26.0 to -6.3), a clear loss. Both intervals come from bootstrapping over seeds and patients. The other PID + SAC differences are small and uncertain. The largest, fewer patients reaching BIS < 20 (-5 points, -15 to +3), is not statistically significant. All the intervals are in [results/benchmark.md](results/benchmark.md). MDAPE is the median absolute performance error, a standard measure of how far BIS strays from the target ([Varvel 1992](#references)).

<img src="media/paired_patients.png" width="80%" alt="Per-patient comparison of each RL controller against the PID on the 30 test patients">

*Each point is a test patient, with RL averaged over seeds; note the different y scales. (b) PID + SAC gains up to 12 points on patients the PID handles badly and loses up to 4 on others (dashed: ±1).*

![BIS traces on six test patients](media/test_traces.png)

*True (noise-free) BIS for six test patients across the age range, using each RL controller's best seed on the tuning set; top left is the GIF patient. Green: 40-60. Grey: surgical stimulation.*

## What each controller learned

**SAC alone found a loophole first.** Induction could end on the 3 min timer without any bolus, so it gave none and then ran the infusion at its maximum rate. A penalty for time above 60 didn't stop it, so I made a minimum bolus of 1 mg/kg part of the simulation. Now it gives the full 2.5 mg/kg every time, and 38% of patients go below BIS 20.

**PID + SAC adjusted the first dose by age.** It gave the 8 test patients aged 55 and over a smaller bolus than the PID (1.47 vs 1.67 mg/kg), and their time in range went from 81% to 84%. Younger patients got slightly more (2.10 vs 2.00). With only 8 patients I wouldn't lean on this. Its infusion corrections made no measurable difference. I suspect a credit-assignment problem, though I haven't tested it: a dose change only shows up in the measured BIS 30 to 60 s later, and the PID partly cancels each correction, so the agent struggles to tell which of its corrections helped.

**Induction is the weak point for all three.** Even the PID takes 7 of 30 patients below BIS 20. The 20-year-old (bottom right) is the opposite case: she is sampled as 40% less sensitive to propofol than average and stays above 60 for 17 min under the PID and PID + SAC. SAC's larger bolus gets her down sooner.

## Code and how to run it

| File | |
|---|---|
| `EleveldPatient.py` | 3-compartment patient model, stepped exactly with a matrix exponential (matches an ODE solver to 1e-9 BIS and is about 10x faster; run the file to check) |
| `patients.py` | Patient sampling; training, tuning and test sets use separate seeds |
| `AnesthesiaEnv.py` | Gymnasium environment: 5 s steps, 40 min case, delayed noisy BIS, stimulation, minimum bolus |
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
python -m unittest discover tests    # ~2 s: patient model vs ODE, env, PID reproduces its column above
```

Five seeds of one controller take about 10 min on an Apple M5 laptop, run in parallel (8 simulated patients per run, 2 gradient steps per 8 environment steps).

## Limitations and next steps

This is simulation only, with a published population model at reduced variation between patients. It models BIS only: no blood pressure and no opioid dosing. Surgical stimulation is a simple offset on BIS. Neither RL policy has memory, which is a handicap when the signal arrives 20 s late. Not for clinical use.

Next I would test the age effect on more older patients, give the policy memory (a recurrent network, or a stack of recent BIS readings and doses), and compare against MPC on the patient model.

## Contact

For questions about the project or the code, [open an issue](https://github.com/fechachris4/Reinforcement_learning_for_Anesthesia/issues) or email fecha412@gmail.com. The repository is maintained by Christian Akabueze.

## References

- Eleveld, D.J. et al. (2018). Pharmacokinetic–pharmacodynamic model for propofol for broad application in anaesthesia and sedation. *Br J Anaesth* 120(5), 942–959.
- Haarnoja, T. et al. (2018). Soft Actor-Critic. *ICML*.
- Varvel, J.R. et al. (1992). Measuring the predictive performance of computer-controlled infusion pumps. *J Pharmacokinet Biopharm* 20(1), 63–94.

*Coursework for BIOE70077 Reinforcement Learning for Bioengineers, Imperial College London.*
