# Teaching an AI to dose anaesthesia

During surgery, an anaesthetist adjusts the propofol dose to keep the patient at the right depth of unconsciousness. Too little and they may wake up; too much and blood pressure falls and recovery is slow. This project trains reinforcement learning agents (Soft Actor-Critic) to do that job and tests them against a clinical-style PID controller on **30 simulated patients they have never seen.**

![SAC vs PID on an unseen patient](media/sac_vs_pid.gif)

*40 minutes of surgery in 30 seconds, on a 72-year-old patient neither controller has seen. Both face the same monitor noise and the same surgical stimulation. It shows the two failure modes this project uncovered: the PID's standard induction dose sends this older patient far too deep (BIS below 10), while the RL agent avoids that by leaving her awake for three minutes. [Full-resolution video](media/sac_vs_pid.mp4).*

**Result:** the tuned PID kept BIS in the target range **81%** of the time. Pure SAC reached **71%** and learned an unsafe shortcut. SAC learning corrections on top of the PID reached **79%**. Neither reinforcement learning variant beat the classical baseline, and the reasons why are the most useful part of the project.

## Why this is hard

- **Every patient is different.** The same dose can put one person at the right depth and another far too deep. Patients are sampled with individual variation in how they distribute, clear and respond to the drug.
- **The monitor is noisy and late.** The depth signal (BIS, 0 = no brain activity, ~93 = awake) arrives 20 seconds late with random noise, as real monitors do.
- **Surgery wakes the patient up.** Incision and other stimulation push BIS up at unpredictable times, and the controller has to respond.
- **The controller only sees what a clinician sees:** the BIS reading, the pump history, age and weight. It never sees drug levels inside the body.

## Results on 30 unseen patients

Three controllers, each tested on the same 30 patients with identical noise and stimulation:

- **PID:** tuned clinical-style baseline.
- **SAC:** pure reinforcement learning, choosing the bolus and infusion directly.
- **PID + SAC:** residual reinforcement learning. The PID runs underneath and SAC learns bounded corrections (about ±3 mg/kg/h on the infusion, 0.6× to 1.4× on the induction bolus). With zero correction it is exactly the PID.

| Metric | PID (tuned) | SAC (3 seeds) | PID + SAC (3 seeds) |
|---|---|---|---|
| Time in target, BIS 40-60 (%) | **81.2** | 71.3 ± 1.8 | 78.5 ± 0.8 |
| MDAPE, inaccuracy (%) | **8.0** | 11.8 ± 1.1 | 8.4 ± 0.4 |
| MDPE, bias (%) | 1.9 | **0.5 ± 0.3** | 2.1 ± 0.1 |
| Wobble (%) | **6.1** | 7.4 ± 1.0 | 6.7 ± 0.2 |
| Time too deep, BIS < 40 (%) | **8.8** | 14.4 ± 2.2 | 11.9 ± 0.9 |
| Lowest BIS reached | **33.1** | 32.4 ± 4.8 | 29.6 ± 0.9 |
| Induction time (min) | 2.0 | 1.7 ± 0.8 | **1.2 ± 0.1** |
| Maintenance propofol (mg/kg/h) | 7.1 | 6.8 ± 0.7 | **6.7 ± 0.1** |

Per patient: pure SAC beat the PID on 3 of 30 patients; PID + SAC beat it on 9 of 30 and was worse on 18. Metrics are for the maintenance phase; ± is the spread across training seeds.

<p>
<img src="media/paired_patients.png" width="38%"> <img src="media/learning_curves.png" width="58%">
</p>

*Left: every dot is one unseen patient; above the diagonal, the RL controller did better than the PID. Right: validation learning curves (pure SAC was validated on 8 patients, PID + SAC on 20; neither uses the test set).*

![BIS traces on six test patients](media/test_traces.png)

*PID vs PID + SAC on six test patients.*

## What the agents learned, and why they lost

**Pure SAC found a shortcut.** In older patients it learned to skip the induction bolus and hold off for about three minutes, then run the infusion at maximum. It learned this because the PID's fixed bolus overdoses older patients (the video shows BIS below 10 for ten minutes), and waiting avoids that. But a patient left awake at the start of surgery is a failure a clinician would never accept. Adding a penalty for staying too light did not remove the shortcut. The takeaway: an agent optimises exactly the reward you write, including its loopholes.

**Residual SAC could not beat the controller it started from.** It begins as the PID and only has to learn small improvements. On the validation patients its corrections made its own reward worse (211 against 285 for no correction at all), so this is a learning failure, not a reward-design problem. The likely cause is credit assignment: a dose change only shows up in the measured BIS 30 to 60 seconds later (drug equilibration plus the 20 s monitor delay), and the PID underneath partly cancels every correction. The agent cannot see clearly what its own actions did.

**Where RL did help.** PID + SAC induced faster (1.2 vs 2.0 min), used slightly less propofol, and beat the PID on 9 of 30 patients. Pure SAC had the lowest bias of the three.

**Training was unstable.** Pure SAC's learning curves differ a lot between seeds and collapse and recover mid-training (learning curves above), which is why every result here uses three seeds and a best-on-validation checkpoint.

## What would come next

- **Handle the delay explicitly:** a recurrent policy or a stacked history of BIS and doses, so the agent can connect actions to effects that arrive a minute later.
- **Start from the PID's behaviour:** pre-train the policy to imitate the PID, then fine-tune with RL, instead of learning from random dosing.
- **A stronger classical baseline:** model predictive control using the patient model, which is what RL would really need to beat.
- **Individualised induction:** the clearest weakness of both the PID and SAC is the first five minutes, where a weight- and age-adjusted induction strategy has the most to gain.

## How the evaluation is kept fair

- **Held-out patients.** Test patients come from a separate random stream and are never used for training or tuning.
- **A properly tuned baseline.** The PID gains are grid-searched on a separate tuning population, and it uses clinical-style dosing: an age-adjusted induction bolus, then PI control with anti-windup.
- **Common random numbers.** For each test patient, both controllers get identical monitor noise and stimulation timing, so differences come from the controller, not from luck.
- **Several training seeds.** SAC results are reported as mean ± spread across independent training runs.
- **Standard clinical metrics** from Varvel et al. (1992): bias (MDPE), inaccuracy (MDAPE), wobble, plus time in the 40–60 target band.

## How it works

| Part | What it does |
|---|---|
| `EleveldPatient.py` | Eleveld (2018) propofol PK/PD model: 3-compartment pharmacokinetics, effect-site delay, sigmoid BIS response. The model is linear, so it is stepped with an exact matrix-exponential update (checked against an ODE solver to within 3×10⁻⁷, about 100× faster). |
| `patients.py` | Samples adult patients (age 18–80, sex, height, BMI) with log-normal variation on volumes, clearances, ke0, Ce50 and baseline BIS. |
| `AnesthesiaEnv.py` | Gymnasium environment. 5 s control steps over a 40 min case: induction by bolus, then maintenance by infusion (0–20 mg/kg/h). Delayed noisy BIS monitor and surgical stimulation events. |
| `train_sac.py` | Stable-Baselines3 SAC, 8 patients simulated in parallel, a new patient every episode. Keeps the checkpoint that scores best on validation patients. |
| `residual.py` | Residual controller: the PID proposes an action and SAC learns a bounded correction. |
| `pid_baseline.py`, `tune_pid.py` | Clinical-style PID baseline and its grid search. |
| `evaluate.py`, `benchmark.py` | Metrics, benchmark on held-out patients, figures. |
| `make_video.py` | Renders the side-by-side video above. |

**Reward** (training only, uses the true simulated BIS): a Gaussian bonus for BIS near 50, penalties for leaving the 40-60 band in either direction (with an extra penalty below 25), and small costs for drug use and abrupt dose changes. The residual agent also pays a small cost for each correction it makes.

## Limitations

- This is a simulation. The patient model is a published population model with approximate inter-patient variation (half the published variances, so every patient is controllable within the infusion limits). It is not validated on real patient data.
- BIS only. No blood-pressure control, no opioid (remifentanil) co-administration, no drug interactions.
- The disturbance model for surgical stimulation is a simple shaped offset on BIS.
- SAC was trained for 500k steps and PID + SAC for 300k steps per seed (about 8 minutes on a laptop). Longer training or other algorithms may do better; the conclusions are about these runs.
- Nothing here is suitable for clinical use.

## Run it

```bash
pip install -r requirements.txt
python tune_pid.py                        # tune the PID baseline (~1 min)
python train_sac.py --seed 0 --steps 500000               # pure SAC (repeat for seeds 1, 2)
python train_sac.py --seed 0 --steps 300000 --residual    # PID + SAC (repeat for seeds 1, 2)
python benchmark.py                                       # results table and figures (~1 min)
python make_video.py --patient 0 --model models/sac_seed2.zip
```

## References

- Eleveld, D.J. et al. (2018). Pharmacokinetic–pharmacodynamic model for propofol for broad application in anaesthesia and sedation. *British Journal of Anaesthesia*, 120(5), 942–959.
- Haarnoja, T. et al. (2018). Soft Actor-Critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. *ICML*.
- Varvel, J.R. et al. (1992). Measuring the predictive performance of computer-controlled infusion pumps. *J Pharmacokinet Biopharm*, 20(1), 63–94.

*Started as coursework for BIOE70077 Reinforcement Learning for Bioengineers, Imperial College London, then extended with held-out evaluation, realistic monitoring, a tuned baseline and residual RL.*
