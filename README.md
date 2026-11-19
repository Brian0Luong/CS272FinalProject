# Communication-Efficient World Models for Connected Driving

A reinforcement learning project on when a vehicle should ask its neighbors for information before merging. In a simulated highway environment, a PPO policy decides whether to request a message. A model predictive controller (MPC) then uses the ego vehicle's observation and any reply to choose a driving action.

## How it works

- **Environment:** A HighwayEnv simulation with limited local sensing and six evaluation scenarios: `coop_easy`, `coop_front`, `coop_rear`, `lead_braking`, `mixed`, and `truck_occlusion`.
- **Request policy:** At each step, PPO chooses no request, front intent, rear intent, front LiDAR, or rear LiDAR. An intent reply costs 2 bits, a LiDAR reply costs 512 bits, and an unavailable reply costs 1 bit. The environment subtracts `bits_price × transmitted_bits / 1000` from the driving reward.
- **Driving policy:** The learned world model predicts the next observation and reward. MPC scores candidate three-step driving sequences and executes the first action of the highest-scoring sequence.
- **Baselines:** No communication, uniform random requests, four fixed single-message strategies, and full communication. The full-communication baseline requests all four replies at each step and averages the MPC sequence scores obtained from each reply separately.

## Setup

From the repository root, install the dependencies.

```bash
python -m pip install -r requirements.txt
```

## Evaluate the saved policies

Run from the repository root so the planner can find `world_model.keras`. The repository already includes the trained models; you don't need to train them to run the evaluator.

```bash
python evaluate_ppo.py
```

The evaluator defaults to cost `0.1`. To reproduce another cost, set both `BITS_PRICE` and `MODEL_PATH` at the top of [evaluate_ppo.py](evaluate_ppo.py), then run the same command.

| `BITS_PRICE` | `MODEL_PATH` | Saved report |
|---:|---|---|
| 0.0 | `cost_runs/cost_0/request_policy_smoke.zip` | [Cost 0.0](results/eval_cost_0p0.txt) |
| 0.1 | `request_policy_smoke.zip` | [Cost 0.1](results/eval_cost_0p1.txt) |
| 1.0 | `cost_runs/cost_1/request_policy_smoke.zip` | [Cost 1.0](results/eval_cost_1p0.txt) |

To display the output in PowerShell and save it to a file, set the cost and checkpoint from the table, then run the command for that cost:

```powershell
# Cost 0.0
python evaluate_ppo.py | Tee-Object -FilePath .\results\eval_cost_0p0.txt

# Cost 0.1
python evaluate_ppo.py | Tee-Object -FilePath .\results\eval_cost_0p1.txt

# Cost 1.0
python evaluate_ppo.py | Tee-Object -FilePath .\results\eval_cost_1p0.txt
```
Run one evaluation command at a time, changing both settings before the next run.

The evaluator uses 50 seeds per scenario (11–60), for 300 episodes per method at each cost. Methods share the same scenario–seed pairs. Each PPO checkpoint was trained with seed 11 and evaluated at its training price. The printed request distribution is ordered as no request, front intent, rear intent, front LiDAR, and rear LiDAR.

## Results

| Method | Cost | Merge success | Collision rate | Mean bits/episode | Mean net reward |
|---|---:|---:|---:|---:|---:|
| No communication | 0.1 | 76.7% | 23.3% | 0.0 | 5.116 |
| Random requests | 0.1 | 71.7% | 28.3% | 707.4 | 4.049 |
| Full communication | 0.1 | 76.0% | 24.0% | 3248.5 | 4.658 |
| PPO | 0.0 | 87.7% | 12.3% | 229.9 | 7.291 |
| PPO | 0.1 | 84.0% | 16.0% | 179.4 | 6.540 |
| PPO | 1.0 | 84.0% | 16.0% | 107.5 | 6.450 |

PPO trained at cost 1.0 matched the 0.1 policy's merge success while using **40% fewer bits per episode**. The free policy achieved the highest merge success. Net reward includes the communication charge at the listed price, so use merge success and bits to compare the bandwidth tradeoff across prices.

## Files and training

| What | Where |
|---|---|
| Driving environment and scenario checks | [environment.py](environment.py), [check_coop_environment.py](check_coop_environment.py) |
| PPO request agent and training entry points | [agent.py](agent.py), [myrunner.py](myrunner.py), [train_costs.py](train_costs.py) |
| World model and MPC | [world_model.py](world_model.py), [train_world_model.py](train_world_model.py), [mpc_planner.py](mpc_planner.py) |
| Raw training data and collector | [driving_dataset.npz](driving_dataset.npz), [collect_data.py](collect_data.py) |
| Trained models | [world_model.keras](world_model.keras) and the PPO checkpoint paths above |
| Evaluation | [evaluate_ppo.py](evaluate_ppo.py), [full_communication.py](full_communication.py), [eval_mpc.py](eval_mpc.py) |

To retrain PPO with the included world model, run `python myrunner.py` for cost 0.1 or `python train_costs.py` for costs 0.0 and 1.0. To retrain the world model from the included dataset, run `python train_world_model.py`; it replaces `world_model.keras`.

## Contributions

- **Brian Luong (`Brian0Luong`):** Driving environment, neighbor communication and LiDAR, cooperative scenarios, merge timing checks, original PPO wrapper and training loop, and communication cost sweep.
- **Tyler Le (`TheLuckyPerson`):** Data collection, world model and trainer, MPC planner and evaluator, and PPO evaluation callback.

[View the GitHub commit history](https://github.com/Brian0Luong/CS272FinalProject/commits/main/)
