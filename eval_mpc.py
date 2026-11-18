
import numpy as np
from environment import DrivingEnv
from mpc_planner import planner as mpc_plan
from test_planner import planner as heuristic_plan

scenarios = [
    # "lead_braking",
    # "truck_occlusion",
    # "mixed",
    # "coop_easy",
    "coop_front",
    # "coop_rear",
    # "coop_mix"
]
episodes_per_scenario = 50
seeds = [11 + i for i in range(episodes_per_scenario)]

def evaluate(planner, name):
    sim = DrivingEnv(bits_price=0.1, render_mode=None)
    results = []

    try:
        for scenario in scenarios:
            for seed in seeds:
                sim.scenario_choice = scenario
                obs, info = sim.reset(seed=seed)

                terminated = False
                truncated = False
                episode_reward = 0.0
                actions = []
                while not terminated and not truncated:
                    request_action = 1
                    reply = sim.request(request_action)

                    driving_action = planner(
                        obs,
                        reply,
                        request_action
                    )

                    obs, reward, terminated, truncated, info = sim.step(
                        driving_action,
                        reply
                    )

                    episode_reward += reward
                    actions.append(int(driving_action))

                results.append({
                    "name": name,
                    "seed": seed,
                    "scenario": scenario,
                    "success": info["success"],
                    "collision": info["collision"],
                    "steps": info["steps"],
                    "reward": episode_reward,
                    "bits": sim.bits,
                    "actions": actions
                })

    finally:
        sim.close()

    print(f"\nOverall results for {name}")
    print("Success rate:", np.mean([r["success"] for r in results]))
    print("Collision rate:", np.mean([r["collision"] for r in results]))
    print("Average steps:", np.mean([r["steps"] for r in results]))
    print("Average reward:", np.mean([r["reward"] for r in results]))
    print("Average bits:", np.mean([r["bits"] for r in results]))

    return results


heuristic_results = evaluate(heuristic_plan, "hueristic")
mpc_results = evaluate(mpc_plan, "mpc")

print("\nPer-episode comparison")

for h, m in zip(heuristic_results, mpc_results):
    print(f"\nSeed {h['seed']}")
    print(f"Heuristic actions: {h['actions']}")
    print(f"MPC actions:       {m['actions']}")
    print(
        f"Heuristic: success={h['success']}, "
        f"collision={h['collision']}"
    )
    print(
        f"MPC:       success={m['success']}, "
        f"collision={m['collision']}"
    )