
import numpy as np
from environment import DrivingEnv
from mpc_planner import planner as mpc_plan
from test_planner import planner

scenarios = [
    "lead_braking",
    "truck_occlusion",
    "mixed",
    "coop_front",
    "coop_easy",
    "coop_rear",
    "coop_mix"
]

episodes_per_scenario = 10

sim = DrivingEnv(bits_price=0.1, render_mode="human")

results = []

try:
    for scenario in scenarios:
        for episode in range(episodes_per_scenario):
            sim.scenario_choice = scenario
            obs, info = sim.reset(seed=0 + episode)

            terminated = False
            truncated = False
            episode_reward = 0.0

            while not terminated and not truncated:
                request_action = 0
                reply = sim.request(request_action)

                driving_action = mpc_plan(
                    obs,
                    reply,
                    request_action
                )

                obs, reward, terminated, truncated, info = sim.step(
                    driving_action,
                    reply
                )

                episode_reward += reward

            results.append({
                "scenario": scenario,
                "success": info["success"],
                "collision": info["collision"],
                "steps": info["steps"],
                "reward": episode_reward,
                "bits": sim.bits
            })

            print(
                scenario,
                "Episode:", episode + 1,
                "Success:", info["success"],
                "Collision:", info["collision"],
                "Steps:", info["steps"],
                "Reward:", round(episode_reward, 3)
            )

finally:
    sim.close()

print("\nOverall results")
print("Success rate:", np.mean([r["success"] for r in results]))
print("Collision rate:", np.mean([r["collision"] for r in results]))
print("Average steps:", np.mean([r["steps"] for r in results]))
print("Average reward:", np.mean([r["reward"] for r in results]))
print("Average bits:", np.mean([r["bits"] for r in results]))
