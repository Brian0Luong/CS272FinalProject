
import numpy as np
from stable_baselines3 import PPO

from agent import RequestEnv
from mpc_planner import planner

MODEL_PATH = "request_policy_smoke"
BITS_PRICE = 0.1

SCENARIOS = [
    "coop_easy",
    "coop_front",
    "coop_rear",
    "lead_braking",
    "mixed",
    "truck_occlusion",
]

SEEDS = range(11, 61)

POLICIES = {
    "No communication": 0,
    "Front intent": 1,
    "Rear intent": 2,
    "Front LiDAR": 3,
    "Rear LiDAR": 4,
    "Random": "random",
    "PPO": "ppo",
}

def evaluate_episode(policy_name, policy_action, model, scenario, seed):
    env = RequestEnv(
        planner=planner,
        bits_price=BITS_PRICE,
        scenario=scenario,
    )

    rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)

    total_reward = 0.0
    total_bits = 0
    request_counts = np.zeros(5, dtype=int)

    terminated = False
    truncated = False

    try:
        while not (terminated or truncated):
            if policy_action == "ppo":
                action, _ = model.predict(obs, deterministic=True)
                action = int(np.asarray(action).item())
            elif policy_action == "random":
                action = int(rng.integers(0, 5))
            else:
                action = int(policy_action)

            request_counts[action] += 1

            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += float(reward)
            total_bits += int(info["request_bits"])

        return {
            "scenario": scenario,
            "seed": seed,
            "policy": policy_name,
            "reward": total_reward,
            "success": int(info.get("success", False)),
            "collision": int(info.get("collision", False)),
            "bits": total_bits,
            "requests": request_counts,
        }

    finally:
        env.close()


def main():
    model = PPO.load(MODEL_PATH)
    results = []

    for scenario in SCENARIOS:
        print(f"\nEvaluating scenario: {scenario}")

        for policy_name, policy_action in POLICIES.items():
            for seed in SEEDS:
                result = evaluate_episode(
                    policy_name,
                    policy_action,
                    model,
                    scenario,
                    seed,
                )
                results.append(result)

            print(f"  Finished: {policy_name}")

    print("\nOVERALL RESULTS")

    for policy_name in POLICIES:
        policy_results = [
            result for result in results
            if result["policy"] == policy_name
        ]

        count = len(policy_results)

        success_rate = np.mean([
            result["success"] for result in policy_results
        ]) * 100

        collision_rate = np.mean([
            result["collision"] for result in policy_results
        ]) * 100

        avg_reward = np.mean([
            result["reward"] for result in policy_results
        ])

        avg_bits = np.mean([
            result["bits"] for result in policy_results
        ])

        request_counts = np.sum(
            [result["requests"] for result in policy_results],
            axis=0,
        )

        total_requests = request_counts.sum()

        request_percentages = (
            request_counts / total_requests * 100
            if total_requests > 0
            else np.zeros(5)
        )

        print(f"\n{policy_name} ({count} episodes)")
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Collision rate: {collision_rate:.1f}%")
        print(f"  Average reward: {avg_reward:.3f}")
        print(f"  Average communication bits: {avg_bits:.1f}")
        print(f"  Request distribution: {request_percentages.round(1)}%")

    print("\nRESULTS BY SCENARIO")

    for scenario in SCENARIOS:
        print(f"\n{scenario}")

        for policy_name in POLICIES:
            subset = [
                result for result in results
                if result["scenario"] == scenario
                   and result["policy"] == policy_name
            ]

            success_rate = np.mean([
                result["success"] for result in subset
            ]) * 100

            collision_rate = np.mean([
                result["collision"] for result in subset
            ]) * 100

            avg_reward = np.mean([
                result["reward"] for result in subset
            ])

            avg_bits = np.mean([
                result["bits"] for result in subset
            ])

            print(
                f"  {policy_name:20s} "
                f"Success: {success_rate:5.1f}%  "
                f"Collision: {collision_rate:5.1f}%  "
                f"Reward: {avg_reward:7.3f}  "
                f"Bits: {avg_bits:7.1f}"
            )


if __name__ == "__main__":
    main()
