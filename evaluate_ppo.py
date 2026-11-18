
from stable_baselines3 import PPO
from agent import RequestEnv
from mpc_planner import planner

import numpy as np

model = PPO.load("./models/best/best_model")

env = RequestEnv(
    planner=planner,
    bits_price=0.1
)

episodes = 50

successes = 0
collisions = 0
total_rewards = 0
total_bits = 0
request_counts = np.zeros(5, dtype=int)

try:
    for episode in range(episodes):
        obs, info = env.reset(seed=episode + 11)

        terminated = False
        truncated = False
        episode_reward = 0

        while not terminated and not truncated:
            action, _ = model.predict(
                obs,
                deterministic=True
            )

            action = int(action)
            request_counts[action] += 1

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            episode_reward += reward
            total_bits += info["communication_bits"]

        successes += int(info["success"])
        collisions += int(info["collision"])
        total_rewards += episode_reward

    print("Success rate:", successes / episodes)
    print("Collision rate:", collisions / episodes)
    print("Average reward:", total_rewards / episodes)
    print("Average bits:", total_bits / episodes)
    print("Request counts:", request_counts)

finally:
    env.close()
