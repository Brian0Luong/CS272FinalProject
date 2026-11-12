import gymnasium as gym
import numpy as np
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from environment import DrivingEnv

class RequestEnv(gym.Env):
    def __init__(self, planner, bits_price=0.1):
        super().__init__()
        self.sim = DrivingEnv(bits_price=bits_price)
        self.planner = planner
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        scenario_seed = int(self.np_random.integers(0, 2**31))
        self.obs, info = self.sim.reset(seed=scenario_seed)
        return self.obs, info

    def step(self, request_action):
        reply = self.sim.request(int(request_action))
        driving_action = self.planner(self.obs, reply)

        self.obs, reward, terminated, truncated, info = self.sim.step(
            int(driving_action), reply
        )
        return self.obs, reward, terminated, truncated, info

    def close(self):
        self.sim.close()

def train_ppo(planner, total_timesteps=4_096, bits_price=0.1):
    env = RequestEnv(planner, bits_price=bits_price)
    try:
        check_env(env)
        policy = PPO("MlpPolicy", env, seed=11, verbose=1)
        policy.learn(total_timesteps=total_timesteps)
        policy.save("request_policy_smoke")
    finally:
        env.close()
