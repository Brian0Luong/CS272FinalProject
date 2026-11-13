
from agent import train_ppo
from test_planner import planner

if __name__ == "__main__":
    model = train_ppo(
        planner=planner,
        total_timesteps=100_000,
        bits_price=0.1
    )
