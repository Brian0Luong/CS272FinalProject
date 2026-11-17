
from agent import train_ppo
from test_planner import planner
from mpc_planner import planner as mpc_plan

if __name__ == "__main__":
    model = train_ppo(
        planner=planner,  # or use planner
        total_timesteps=100_000,
        bits_price=0.1,
        render_mode="human"
    )
