import os
from pathlib import Path

from agent import train_ppo
from mpc_planner import planner as mpc_plan

if __name__ == "__main__":
    project = Path.cwd()

    for label, price in [("cost_0", 0.0), ("cost_1", 1.0)]:
        run_dir = project / "cost_runs" / label
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"Training PPO at bits_price={price}", flush=True)
        try:
            os.chdir(run_dir)
            train_ppo(
                planner=mpc_plan,
                total_timesteps=100_000,
                bits_price=price,
                render_mode=None,
            )
        finally:
            os.chdir(project)