"""Run the cooperative driving environment checks"""

from environment import DrivingEnv

SCENARIOS = ("coop_easy", "coop_front", "coop_rear")
SEEDS = range(20)
WAIT_STEPS = (0, 3, 6, 9)

NO_REQUEST = 0
FRONT_LIDAR = 3
REAR_LIDAR = 4
IDLE = 1
MERGE_RIGHT = 2


def smoke_test():
    """Check that each scenario can reset, return LiDAR, and take a step."""
    for scenario in (*SCENARIOS, "coop_mix"):
        env = DrivingEnv(bits_price=0, scenario=scenario)
        try:
            for seed in SEEDS:
                env.reset(seed=seed)
                request = REAR_LIDAR if seed % 2 else FRONT_LIDAR
                reply = env.request(request)
                if reply["kind"] != "lidar":
                    raise RuntimeError(
                        f"{scenario}, seed {seed}, request {request}: "
                        f"expected LiDAR, got {reply['kind']!r}"
                    )
                env.step(IDLE, reply)
            print(f"{scenario}: passed {len(SEEDS)} seeds")
        finally:
            env.close()


def check_merge_timing():
    """Compare scripted waits on the same seeds with no communication."""
    for scenario in SCENARIOS:
        env = DrivingEnv(bits_price=0, scenario=scenario)
        try:
            for wait in WAIT_STEPS:
                merges = collisions = other = 0
                for seed in SEEDS:
                    env.reset(seed=seed)
                    for step in range(100):
                        reply = env.request(NO_REQUEST)
                        action = IDLE if step < wait else MERGE_RIGHT
                        _, _, terminated, truncated, info = env.step(action, reply)
                        if terminated or truncated:
                            merges += int(info["success"])
                            collisions += int(info["collision"])
                            other += int(not (info["success"] or info["collision"]))
                            break
                    else:
                        other += 1

                total = len(SEEDS)
                line = (f"{scenario}, wait {wait}: {merges}/{total} merges, "
                        f"{collisions}/{total} collisions")
                if other:
                    line += f", {other}/{total} without a merge or collision"
                print(line)
        finally:
            env.close()


if __name__ == "__main__":
    print("Scenario smoke test")
    smoke_test()
    print("\nMerge timing without communication")
    check_merge_timing()
