import numpy as np
from environment import DrivingEnv
from test_planner import planner


def encode_reply(reply):
    """
    converts responses from environment into data usable by the network
    """
    kind_map = {
        "none": 0,
        "intent": 1,
        "lidar": 2,
        "unavailable": 3
    }

    neighbor_map = {
        None: 0,
        "front": 1,
        "rear": 2
    }

    data = np.zeros(32, dtype=np.float32)

    if reply["kind"] == "intent":
        intent_map = {
            "maintain": 0,
            "accelerate": 1,
            "brake": 2
        }
        data[intent_map[reply["data"]]] = 1.0

    elif reply["kind"] == "lidar":
        data = (
                np.asarray(reply["data"], dtype=np.float32).reshape(-1)
                / 32767.0
        )

    return (
        kind_map[reply["kind"]],
        neighbor_map[reply["neighbor"]],
        data
    )



def collect_data(episodes=500, output_file="driving_dataset.npz"):
    """
    tests planner under multiple differernt scenarios and saves dataset to a file
    """
    rng = np.random.default_rng(11)

    scenarios = [
        "lead_braking",
        "truck_occlusion",
        "mixed",
        "wide_gap",
        "closing_gap"
    ]

    sim = DrivingEnv(bits_price=0.1)

    observations = []
    next_observations = []
    driving_actions = []
    request_actions = []
    reply_kinds = []
    reply_neighbors = []
    reply_data = []
    rewards = []
    terminated_values = []
    truncated_values = []
    communication_bits = []
    communication_costs = []
    episode_ids = []
    scenario_names = []
    successes = []
    collisions = []

    try:
        for episode in range(episodes):
            sim.scenario_choice = str(rng.choice(scenarios))

            obs, info = sim.reset(seed=11 + episode)

            terminated = False
            truncated = False

            while not terminated and not truncated:
                request_action = int(rng.integers(0, 5))

                reply = sim.request(request_action)

                driving_action = int(planner(obs, reply))

                next_obs, reward, terminated, truncated, info = sim.step(
                    driving_action, reply
                )

                kind, neighbor, encoded_data = encode_reply(reply)

                observations.append(obs.copy())
                next_observations.append(next_obs.copy())
                driving_actions.append(driving_action)
                request_actions.append(request_action)
                reply_kinds.append(kind)
                reply_neighbors.append(neighbor)
                reply_data.append(encoded_data)
                rewards.append(reward)
                terminated_values.append(terminated)
                truncated_values.append(truncated)
                communication_bits.append(info["communication_bits"])
                communication_costs.append(info["communication_cost"])
                episode_ids.append(episode)
                scenario_names.append(sim.scenario)
                successes.append(info["success"])
                collisions.append(info["collision"])

                obs = next_obs

            print(
                f"Episode {episode + 1}/{episodes} | "
                f"Scenario: {sim.scenario} | "
                f"Success: {info['success']} | "
                f"Collision: {info['collision']} | "
                f"Steps: {info['steps']} | "
                f"Requests: {sim.requests} | "
                f"Bits: {sim.bits}"
            )

    finally:
        sim.close()

    np.savez_compressed(
        output_file,
        observations=np.asarray(observations, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        driving_actions=np.asarray(driving_actions, dtype=np.int64),
        request_actions=np.asarray(request_actions, dtype=np.int64),
        reply_kinds=np.asarray(reply_kinds, dtype=np.int64),
        reply_neighbors=np.asarray(reply_neighbors, dtype=np.int64),
        reply_data=np.asarray(reply_data, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        terminated=np.asarray(terminated_values, dtype=bool),
        truncated=np.asarray(truncated_values, dtype=bool),
        communication_bits=np.asarray(communication_bits, dtype=np.int64),
        communication_costs=np.asarray(communication_costs, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        scenarios=np.asarray(scenario_names),
        successes=np.asarray(successes, dtype=bool),
        collisions=np.asarray(collisions, dtype=bool)
    )

    print(f"\nCollected {len(observations)} transitions.")
    print(f"Saved dataset to {output_file}")


if __name__ == "__main__":
    collect_data(episodes=500)
