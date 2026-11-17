
import itertools
import numpy as np
import tensorflow as tf

LANE_LEFT = 0
MAINTAIN = 1
LANE_RIGHT = 2
ACCELERATE = 3
BRAKE = 4

model = tf.keras.models.load_model("world_model.keras")

HORIZON = 3
DISCOUNT = 0.95
ACTIONS = [LANE_LEFT, MAINTAIN, LANE_RIGHT, ACCELERATE, BRAKE]


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

def planner(obs, reply, request_action):
    sequences = np.array(
        list(itertools.product(ACTIONS, repeat=HORIZON))
    )

    num_sequences = len(sequences)

    states = np.repeat(
        np.asarray(obs, dtype=np.float32)[None, :],
        num_sequences,
        axis=0
    )

    total_rewards = np.zeros(num_sequences, dtype=np.float32)

    kind, neighbor, reply_data = encode_reply(reply)

    for step in range(HORIZON):
        driving_actions = sequences[:, step]

        driving_onehot = np.eye(5)[driving_actions]

        if step == 0:
            request_onehot = np.tile(
                np.eye(5)[request_action],
                (num_sequences, 1)
            )

            kind_onehot = np.tile(
                np.eye(4)[kind],
                (num_sequences, 1)
            )

            neighbor_onehot = np.tile(
                np.eye(3)[neighbor],
                (num_sequences, 1)
            )

            reply_batch = np.tile(
                reply_data,
                (num_sequences, 1)
            )

        else:
            request_onehot = np.tile(
                np.eye(5)[0],
                (num_sequences, 1)
            )

            kind_onehot = np.tile(
                np.eye(4)[0],
                (num_sequences, 1)
            )

            neighbor_onehot = np.tile(
                np.eye(3)[0],
                (num_sequences, 1)
            )

            reply_batch = np.zeros(
                (num_sequences, 32),
                dtype=np.float32
            )

        model_input = np.concatenate([
            states,
            driving_onehot,
            request_onehot,
            kind_onehot,
            neighbor_onehot,
            reply_batch
        ], axis=1).astype(np.float32)

        predicted_states, predicted_rewards = model.predict(
            model_input,
            verbose=0
        )

        total_rewards += (DISCOUNT ** step) * predicted_rewards.flatten()

        states = predicted_states

    best_sequence = np.argmax(total_rewards)

    return int(sequences[best_sequence, 0])
