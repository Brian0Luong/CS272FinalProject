import itertools

import numpy as np

from environment import DrivingEnv
from mpc_planner import ACTIONS, DISCOUNT, HORIZON, encode_reply, model


class FullCommunicationEnv(DrivingEnv):
    """request all four replies in one step"""

    def request_all(self):
        if self.episode_done:
            raise RuntimeError("Reset before requesting in a finished episode")
        if self.pending:
            raise RuntimeError("A request was already made for this step")

        old_requests, old_bits = self.requests, self.bits
        replies = {}
        try:
            for action in (1, 2, 3, 4):
                # A request reads the current state without advancing time.
                self.pending = False
                replies[action] = self.request(action)
        except Exception:
            self.pending = False
            self.requests, self.bits = old_requests, old_bits
            raise

        # The last request leaves pending=True, as required by step().
        return {
            "replies": replies,
            "bits": sum(reply["bits"] for reply in replies.values()),
        }


def _score_sequences(obs, reply, request_action):
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

    return sequences, total_rewards


def plan_full(obs, bundle):
    """choose sequence with best mean score across all four replies"""
    scores_by_reply = []
    for request_action, reply in bundle["replies"].items():
        sequences, scores = _score_sequences(obs, reply, request_action)
        scores_by_reply.append(scores)

    mean_scores = np.mean(scores_by_reply, axis=0)
    return int(sequences[np.argmax(mean_scores), 0])


def main():
    """one-step smoke check"""
    env = FullCommunicationEnv(
        scenario="coop_front", bits_price=0.1, render_mode=None
    )
    try:
        obs, _ = env.reset(seed=10000)
        bundle = env.request_all()
        action = plan_full(obs, bundle)
        _, _, _, _, info = env.step(action, bundle)

        expected_cost = env.bits_price * bundle["bits"] / 1000
        assert len(bundle["replies"]) == env.requests == 4
        assert info["communication_bits"] == bundle["bits"] == env.bits
        assert np.isclose(info["communication_cost"], expected_cost)
        print(
            f"Full communication: {env.requests} requests, {env.bits} bits, "
            f"action={action}, cost={info['communication_cost']:.4f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
