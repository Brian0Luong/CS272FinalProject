import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

data = np.load("driving_dataset.npz")

observations = data["observations"]
next_observations = data["next_observations"]
driving_actions = data["driving_actions"]
request_actions = data["request_actions"]
reply_kinds = data["reply_kinds"]
reply_neighbors = data["reply_neighbors"]
reply_data = data["reply_data"]
rewards = data["rewards"]
episode_ids = data["episode_ids"]

driving_onehot = np.eye(5)[driving_actions]
request_onehot = np.eye(5)[request_actions]
kind_onehot = np.eye(4)[reply_kinds]
neighbor_onehot = np.eye(3)[reply_neighbors]

X = np.concatenate([
    observations,
    driving_onehot,
    request_onehot,
    kind_onehot,
    neighbor_onehot,
    reply_data
], axis=1).astype(np.float32)

y_state = next_observations.astype(np.float32)
y_reward = rewards.astype(np.float32).reshape(-1, 1)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=11
)

train_idx, val_idx = next(splitter.split(
    X,
    y_state,
    groups=episode_ids
))

model = tf.keras.models.load_model("world_model.keras")

predicted_state, predicted_reward = model.predict(
    X[val_idx[:10]],
    verbose=0
)

for i in range(10):
    print("\nSample:", i)
    print("Actual reward:", y_reward[val_idx[i]])
    print("Predicted reward:", predicted_reward[i])
    print("State MSE:", np.mean(
        (y_state[val_idx[i]] - predicted_state[i]) ** 2
    ))

predicted_state, predicted_reward = model.predict(
    X[val_idx],
    verbose=0
)

print("\nOverall State MSE:",
      np.mean((y_state[val_idx] - predicted_state) ** 2))

print("Overall Reward MSE:",
      np.mean((y_reward[val_idx] - predicted_reward) ** 2))