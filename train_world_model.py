
import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from world_model import create_world_model

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

# split up training and validation datasets
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

model = create_world_model()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

history = model.fit(
    X[train_idx],
    {
        "next_state": y_state[train_idx],
        "reward": y_reward[train_idx]
    },
    validation_data=(
        X[val_idx],
        {
            "next_state": y_state[val_idx],
            "reward": y_reward[val_idx]
        }
    ),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping],
    verbose=1
)

model.save("world_model.keras")