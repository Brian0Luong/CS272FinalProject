import tensorflow as tf

layers = tf.keras.layers
Model = tf.keras.Model


def create_world_model(input_dim=85, state_dim=36):
    inputs = layers.Input(shape=(input_dim,))

    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)

    next_state = layers.Dense(
        state_dim,
        activation="linear",
        name="next_state"
    )(x)

    reward = layers.Dense(
        1,
        activation="linear",
        name="reward"
    )(x)

    model = Model(
        inputs=inputs,
        outputs=[next_state, reward]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={
            "next_state": "mse",
            "reward": "mse"
        }
    )

    return model
