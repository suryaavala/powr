from typing import Tuple

import tensorflow as tf

from powr.window import WindowGenerator


def build_model(output_steps: int, num_features: int) -> tf.keras.Model:
    """Build a model with the given output steps and number of features.

    Args:
        output_steps (int): The number of time steps to predict into the future
        num_features (int): The number of features in the input data

    Returns:
        tf.keras.Model: the model (uncompiled)
    """
    multi_linear_model = tf.keras.Sequential(
        [
            # Take the last time-step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, output_steps*features]
            tf.keras.layers.Dense(
                output_steps * num_features, kernel_initializer=tf.initializers.zeros()
            ),
            # Shape => [batch, window_size, features]
            tf.keras.layers.Reshape([output_steps, num_features]),
        ]
    )
    return multi_linear_model


def train_model(
    model: tf.keras.Model, window: WindowGenerator, epochs: int, patience=2
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """Train the given model on the given window.

    Args:
        model (tf.keras.Model): the model to train
        window (WindowGenerator): window generator with dataset to train on
        epochs (int): the number of epochs to train for
        patience (int): the number of epochs to wait before early stopping

    Returns:
        Tuple[tf.keras.Model, tf.keras.callbacks.History]: the trained model and the training history
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, mode="min"
    )

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    history = model.fit(
        window.train,
        epochs=epochs,
        validation_data=window.val,
        callbacks=[early_stopping],
    )
    return model, history
