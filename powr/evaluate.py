from typing import Any, Tuple

import tensorflow as tf

from powr.window import WindowGenerator


def evaluate_model(model: tf.keras.Model, window: WindowGenerator) -> Tuple[Any, Any]:
    """Evaluate the given model on the given window.

    Args:
        model (tf.keras.Model): the model to evaluate
        window (WindowGenerator): window generator with dataset to evaluate on
    """
    val_performance = model.evaluate(window.val)
    test_performance = model.evaluate(window.test)

    return val_performance, test_performance
