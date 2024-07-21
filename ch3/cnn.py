import tensorflow as tf


class CnnModel:
    def __init__(self):
        self._model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu,
                                   input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

    def _get_model(self) -> tf.keras.Sequential:
        return self._model

    model: tf.keras.Sequential = property(
        fget = _get_model
    )