import tensorflow as tf

# CONVOLUTION LAYER
# we define the number of convolutions we want the model to learn as 64;
# (3, 3) is the filter size, i.e. the number of pixels that will be part
# of the convolution; we define the input shape initially, and the
# activation function per usual; we want the model to learn what the best
# filter values are for these convolutions

# MAX POOLING LAYER
# this groups ('pools') 2x2 sets of pixels together and takes the max, this
# has the affect of reducing the number of pixels by 4


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
        fget=_get_model
    )


class HorseHumanCnnModel:
    def __init__(self):
        self._model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                   input_shape=(300, 300, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
            ])

    def _get_model(self) -> tf.keras.Sequential:
        return self._model

    model = property(
        fget=_get_model
    )
