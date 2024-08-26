import tensorflow as tf
import numpy as np


# define the layer(s) that you will use
# here we just use one densely connected layer with one node
# (densely connected => every node connected to every other)
layer = tf.keras.layers.Dense(units=1, input_shape=[1])


# a model is defined as a sequence of layers
model = tf.keras.Sequential(layer)

# the model is then compiled with a selected optimizer and loss
# loss: used to check the diff b/w guess and the correct value
# optimizer: used to move the model values toward the correct ones
model.compile(optimizer="sgd", loss="mean_squared_error")

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# iterate 500 times to try to get the correct result
model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
print(layer.get_weights())
