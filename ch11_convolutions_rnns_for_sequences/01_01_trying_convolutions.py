from generate_data import generate_test_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# now we will add a convolutional layer to aid in model predictions, similar to computer vision,
# the idea is to identify 'features' of the data and use these as part of the prediction; we will
# use 1D convolution in this case


# note we need to add another axis to the series data to ensure it is passed in to the model with
# the expected dimensions; do this with tf.expand_dims()


def windowed_dataset(series, window_sz, batch_sz, shuffle_buf) -> tf.data.Dataset:
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_sz + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_sz + 1))
    dataset = dataset.shuffle(shuffle_buf)
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_sz).prefetch(1)
    return dataset


def model_forecast(model, series, window_size):
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size))
    dataset = dataset.batch(32).prefetch(1)
    forecast = model.predict(dataset)
    return forecast


split_time = 1000

(time, series) = generate_test_data()

time_train = time[:split_time]
x_train = series[:split_time]

time_valid = time[split_time:]
x_valid = series[split_time:]

window_sz = 20
batch_sz = 32
suffle_buf = 1000

dataset = windowed_dataset(x_train, window_sz, batch_sz, suffle_buf)

# we add a single convolutional layer; filters is the number of filters for the layer to learn,
# kernel_size is the number of elements the filter considers when reducing the data; strides is
# essentially the 'step', padding dictates how to handle the values that will be lost at the 'ends'
# of the data (causal will only take data from current and past timesteps), activation func is
# what it sounds like (relu here essentially means to reject negative values)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=3,
            strides=1,
            padding="causal",
            activation="relu",
            input_shape=[None, 1],
        ),
        tf.keras.layers.Dense(28, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

optimizer = tf.keras.optimizers.legacy.SGD(lr=1e-5, momentum=0.5)
model.compile(loss="mse", optimizer=optimizer)
history = model.fit(dataset, epochs=100, verbose="1")

forecast = model_forecast(model, series[..., np.newaxis], window_sz)
results = forecast[split_time - window_sz : -1, -1, 0]

print(results[:20])
print(x_valid[:20])

plt.plot(time_valid, x_valid)
plt.plot(time_valid, results)
plt.show()
