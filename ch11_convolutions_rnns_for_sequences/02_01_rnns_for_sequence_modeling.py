import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from read_station_data import get_normalized_data


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


time, series = get_normalized_data()

# use 94% of the data for training
split_time = floor(0.94 * len(time))

time_train = time[:split_time]
series_train = series[:split_time]

time_valid = time[split_time:]
series_valid = series[split_time:]

# create datasets per normal

window_sz = 24
batch_sz = 12
shuffle_buf_sz = 48

dataset = windowed_dataset(series, window_sz, batch_sz, shuffle_buf_sz)
dataset_valid = windowed_dataset(series_valid, window_sz, batch_sz, shuffle_buf_sz)

# we use a simple RNN implementation here, where the output of each timestep
# is fed into the next timestep

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.SimpleRNN(
            units=100, return_sequences=True, input_shape=[None, 1]
        ),
        tf.keras.layers.SimpleRNN(units=100),
        tf.keras.layers.Dense(units=1),
    ]
)

optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=1.5e-6, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])

print("Starting training...")
history = model.fit(dataset, epochs=100, verbose="auto", validation_data=dataset_valid)
print("...done")


forecast = model_forecast(model, series, window_sz)
results = forecast[split_time - window_sz : -1, -1]

print("MSE", tf.keras.metrics.mean_squared_error(results, series_valid))
print("MAE", tf.keras.metrics.mean_absolute_error(results, series_valid))

# plot results

plt.plot(time_valid, series_valid, label="Truth")
plt.plot(time_valid, results, label="Prediction")
plt.legend()
plt.show()
