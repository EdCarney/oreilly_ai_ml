from generate_data import generate_test_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# from our hyperparam tuning, we know that we had the best predictions with the following params:
#   units  = 256
#   kernel = 6
#   stride = 1
# will now try training with these params
# ...and...
# we can see a significant improvement in the results in terms of MSE and MAE


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

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv1D(
            filters=256,
            kernel_size=6,
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

optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=1e-5, momentum=0.5)
model.compile(loss="mse", optimizer=optimizer)

print("Starting training...")
history = model.fit(dataset, epochs=100, verbose="0")
print("...done")

forecast = model_forecast(model, series[..., np.newaxis], window_sz)
results = forecast[split_time - window_sz : -1, -1, 0]

print("MSE", tf.keras.metrics.mean_squared_error(results, x_valid))
print("MAE", tf.keras.metrics.mean_absolute_error(results, x_valid))

plt.plot(time_valid, x_valid)
plt.plot(time_valid, results)
plt.show()
