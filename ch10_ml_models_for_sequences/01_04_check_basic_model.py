from generate_data import generate_test_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def windowed_dataset(series, window_sz, batch_sz, shuffle_buf) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_sz + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_sz + 1))
    dataset = dataset.shuffle(shuffle_buf)
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_sz).prefetch(1)
    return dataset


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

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, input_shape=[window_sz], activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
    ])
sgd = tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9)
model.compile(loss='mse', optimizer=sgd)
model.fit(dataset, epochs=100, verbose=1)

# now that we have a model, we can predict the values given our windowed data;
# note we need to shift the series right to accommadate the window size

print('Forecasting using only prediction data...')
true_forecast = series[split_time - window_sz:split_time]
for i in range(len(series) - split_time):
    window_data = true_forecast[-window_sz:]
    true_forecast = np.append(
            true_forecast,
            model.predict(window_data[np.newaxis], verbose=0)
            )
print('...done')

print('Forecasting using all available data...')
forecast = []
for i in range(len(series) - window_sz):
    forecast.append(
            model.predict(series[i:window_sz + i][np.newaxis], verbose=0)
            )
print('...done')

# we ignore all the values before the split time, since that was our training
# data; note we need to reverse the right shift we did earlier, but only for
# the non-true forecast, since the true forecast actually has all the data

forecast = forecast[split_time - window_sz:]
true_forecast = true_forecast[window_sz:]

# lastly, we have to reformat the data, since the model outputs a number x as
# [[x]] and we have an array of these values; we will convert this to an np
# array to use slicing to extract the values in the format we want

results = np.array(forecast)[:, 0, 0]
true_results = np.array(true_forecast)

plt.plot(time_valid, x_valid)
plt.plot(time_valid, results)
plt.plot(time_valid, true_results)
plt.show()
