from generate_data import generate_test_data
import tensorflow as tf


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

# now we will create a very simple DNN model to predict the values
# note that because we are using a tf Dataset, this is very easy
#
# we will start with a very simple model; note that the final layer is just a
# single neuron that we do not have an activation function for, this is because
# we want the final value to be a float, so we just take the raw value output

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(10, input_shape=[window_sz], activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1),
    ]
)

# we specify mean-squared-error as the loss function and stochastic gradient
# decent for our optimizer; these are good choices for what ultimately boils
# down to a regression problem

sgd = tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9)
model.compile(loss="mse", optimizer=sgd)
model.fit(dataset, epochs=100, verbose="1")
