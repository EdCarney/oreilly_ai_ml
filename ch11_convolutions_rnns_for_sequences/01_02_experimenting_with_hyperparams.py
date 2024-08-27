from generate_data import generate_test_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_tuner as kt


# now we will add a convolutional layer to aid in model predictions, similar to computer vision,
# the idea is to identify 'features' of the data and use these as part of the prediction; we will
# use 1D convolution in this case


# note we need to add another axis to the series data to ensure it is passed in to the model with
# the expected dimensions; do this with tf.expand_dims()

hp = kt.HyperParameters()


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


def build_model(hp: kt.HyperParameters):
    window_sz = 20
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv1D(
            filters=hp.Int(name="units", min_value=128, max_value=256, step=64),
            kernel_size=hp.Int(name="kernels", min_value=3, max_value=9, step=3),
            strides=hp.Int(name="strides", min_value=1, max_value=3, step=1),
            padding="causal",
            activation="relu",
            input_shape=[None, 1],
        )
    )
    model.add(tf.keras.layers.Dense(28, activation="relu"))
    model.add(tf.keras.layers.Dense(10, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.legacy.SGD(momentum=0.5, learning_rate=1e-5),
    )
    return model


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

# note that the results with a conv1d layer were actually slightly worse or at a minimum not
# very much improved from the non-cnn design; this could be due to poor choice of hyperparams
# (in this case for the convolutional layer, would be stride len, kernel size, and filter #)
#
# we will use the keras tuner to help select these params

tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="loss",
    max_trials=500,
    executions_per_trial=3,
    directory="my_dir",
    project_name="cnn-tuner",
)

tuner.search_space_summary()
tuner.search(dataset, epochs=100, verbose=2)
tuner.results_summary()
