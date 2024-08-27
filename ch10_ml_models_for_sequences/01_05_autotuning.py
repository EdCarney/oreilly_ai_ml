from generate_data import generate_test_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_tuner as kt

hp = kt.HyperParameters()


def windowed_dataset(series, window_sz, batch_sz, shuffle_buf) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_sz + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_sz + 1))
    dataset = dataset.shuffle(shuffle_buf)
    dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
    dataset = dataset.batch(batch_sz).prefetch(1)
    return dataset


def build_model(hp: kt.HyperParameters):
    window_sz = 20
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Dense(
            units=hp.Int(name="units", min_value=10, max_value=30, step=2),
            activation="relu",
            input_shape=[window_sz],
        )
    )
    model.add(tf.keras.layers.Dense(10, activation="relu"))
    model.add(tf.keras.layers.Dense(1))
    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.legacy.SGD(
            momentum=hp.Choice(name="momentum", values=[0.9, 0.7, 0.5, 0.3]),
            learning_rate=1e-5,
        ),
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

tuner = kt.RandomSearch(
    hypermodel=build_model,
    objective="loss",
    max_trials=150,
    executions_per_trial=3,
    directory="my_dir",
    project_name="hello",
)

tuner.search(dataset, epochs=100, verbose=1)
tuner.results_summary()
