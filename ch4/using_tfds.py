import tensorflow as tf
import tensorflow_datasets as tfds

(training_img, training_lbl), (test_img, test_lbl) = \
    tfds.as_numpy(tfds.load('fashion_mnist',
                            split=['train', 'test'],
                            batch_size=-1,
                            as_supervised=True))

training_img = training_img / 255.0
test_img = test_img / 255.0

model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

model.fit(training_img, training_lbl, epochs=5)
