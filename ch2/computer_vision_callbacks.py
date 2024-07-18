import tensorflow as tf

# here we do the same thing as for computer_vision, but now we run our
# training with a callback that can update the state of the model; we use
# this to automatically stop when we hit a desired accuracy


class my_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy, terminating training")
            self.model.stop_training = True


data = tf.keras.datasets.fashion_mnist

(training_imgs, training_lbls), (test_imgs, test_lbls) = data.load_data()

training_imgs = training_imgs / 255.0
test_imgs = test_imgs / 255.0

model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cb = my_callback()
model.fit(training_imgs, training_lbls, epochs=50, callbacks=[cb])

model.evaluate(test_imgs, test_lbls)

classifications = model.predict(test_imgs)
print(classifications[0])
print(test_lbls[0])
