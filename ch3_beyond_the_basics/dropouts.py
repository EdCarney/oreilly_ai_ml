import tensorflow as tf

# here we introduce the concept of 'dropout' layers; these will randomly
# deactivate a specified percentage of nodes (here 20%) in the preceding
# layer; this helps to avoid overfitting by ensuring nodes do not learn
# very similar values

# generally, this should be observable as accuracy values that are more
# similar b/w your training and testing sets; higher accuracy in your
# training data is a sign of overfitting

data = tf.keras.datasets.fashion_mnist

(training_imgs, training_lbls), (test_imgs, test_lbls) = data.load_data()

training_imgs = training_imgs / 255.0
test_imgs = test_imgs / 255.0

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ]
)

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(training_imgs, training_lbls, epochs=20)

model.evaluate(test_imgs, test_lbls)

classifications = model.predict(test_imgs)
print(classifications[0])
print(test_lbls[0])
