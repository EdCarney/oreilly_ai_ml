import tensorflow as tf

data = tf.keras.datasets.fashion_mnist

(training_imgs, training_lbls), (test_imgs, test_lbls) = data.load_data()

# normalize the test data values to 0-1
training_imgs = training_imgs / 255.0
test_imgs = test_imgs / 255.0

model = tf.keras.models.Sequential([
        # optional shape dimension for the data
        tf.keras.layers.Flatten(input_shape=(28, 28)),

        # middle layer; every node takes in all 184 (28*28) values
        # activation function is the logic that executes on each node
        # RELU (rectified linear unit) just returns a value if > 0
        tf.keras.layers.Dense(128, activation=tf.nn.relu),

        # output layer; one node for each category
        # we only want the value from the node with the highest value
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

# ADAM optimizer is an improvement on SGD (stochastic gradient descent)
# the loss function is specified as one of the 'categorical' set
# we also add a metric here to report on; there are a defined set of these
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train the model
history: tf.keras.callbacks.History = model.fit(training_imgs, training_lbls, epochs=5)

# check the model's accuracy
model.evaluate(test_imgs, test_lbls)

# now let's get some info on one of the pieces of test data
# note that classifications will return an array of output node values
classifications = model.predict(test_imgs)
print(classifications[0])
print(test_lbls[0])
print(history.history)
