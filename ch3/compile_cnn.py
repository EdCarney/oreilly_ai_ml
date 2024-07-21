import tensorflow as tf
from cnn import  CnnModel

data = tf.keras.datasets.fashion_mnist

(training_img, training_lbl), (test_img, test_lbl) = data.load_data()

# the convolution layer works on image data directly, so we will pass
# that in; but it also requires a 'depth' layer for each image as this
# is typically the color layer used to indicate RGB values; for grayscale
# this is just 1
training_img = training_img.reshape(60_000, 28, 28, 1)
test_img = test_img.reshape(10_000, 28, 28, 1)

# normalize per usual
training_img = training_img / 255.0
test_img = test_img / 255.0

# CONVOLUTION LAYER
# we define the number of convolutions we want the model to learn as 64;
# (3, 3) is the filter size, i.e. the number of pixels that will be part
# of the convolution; we define the input shape initially, and the
# activation function per usual; we want the model to learn what the best
# filter values are for these convolutions

# MAX POOLING LAYER
# this groups ('pools') 2x2 sets of pixels together and takes the max, this
# has the affect of reducing the number of pixels by 4

cnn_model = CnnModel()
cnn_model.model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.SparseCategoricalCrossentropy(),
              metrics='accuracy')

cnn_model.model.fit(training_img, training_lbl, epochs=50)

cnn_model.model.evaluate(test_img, test_lbl)

classifications = cnn_model.model.predict(test_img)
print(classifications[0])
print(test_lbl[0])
