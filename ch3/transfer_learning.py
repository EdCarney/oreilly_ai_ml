import os
import tensorflow as tf
import urllib

from compile_horse_human_cnn import (
        get_horse_human_validation_datagen,
        get_horse_human_train_datagen
        )

weights_url = "https://storage.googleapis.com/mledu-datasets/"
weights_file = "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

if not os.path.exists(weights_file):
    urllib.request.urlretrieve(weights_url + weights_file, weights_file)

# here we use a very large pretrained model that has already been trained
# for computer vision applications; note this uses tens of millions of
# learned parameters, so is too expensive to train ourselves; instead we
# will load a predefined set of weights and then only train a few layers
# that we add at the end to do the job we need them to

pretrained_model = tf.keras.applications.inception_v3.InceptionV3(
        input_shape=(300, 300, 3),
        include_top=False,
        weights=None)
pretrained_model.load_weights(weights_file)

pretrained_model.summary()

for layer in pretrained_model.layers:
    layer.trainable = False

# we chose an output layer that will be easy to modify and work with; in this
# case we chose one with an output of 7x7 images that is relatively easy to
# handle in our layers

last_layer = pretrained_model.get_layer('mixed7')
print('last layer out shape:', last_layer.output_shape)
last_output = last_layer.output


# flatten the output layer for use with a dense layer
x = tf.keras.layers.Flatten()(last_output)

# add a fully connected layer with 1024 nodes and ReLU activation
x = tf.keras.layers.Dense(1024, activation='relu')(x)

# add final sigmoid layer to give us the binary classification
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

# now define the model as the pretrained model followed by our trainable layers
model = tf.keras.Model(pretrained_model.input, x)
model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['acc'])

model.fit_generator(
        generator=get_horse_human_train_datagen(),
        epochs=40,
        validation_data=get_horse_human_validation_datagen())
