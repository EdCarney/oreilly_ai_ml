import tensorflow as tf
from cnn import HorseHumanCnnModel
from get_data import TRAINING_DIR, VALIDATION_DIR


# if we want to generate our data from our own local sources, we can use an
# image data generator; this will allow us to have data in a format like
#   data_dir
#   |---- training
#   |--------  category_1
#   |--------  category_2
#   |--------  category_3
#   |---- validation
#   |--------  category_1
#   |--------  category_2
#   |--------  category_3

# note that the validation data is meant to be used DURING the training as
# an indicator for how well the model is doing; this is different from
# training data that is meant to be used only AFTER the model is trained

# we want the image pixel values to be normalized b/w 0 and 1
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        directory=TRAINING_DIR,
        target_size=(300, 300),
        class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(
        directory=VALIDATION_DIR,
        target_size=(300, 300),
        class_mode='binary')

cnn_model = HorseHumanCnnModel()
model: tf.keras.Sequential = cnn_model.model
model.compile(
        loss='binary_crossentropy',
        optimizer=tf.optimizers.RMSprop(learning_rate=0.001),
        metrics=['accuracy'])

history = model.fit_generator(
        train_generator,
        epochs=15,
        validation_data=validation_generator)
