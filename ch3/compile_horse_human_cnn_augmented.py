import tensorflow as tf
from cnn import HorseHumanCnnModel
from get_data import TRAINING_DIR, VALIDATION_DIR


# this is similar to the non-augmented version of the horse-huaman CNN,
# but this model also uses IMAGE AUGMENTATION; this process uses the image
# data generators to preprocess the data by randomly applying zooms, shifts,
# skews, rotations, etc.; this makes the data less uniform and can possibly
# avoid some overfitting

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

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
