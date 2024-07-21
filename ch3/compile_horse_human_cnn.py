import tensorflow as tf
from cnn import HorseHumanCnnModel
from get_data import TRAINING_DIR, VALIDATION_DIR


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
