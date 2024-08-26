import tensorflow as tf
import tensorflow_datasets as tfds
import multiprocessing as mp

data, info = tfds.load("cats_vs_dogs", split="train", with_info=True)

print(info)

# this parallelizes the loading of the data; the cycle length indicates the number of elements
# processed concurrently, and the parallel calls is how many CPUs are used; note that this is
# almost exclusively a CPU process

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            16, (3, 3), activation="relu", input_shape=(300, 300, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])

file_pattern = "C:\\Users\\carne\\tensorflow_datasets\\cats_vs_dogs\\4.0.0\\cats_vs_dogs-train.tfrecord*"
files = tf.data.Dataset.list_files(file_pattern=file_pattern)

train_dataset = files.interleave(
    tf.data.TFRecordDataset,
    cycle_length=4,
    num_parallel_calls=tf.data.experimental.AUTOTUNE,
)


def read_tfrecord(serialized_example):
    feature_description = {
        "image": tf.io.FixedLenFeature((), tf.string, ""),
        "label": tf.io.FixedLenFeature((), tf.int64, -1),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_jpeg(example["image"], channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255
    image = tf.image.resize(image, (300, 300))
    return image, example["label"]


cores = mp.cpu_count()
print("Cores available:", cores)

train_dataset = train_dataset.map(read_tfrecord, num_parallel_calls=cores)
train_dataset = train_dataset.cache()

train_dataset = train_dataset.shuffle(1024).batch(32)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

model.fit(train_dataset, epochs=10, verbose=1)
