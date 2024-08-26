import tensorflow as tf
import tensorflow_datasets as tfds

# print info to see where the data is downloaded to locally
data, info = tfds.load("mnist", with_info=True)
print(info)

# load one of the sharded datasets manually
filename = "C:\\Users\\carne\\tensorflow_datasets\\mnist\\3.0.1\\mnist-test.tfrecord-00000-of-00001"

raw_dataset = tf.data.TFRecordDataset(filename)

for raw_record in raw_dataset.take(1):
    print(repr(raw_record))

# create a feature description
feature_desc = {
    "image": tf.io.FixedLenFeature([], dtype=tf.string),
    "label": tf.io.FixedLenFeature([], dtype=tf.int64),
}


# this parses the provided data into a feature dictionary that we can
# then use later
def _parse_function(example_proto):
    # parse the input proto using the feature description dictionary
    return tf.io.parse_single_example(example_proto, feature_desc)


# apply the parsing function to every element of the dataset
parsed_dataset = raw_dataset.map(_parse_function)
for parsed_record in parsed_dataset.take(1):
    print("\n", parsed_record)
