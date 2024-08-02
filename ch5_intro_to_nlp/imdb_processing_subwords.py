import tensorflow_datasets as tfds

# subword datasets are an in-between of a dataset composed of single characters
# (small tokens, little sematic meaning) and a dataset of individual words
# (many tokens, high sematic meaning)

(train_data, test_data), info = tfds.load(
        "imdb_reviews/subwords32k",
        split=(tfds.Split.TRAIN, tfds.Split.TEST),
        as_supervised=True,
        with_info=True)

# we can access the encoder via the dataset information

encoder: tfds.deprecated.text.TextEncoder = info.features['text'].encoder
print("Vocab size of encoder is {}".format(encoder.vocab_size))

print(encoder.subwords[:15])

sample_str = "Today is a hot day"
encoding = encoder.encode(sample_str)

print("Encoded string: ", encoding)

decoding = encoder.decode(encoding)

print("Decoded string: ", decoding)
