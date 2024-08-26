import numpy as np
import tensorflow as tf
from sarcasm_data_processing import get_sarcasm_sequences_and_labels

# note that if you plot a histogram of the word count in the corpus
# you can see the vast majority (~80%) of the words occur less than
# 20 times; this can mean a lot of noise in the data; limiting our
# vocab size can help to prevent this

VOCAB_SIZE = 2_000

(train_seqs, train_lbls), (test_seqs, test_lbls) = get_sarcasm_sequences_and_labels(
    VOCAB_SIZE
)

train_seqs = np.array(train_seqs)
train_lbls = np.array(train_lbls)
test_seqs = np.array(test_seqs)
test_lbls = np.array(test_lbls)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Embedding(VOCAB_SIZE, 16),
        tf.keras.layers.GlobalAvgPool1D(),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=["acc"]
)

print(model.summary())

model.fit(train_seqs, train_lbls, epochs=100, validation_data=(test_seqs, test_lbls))
model.evaluate(test_seqs, test_lbls)
