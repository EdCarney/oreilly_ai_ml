import numpy as np
import tensorflow as tf
from sarcasm_data_processing import get_sarcasm_sequences_and_labels

# we chose a dense layer of 24 in part based on the previous embedding
# dimensionality; with a dimension size of 7, we can reduce this; again
# this simplifies the model without serious impact to accuracy or
# convergence

VOCAB_SIZE = 2_000
EMBEDDING_DIMENSIONAITY = 7
DENSE_LAYER_NEURONS = 8

(train_seqs, train_lbls), (test_seqs, test_lbls) = get_sarcasm_sequences_and_labels(
    VOCAB_SIZE
)

train_seqs = np.array(train_seqs)
train_lbls = np.array(train_lbls)
test_seqs = np.array(test_seqs)
test_lbls = np.array(test_lbls)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIMENSIONAITY),
        tf.keras.layers.GlobalAvgPool1D(),
        tf.keras.layers.Dense(DENSE_LAYER_NEURONS, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=["acc"]
)

print(model.summary())

model.fit(train_seqs, train_lbls, epochs=100, validation_data=(test_seqs, test_lbls))
model.evaluate(test_seqs, test_lbls)
