import numpy as np
import tensorflow as tf
from sarcasm_data_processing import get_sarcasm_sequences_and_labels

# a dimension size of 16 was arbitrary; as a rule of thumb, the dimensionality
# of the vector space should be the fourth-root of the vocab size; so for 2000
# we are looking at a size of about 7
#
# we will see that the results are similar to the previous training, but the
# loss for the validation set stabilizes quickly and allows for faster, more
# regular training

VOCAB_SIZE = 2_000

(train_seqs, train_lbls), (test_seqs, test_lbls) =\
        get_sarcasm_sequences_and_labels(VOCAB_SIZE)

train_seqs = np.array(train_seqs)
train_lbls = np.array(train_lbls)
test_seqs = np.array(test_seqs)
test_lbls = np.array(test_lbls)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 7),
    tf.keras.layers.GlobalAvgPool1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(), metrics=['acc'])

print(model.summary())

model.fit(train_seqs, train_lbls, epochs=100,
          validation_data=(test_seqs, test_lbls))
model.evaluate(test_seqs, test_lbls)
