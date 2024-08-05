import numpy as np
import tensorflow as tf
from sarcasm_data_processing import get_sarcasm_sequences_and_labels


(train_seqs, train_lbls), (test_seqs, test_lbls) =\
        get_sarcasm_sequences_and_labels(10_000)

train_seqs = np.array(train_seqs)
train_lbls = np.array(train_lbls)
test_seqs = np.array(test_seqs)
test_lbls = np.array(test_lbls)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(10_000, 16),
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
