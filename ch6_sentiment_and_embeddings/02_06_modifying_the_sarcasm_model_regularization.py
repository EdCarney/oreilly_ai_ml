import numpy as np
import tensorflow as tf
from sarcasm_data_processing import get_sarcasm_sequences_and_labels

# note that a dropout layer is not very effective here due to the samll
# number of neurons in our hidden dense layer; but, regualrization might
# be; this takes the weights on the neurons in the layer and can either
# reduce the importance of weights very close to zero (L1 / LASSO),
# increase the differences b/w wieght values (L2), or both (elastic)
#
# this effect of regularization here is still small, but slightly positive

VOCAB_SIZE = 2_000
EMBEDDING_DIMENSIONAITY = 7
DENSE_LAYER_NEURONS = 8

(train_seqs, train_lbls), (test_seqs, test_lbls) =\
        get_sarcasm_sequences_and_labels(VOCAB_SIZE)

train_seqs = np.array(train_seqs)
train_lbls = np.array(train_lbls)
test_seqs = np.array(test_seqs)
test_lbls = np.array(test_lbls)

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIMENSIONAITY),
    tf.keras.layers.GlobalAvgPool1D(),
    tf.keras.layers.Dense(DENSE_LAYER_NEURONS, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    ])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(), metrics=['acc'])

print(model.summary())

model.fit(train_seqs, train_lbls, epochs=100,
          validation_data=(test_seqs, test_lbls))
model.evaluate(test_seqs, test_lbls)
