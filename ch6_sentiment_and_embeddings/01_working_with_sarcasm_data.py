import numpy as np
import tensorflow as tf
from sarcasm_data_processing import get_sarcasm_sequences_and_labels, VOCAB_SIZE


# load the data from our pre-processing module

(train_seqs, train_lbls), (test_seqs, test_lbls) = get_sarcasm_sequences_and_labels()

# convert to numpy arrays to put them in a format tensorflow
# can work with

train_seqs = np.array(train_seqs)
train_lbls = np.array(train_lbls)
test_seqs = np.array(test_seqs)
test_lbls = np.array(test_lbls)

# define the model, for NLP we have the concept of 'embeddings' that
# define a higher-dimension vector representation of a word given the
# training data; the goal here is to determine whether a headline is
# sarcastic or not, so we basically get this vector representation
# (here it is in 16 dimensions) for each word and then sum across the
# words to get a single 16-dimension vector representation for the headline
# (this is what the global average pooling does)
#
# we then feed this representation into a dense NN and eventually to a NN
# of one node to give us the final indication of sarcastic or not

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

model.fit(train_seqs, train_lbls, epochs=30)
model.evaluate(test_seqs, test_lbls)
