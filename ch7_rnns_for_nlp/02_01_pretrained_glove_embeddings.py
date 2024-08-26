import json
import numpy as np
import tensorflow as tf
from sarcasm_data_processing import get_sarcasm_sequences_and_labels
from process_glove_data import get_embedding_matrix


# now, we will do basically the same operation, but we will used pretrained
# embeddings (i.e. transfer learning) to get the weights for our embedding
# layer
#
# recall that the number of tokens in a dataset is NOT necessarily the number
# of words (although it was for our embedding); a single word may be comprised
# of multiple tokens; this is why some datasets have a token count much higher
# than their vocab size
#
# this data is broken into tokens with the embedding weights, we still need to
# tokenize our sarcasm data, but now we just iterate through the pretrained
# data and get the embedding weights that already exist for us, we then supply
# these to the embedding layer and set is as not trainable


VOCAB_SIZE = 14_000
EMBEDDING_DIM = 50
DENSE_LAYER_NEURONS = 24
EPOCHS = 30
PRETRAINED_EMBEDDING_FILE = "./glove.twitter.27B/glove.twitter.27B.50d.txt"
HISTORY_OUT_FILE = "02_02_history.json"

# ------------ LOGIC ------------ #

(train_seqs, train_lbls), (test_seqs, test_lbls) = get_sarcasm_sequences_and_labels(
    VOCAB_SIZE
)

embedding_mat = get_embedding_matrix(PRETRAINED_EMBEDDING_FILE, VOCAB_SIZE)

train_seqs = np.array(train_seqs)
train_lbls = np.array(train_lbls)
test_seqs = np.array(test_seqs)
test_lbls = np.array(test_lbls)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Embedding(
            VOCAB_SIZE, EMBEDDING_DIM, weights=[embedding_mat], trainable=False
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(EMBEDDING_DIM, return_sequences=True)
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(EMBEDDING_DIM)),
        tf.keras.layers.Dense(DENSE_LAYER_NEURONS, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

adam = tf.keras.optimizers.Adam(
    learning_rate=0.00001, beta_1=0.9, beta_2=0.999, amsgrad=False
)
model.compile(
    optimizer=adam, loss=tf.keras.losses.BinaryCrossentropy(), metrics=["acc"]
)

print(model.summary())

history = model.fit(
    train_seqs, train_lbls, epochs=EPOCHS, validation_data=(test_seqs, test_lbls)
)

with open(HISTORY_OUT_FILE, "w") as f:
    json.dump(history.history, f)

model.evaluate(test_seqs, test_lbls)
