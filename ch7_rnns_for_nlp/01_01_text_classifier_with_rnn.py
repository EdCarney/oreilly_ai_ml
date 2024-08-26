import numpy as np
import tensorflow as tf
from sarcasm_data_processing import get_sarcasm_sequences_and_labels


# note that we lose a lot of context by just looking at the individual words;
# that is basically what we are doing with the tokenization + embedding method
#
# consider that the word 'blue' can have vastly different meanings depending on
# other words earlier or later in the sentence (e.g. "I'm feeling blue" vs. "A
# very pretty blue sky today"); we need to be able to carry over context from
# other words
#
# this is where RNNs come into play; these are essentially a function that at
# each step takes in some value AND the value from the previous iteration, and
# outputs a new value; for us, this allows some carry over of values from words
# earlier in the sentence
#
# however, this basic functionality still means that words lose impact on other
# words the further apart they are (the first word in the sentence has almost no
# impact on the last word), but this is not necessarily true for speech; so to
# ensure we have some memory of these values, Long Short-Term Memory (LSTM)
# models were introduced, which basically maintain a longer state memory that
# better models this impact
#
# finally, we run this process both ways (forwards and backwards) through the
# sentence, since the impact on the meaning can go both ways

VOCAB_SIZE = 20_000
EMBEDDING_DIM = 64
DENSE_LAYER_NEURONS = 24
EPOCHS = 30

(train_seqs, train_lbls), (test_seqs, test_lbls) = get_sarcasm_sequences_and_labels(
    VOCAB_SIZE
)

train_seqs = np.array(train_seqs)
train_lbls = np.array(train_lbls)
test_seqs = np.array(test_seqs)
test_lbls = np.array(test_lbls)

# rule of thumb for # nuerons after the embedding layer was the 4th root of the
# vocab size; but for an RNN layer that would be too small, so instead we have
# the same size as the embedding dimension

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(EMBEDDING_DIM)),
        tf.keras.layers.Dense(DENSE_LAYER_NEURONS, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# note that this model is greatly improved by creating your own optimizer with
# a reduced learning rate; otherwise the model quickly overfits to the data in
# this case

model.compile(
    optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=["acc"]
)

print(model.summary())

model.fit(train_seqs, train_lbls, epochs=EPOCHS, validation_data=(test_seqs, test_lbls))
model.evaluate(test_seqs, test_lbls)
