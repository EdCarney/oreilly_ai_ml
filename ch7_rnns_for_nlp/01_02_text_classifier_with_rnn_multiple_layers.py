import json
import numpy as np
import tensorflow as tf
from sarcasm_data_processing import get_sarcasm_sequences_and_labels


# this model will be similar, except we will stack RNN layers on top of
# each other and implement dropout and reduced learning rate in the
# optimizer to help fight against overfitting

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

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(EMBEDDING_DIM, dropout=0.2, return_sequences=True)
        ),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(EMBEDDING_DIM, dropout=0.2)),
        tf.keras.layers.Dense(DENSE_LAYER_NEURONS, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

adam = tf.keras.optimizers.Adam(
    learning_rate=0.000008, beta_1=0.9, beta_2=0.999, amsgrad=False
)
model.compile(
    optimizer=adam, loss=tf.keras.losses.BinaryCrossentropy(), metrics=["acc"]
)

print(model.summary())

history = model.fit(
    train_seqs, train_lbls, epochs=EPOCHS, validation_data=(test_seqs, test_lbls)
)

with open("01_02_history.json", "w") as f:
    json.dump(history.history, f)

model.evaluate(test_seqs, test_lbls)
