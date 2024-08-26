import numpy as np
import tensorflow as tf
from sarcasm_data_processing import get_sarcasm_sequences_and_labels


(train_seqs, train_lbls), (test_seqs, test_lbls) = get_sarcasm_sequences_and_labels(
    10_000
)

train_seqs = np.array(train_seqs)
train_lbls = np.array(train_lbls)
test_seqs = np.array(test_seqs)
test_lbls = np.array(test_lbls)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Embedding(10_000, 16),
        tf.keras.layers.GlobalAvgPool1D(),
        tf.keras.layers.Dense(24, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# note that without augmentation the loss values on the validation data
# starts increasing rapidly; this can partly be due to overfitting of the
# model to noise that is specific to the training data
#
# one way to combat this is to reduce the learning rate of the optimizer;
# typically for ADAM this is 0.001, we will reduce by on order of mag. to
# 0.0001 and also increase the number of epochs

adam = tf.keras.optimizers.Adam(
    learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False
)

model.compile(
    optimizer=adam, loss=tf.keras.losses.BinaryCrossentropy(), metrics=["acc"]
)

print(model.summary())

model.fit(train_seqs, train_lbls, epochs=100, validation_data=(test_seqs, test_lbls))
model.evaluate(test_seqs, test_lbls)
