import tensorflow as tf
import numpy as np
import json


# the idea for this is to train the model to understand what word should
# come next, given a sequence of words; we do this by taking some long form
# text and breaking it into sentences and tokenizing those sentences as we
# did previously
#
# then, we toke the sentence and break it into larger and larger sequences
# of data and label, where the label is the last token in the sequence; we
# then pad the data for input as normal and separate the labels and input
# data


LYRIC_FILE = "lanigans_ball_lyrics.txt"


with open(LYRIC_FILE, mode='r') as f:
    lines = f.readlines()

corpus = [line.lower() for line in lines]

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(corpus)

input_sequences = []
for line in corpus:
    # get the sequence for the full sentence
    token_list = tokenizer.texts_to_sequences([line])[0]

    # break the sentence into larger and larger pieces
    # with the smallest piece having 2 elements
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# we still need to pad the data, so get the longest sequence
# and then pre-pad to that length
max_seq_len = max([len(x) for x in input_sequences])
input_sequences = np.array(tf.keras.utils.pad_sequences(
    input_sequences, maxlen=max_seq_len, padding='pre'))

# now use numpy slice syntax to easily get the trainind data
# (everything up to the last token) and the labels (the last token)
inputs, labels = input_sequences[:, :-1], input_sequences[:, -1]

# note that we need way for the model to tell us which word it predicts will
# be next, to do this we need a neuron for each word in the corpus, and take
# the highest value as the word that the model predicts should be next
#
# to do this, we need to map the labels (which are token values) to an array
# of all zeros, with a one at the token index corresponding to the token
# (i.e. for 100 words if token 37 is the end of the sequence and thus the label
# we need an array that is 99 zeros and 1 one in the 37th index)
#
# we can use to_categorical for this

num_words = len(tokenizer.word_index) + 1
outputs = tf.keras.utils.to_categorical(labels, num_classes=num_words)

print(list(input_sequences[0]))
print(list(inputs[0]))
print(list(outputs[0]))

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=num_words, output_dim=8))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(max_seq_len - 1)))
model.add(tf.keras.layers.Dense(num_words, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['acc'])

history = model.fit(inputs, outputs, epochs=1500, verbose=1)

with open('01_01_history.json', 'w') as f:
    json.dump(history.history, f)
