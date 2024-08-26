from typing import List
import tensorflow as tf
import numpy as np
import json

# we will do the same thing as before, except now we will feed some sample
# text to our model and have it predict the next word


def trim_and_buffer(sz: int, buf_char: str, text: str) -> str:
    if len(text) > sz:
        return text[:sz]
    return text + buf_char * (sz - len(text))


def get_words() -> List[str]:
    LYRIC_FILE = "irish_lyrics_eof.txt"
    with open(LYRIC_FILE, mode="r") as f:
        lines = f.readlines()
    return " ".join(lines)


def get_windowed_data(corpus: str, window_sz: int):
    all_words = corpus.split()
    range_sz = len(all_words) - window_sz + 1
    sentences = []
    for i in range(range_sz):
        for j in range(2, window_sz + 1):
            words = all_words[i : i + j]
            sentences.append(" ".join(words))
    return sentences


corpus = get_words()
sentences = get_windowed_data(corpus, 9)

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(sentences)

input_sequences = []
for line in sentences:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[: i + 1]
        input_sequences.append(n_gram_sequence)

max_seq_len = max([len(x) for x in input_sequences])
input_sequences = np.array(
    tf.keras.utils.pad_sequences(input_sequences, maxlen=max_seq_len, padding="pre")
)

inputs, labels = input_sequences[:, :-1], input_sequences[:, -1]

num_words = len(tokenizer.word_index) + 1
outputs = tf.keras.utils.to_categorical(labels, num_classes=num_words)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=num_words, output_dim=16))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(max_seq_len - 1)))
model.add(tf.keras.layers.Dense(num_words, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

history = model.fit(inputs, outputs, epochs=1500, verbose=1)

with open("02_02_history.json", "w") as f:
    json.dump(history.history, f)

# let's define the start of the sentence we want to use
# note that the words you use should ideally be in the corpus

my_texts = [
    "sweet jeremy saw dublin",
    "you know nothing jon snow",
    "I have a friend named farrell",
]
out_texts = []
next_words = 12
for my_text in my_texts:
    print()
    for _ in range(next_words):

        # tokenize the sentence and pad it to be the proper length

        token_list = tokenizer.texts_to_sequences([my_text])
        token_list = tf.keras.utils.pad_sequences(
            token_list, maxlen=max_seq_len - 1, padding="pre"
        )

        predicted = model.predict(token_list, verbose=0)
        max_index = np.argmax(predicted)
        max_value = predicted[0][max_index]

        # get the actual output word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == max_index:
                output_word = word
                break

        print(
            "Confidence:",
            trim_and_buffer(6, "0", str(max_value)),
            "Predicted word:",
            output_word,
        )
        my_text += " " + output_word

for out_text in out_texts:
    print(out_text)
