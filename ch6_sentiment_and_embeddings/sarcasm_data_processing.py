import json
import string
import tensorflow as tf
from bs4 import BeautifulSoup
from constants import STOPWORDS


DATA_FILE = "Sarcasm_Headlines_Dataset_v2.json"
TRAINING_SIZE = 23_000
VOCAB_SIZE = 20_000
OOV_TOKEN = "<OOV>"
MAX_LENGTH = 10
PAD_TYPE = "post"
TRUNC_TYPE = "post"


def _get_sarcasm_data():
    with open(DATA_FILE, mode='r') as f:
        dataset = []
        for line in f.readlines():
            dataset.append(json.loads(line))

    sentences = []
    labels = []
    urls = []

    table = str.maketrans('', '', string.punctuation)

    for line in dataset:
        sentence: str = line['headline'].lower()

        # separate punctuation and words
        sentence = sentence.replace('.', ' . ')
        sentence = sentence.replace(',', ' , ')
        sentence = sentence.replace('-', ' - ')
        sentence = sentence.replace('/', ' / ')

        # remove any HTML tags
        sentence = BeautifulSoup(sentence, features='html.parser').get_text()

        # remove stop words
        filtered_sentence = ""
        for word in sentence.split():
            word = word.translate(table)

            if word not in STOPWORDS:
                filtered_sentence += word + " "

        sentences.append(filtered_sentence)
        labels.append(line['is_sarcastic'])
        urls.append(line['article_link'])

    return (sentences, labels, urls)


def _get_sarcasm_training_testing_splits():
    sentences, labels, urls = _get_sarcasm_data()

    training_sentences = sentences[:TRAINING_SIZE]
    testing_sentences = sentences[TRAINING_SIZE:]
    training_labels = labels[:TRAINING_SIZE]
    testing_labels = labels[TRAINING_SIZE:]

    return (
            (training_sentences, training_labels),
            (testing_sentences, testing_labels)
            )


def get_sarcasm_sequences_and_labels(vocab_size=VOCAB_SIZE):
    (training_sentences, training_labels), (testing_sentences, testing_labels)\
            = _get_sarcasm_training_testing_splits()

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=vocab_size,
            oov_token=OOV_TOKEN)

    tokenizer.fit_on_texts(training_sentences)

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences=training_sequences,
            maxlen=MAX_LENGTH,
            padding=PAD_TYPE,
            truncating=TRUNC_TYPE)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences=testing_sequences,
            maxlen=MAX_LENGTH,
            padding=PAD_TYPE,
            truncating=TRUNC_TYPE)

    return (
            (training_sequences, training_labels),
            (testing_sequences, testing_labels)
            )
