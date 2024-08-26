from collections import OrderedDict
import tensorflow as tf
import matplotlib as mp
from sarcasm_data_processing import _get_sarcasm_training_testing_splits


def get_tokenizer(vocab_size: int) -> tf.keras.preprocessing.text.Tokenizer:
    (training_sentences, training_labels), (testing_sentences, testing_labels) = (
        _get_sarcasm_training_testing_splits()
    )

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=vocab_size, oov_token="<OOV>"
    )

    tokenizer.fit_on_texts(training_sentences)

    return tokenizer


tknzer_1 = get_tokenizer(20_000)
tknzer_2 = get_tokenizer(2_000)

# word_count is a dict of tuples of (WORD, WORD_COUNT); we want an list ordered
# by the number of times the word appears

wc_1 = tknzer_1.word_counts
wc_2 = tknzer_2.word_counts

newlist_1 = OrderedDict(sorted(wc_1.items(), key=lambda t: t[1], reverse=True))
newlist_2 = OrderedDict(sorted(wc_2.items(), key=lambda t: t[1], reverse=True))

print(newlist_1)
print(newlist_2)
