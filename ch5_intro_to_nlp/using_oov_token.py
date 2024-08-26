import tensorflow as tf


# what happens if you feed the tokenizer sentences with words that are
# not in its corpus? it will simply ignore them
# this can lead to a lot of context being lost and impact any predictions

# we can specify an 'out-of-vocabulary' (OOV) token to help here, which
# will take the palce of any unknown tokens in the sequences; note that
# the token should be something that won't appear elsewhere in your
# training data

training_data = ["it is rainy today", "it is sunny today", "is it raining today?"]
test_data = ["will it snow today?", "it will rain later"]

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=100, oov_token="<OOV>")

tokenizer.fit_on_texts(training_data)

word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(test_data)

# note that the OOV token is now the first in the word index; there can
# still be considerable context lost, but this helps

print(word_index)
print(sequences)
