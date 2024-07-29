import tensorflow as tf


# we will use the keras library tokenizer to tokenize a small
# set of sentenences; specifically, keras has a preprocessing
# library that contains tools for preprocessing and tokenizing
# data for ML
#
# to 'tokenize' here means to map a word to a number; note that
# this could be done at the letter level, but is a bit more tricky
# since the same set of letters can be arranged in different ways
# and have opposite meanings ('antigrams'); this is a problem for
# training

sentences = [
        'it is rainy today',
        'it is sunny today',
        'is it raining today?'
        ]

# we initialize the tokenizer and specify the max number of words
# allowed in its corpus

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)

# this will print out the word-to-number mapping with the words as
# keys; note that punctuation is automatically ignored by default
# but this can be adjusted by the filters during initialization

print(tokenizer.word_index)
