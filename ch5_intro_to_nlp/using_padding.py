import tensorflow as tf

# you generally want to normalize your data, for NLP that means ensuring
# your tokenized sequences are of the same length; you can do this by
# padding (either pre or post) the sequence arrays with zeros

# recall that the tokenizer does not use zero as a sequence value; this
# is why

# note that this may cause large amounts of zeros if you are dealing with
# one very large sentence relative to all the other data; you can set a
# maxlen to limit the max padded length as well (data is tructed either
# from the front or back)

training_data = [
        'it is rainy today',
        'it is sunny today',
        'it will snow later'
        ]
test_data = [
        'will it snow today?',
        'it will rain later',
        'I hope that it is sunny later today and not rainy'
        ]

tokenizer = tf.keras.preprocessing.text.Tokenizer(
        num_words=100,
        oov_token='<OOV>')
tokenizer.fit_on_texts(training_data)

sequences = tokenizer.texts_to_sequences(test_data)

# note that now we are preprocessing the sequences

padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences=sequences,
        maxlen=8,
        padding='pre',
        truncating='post')

for seq in padded:
    print(seq)
