import tensorflow as tf
import tensorflow_datasets as tfds

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split='train'))

for item in train_data:
    imdb_sentences.append(str(item['text']))

print('first sentence:', imdb_sentences[0])

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(imdb_sentences)

i = 0
for key, value in tokenizer.word_index.items():
    if i < 10:
        i += 1
        print("{ ", key, ", ", value, " }")
    else:
        break
