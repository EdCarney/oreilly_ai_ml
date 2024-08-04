import string
import tensorflow as tf
import tensorflow_datasets as tfds
from bs4 import BeautifulSoup
from constants import STOPWORDS

# we create a translation table to translate common punctuation and remove
# it from sentences

table = str.maketrans('', '', string.punctuation)

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split='train'))

for item in train_data:
    # get the data in normalized lower-case
    sentence = str(item['text'].decode('UTF-8').lower())

    # use bs4 to remove any HTML related tags
    soup = BeautifulSoup(sentence)
    sentence = soup.get_text()

    # words are often combined with punctuation and would be seen as a single
    # word by the tokenizer; insert spaces to remove punctuation later
    sentence = sentence.replace(',', ' , ')
    sentence = sentence.replace('.', ' . ')
    sentence = sentence.replace('-', ' - ')
    sentence = sentence.replace('/', ' / ')

    # split sentence into words (splits by space by default)
    words = sentence.split()
    filtered_sentence = ""

    for word in words:
        # remove the punctuation
        word = word.translate(table)

        # remove the stopwords
        if word not in STOPWORDS:
            filtered_sentence += word + " "

    imdb_sentences.append(filtered_sentence)

print('first sentence:', imdb_sentences[0])

# now we will see the words with the lowest index (the most common) as words
# that are not stopwords

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(imdb_sentences)

i = 0
for key, value in tokenizer.word_index.items():
    if i < 10:
        i += 1
        print("{ ", key, ", ", value, " }")
    else:
        break
