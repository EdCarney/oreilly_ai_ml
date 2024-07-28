import tensorflow_datasets as tfds

# note that some datasets do not have pre-builtin splits for test, training,
# and validation, but we can still establish this ourselves

# the first 80% of the data
train = tfds.load('cats_vs_dogs', split='train[:80%]', as_supervised=True)

# the middle 80-90% of the data
validation = tfds.load('cats_vs_dogs', split='train[80%:90%]', as_supervised=True)

# the last 10% of the data
test = tfds.load('cats_vs_dogs', split='train[-10%:]', as_supervised=True)

# enumerate returns a set of (index, element), then we take the last element
# and add one
train_count = [i for i, _ in enumerate(train)][-1] + 1
print("Training data has this many elements:", train_count)
