import numpy as np
import tensorflow as tf


# the idea here is to generate data that we can use to train an ML model for
# make predictions on data sequences; i.e. we would have a series and a label
#
# to do this, we will take our data and break it into windows, where all but
# the final value is the series data, and the final value is the label
#
# we can do this using existing tensorflow tooling to simplify the windowing
# process and data loading


# this creates a sample dataset from 0-9, with windows of size 5 that have an
# overlap of 5-1=4

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(size=5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))

print('Windows:')
for window in dataset:
    print(window.numpy())

# now, we need to take these windows and turn them into series/label data that
# can be used for training

dataset = dataset.map(lambda window: (window[:-1], window[-1:]))

print('\nRaw Data:')
for x, y in dataset:
    print(x.numpy(), y.numpy())

# note that because we are working with tf datasets, we can also easily shuffle
# and batch the data

dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)

print('\nBatched and Shuffled:')
for x, y in dataset:
    print(x.numpy(), y.numpy())
