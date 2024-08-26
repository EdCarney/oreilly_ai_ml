import tensorflow as tf
import tensorflow_datasets as tfds

mnist_data = tfds.load("fashion_mnist")

for item in mnist_data:
    print(item)

print("\nData about mnist fashion training data")
mnist_train, info = tfds.load("fashion_mnist", split="train", with_info=True)
assert isinstance(mnist_train, tf.data.Dataset)
print(type(mnist_train))

print("\n")
print(info)
for item in mnist_train.take(1):
    print(type(item))
    print(item.keys())
    # print(item['image'])
    print(item["label"])
