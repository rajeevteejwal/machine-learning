import os
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# Initialization of Tensors
x = tf.constant(4, shape=(1, 1))
x = tf.ones((1, 5))
x = tf.zeros((4, 2))
x = tf.eye(5)
x = tf.random.normal((4, 4), mean=0, stddev=1.0)
x = tf.random.uniform((1, 3), minval=1, maxval=2)
x = tf.range(start=1, limit=10, delta=3)
x = tf.cast(x, dtype=tf.float32)

# Mathematical Operations
x = tf.constant([1, 2, 3])
y = tf.constant([9, 8, 7])
z = tf.add(x, y)
z = tf.subtract(x, y)
z = tf.multiply(x, y)
z = tf.divide(x, y)

# Indexes
x = tf.range(0, 10)
i = tf.constant([1, 3])
y = tf.gather(x, i)

print(x)
print(y)

