import os
import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

a = np.random.randn(3, 3)
b = np.random.randn(3, 1)
c = a * b

print(c)