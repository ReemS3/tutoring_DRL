import tensorflow as tf
import numpy as np

"""To show that the tensors are way larger than a numpy array. """
a = np.zeros(shape=(int(1e12), int(1e12)))
# b = tf.zeros(shape=(int(1e12), int(1e12)))