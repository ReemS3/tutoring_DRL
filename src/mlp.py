
 
import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer


class MLP(tf.keras.Model):
    def __init__(self, obs_dim):
        super(MLP, self).__init__()
        self.input_layer = InputLayer(input_shape=[obs_dim])
        self.layers_ls = [
            Dense(16, input_shape=[obs_dim], activation="relu"),
            Dense(8, activation="relu"),
            Dense(4, input_shape=[8], activation="softmax"),
        ]
        self.optimizer = tf.optimizers.Adam(learning_rate=0.01)

    @tf.function
    def call(self, input):
        output = self.input_layer(input)
        for layer in self.layers_ls:
            output = layer(output)
        return output
