import tensorflow as tf
from tensorflow.keras.layers import Layer

class Attention(Layer):
    def call(self, inputs):
        score = tf.nn.softmax(inputs, axis=1)
        context = score * inputs
        return tf.reduce_sum(context, axis=1)
