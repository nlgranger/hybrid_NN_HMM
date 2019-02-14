import tensorflow as tf


class SkelFeatureExtractor:
    def __init__(self, output_size):
        with tf.variable_scope("SkelFeatureExtractor") as scope:
            self.dense1 = tf.layers.Dense(16, use_bias=False)
            self.bn1 = tf.layers.BatchNormalization()
            self.dense2 = tf.layers.Dense(output_size)

            self.scope = scope

    def __call__(self, *args, **kwargs):
        with tf.variable_scope(self.scope, auxiliary_name_scope=False) as scope:
            with tf.name_scope(scope.original_name_scope):
                return self.call(*args, **kwargs)

    def call(self, x, trainable=True):
        out = self.dense1(x)
        out = self.bn1(out, trainable)
        out = tf.nn.relu(out)
        out = self.dense2(out)

        return out

    @property
    def trainable_variables(self):
        return (self.dense1.trainable_variables
                + self.dense2.trainable_variables)
