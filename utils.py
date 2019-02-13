from functools import wraps
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors


def method_scope(name=None):
    def wrap(f):
        @wraps(f)
        def wrapped(obj, *args, **kwargs):
            with tf.variable_scope(obj.scope, auxiliary_name_scope=False) as scope:
                with tf.name_scope(scope.original_name_scope):
                    with tf.name_scope(name or f.__name__):
                        return f(obj, *args, **kwargs)

        return wrapped

    return wrap


def plot_transition_matrix(mat):
    plt.imshow(mat,
               clim=(0, 1),
               norm=plt_colors.PowerNorm(gamma=1./2.))
    plt.colorbar()
    plt.title("state transition matrix")


def clipped_log(a):
    non_zero = tf.greater(a, 1e-7)
    return tf.where(non_zero, tf.log(a), tf.ones_like(a) * -1000)


def logaddexp(x, y):
    # from https://github.com/tensorflow/tensorflow/issues/3682#issue-169774585
    temp = x - y
    return tf.where(temp > 0.0,
                    x + tf.log1p(tf.exp(-temp)),
                    y + tf.log1p(tf.exp(temp)))
