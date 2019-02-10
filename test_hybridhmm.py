import numpy as np
import tensorflow as tf

from hybridhmm import HybridHMM

posterior = lambda x: x
A_init = np.array([
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.],
    [1., 0., 0., 0.]])
hmm = HybridHMM(posterior, A_init)

pseudo_lkh = tf.placeholder('float32', [3, 4], 'pseudo_lkh')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    alpha = hmm.forward(pseudo_lkh)
    beta = hmm.backward(pseudo_lkh)

    alpha_value, beta_value = sess.run(
        [alpha, beta],
        feed_dict={
            pseudo_lkh: np.array([
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [.0, 0., 1., 0.], ])
        })

    print(alpha_value)
    print(beta_value)


posterior = lambda x: x
A_init = np.array([
    [.9, .1, 0., 0.],
    [0., .9, .1, 0.],
    [0., 0., 1., 0.],
    [1., 0., 0., 0.]])
hmm = HybridHMM(posterior, A_init)

pseudo_lkh_value = np.array([
    [1., 0., 0., 0.],
    [1., 0., 0., 0.],
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 1., 0.],
    [0., 0., 1., 0.]])
pseudo_lkh = tf.placeholder('float32', [len(pseudo_lkh_value), 4], 'pseudo_lkh')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    path = hmm.viterbi(pseudo_lkh)

    path_value = sess.run(
        path,
        feed_dict={pseudo_lkh: pseudo_lkh_value})

    print(path_value)

posterior = lambda x: x

A_init = np.array([
    [.9, .1, 0., 0.],
    [0., .9, .1, 0.],
    [0., 0., 1., 0.],
    [9., .1, 0., 0.]])

pseudo_lkh_value = np.array([
    [1., 0., 0., 0.],
    [1., 0., 0., 0.],
    [1., 0., 0., 0.],
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 1., 0.],
    [0., 0., 1., 0.]])

hmm = HybridHMM(posterior, A_init)

pseudo_lkh = tf.placeholder('float32', [len(pseudo_lkh_value), 4], 'pseudo_lkh')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    sess.run(hmm.accumulate_updates(pseudo_lkh),
             feed_dict={pseudo_lkh: pseudo_lkh_value})

    sess.run(hmm.update_transitions())

    print(sess.run(hmm.A))