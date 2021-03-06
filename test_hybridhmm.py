import numpy as np
import tensorflow as tf
from hybridhmm import HybridHMMTransitions


def clipped_log(a):
    a = np.asarray(a)
    non_zero = a > 1e-7
    return np.where(non_zero, np.log(a, where=non_zero), - 1000)


def test_forwward_backward():
    A = [[0., 1., 0.],
         [0., 0., 1.],
         [0., 0., 0.]]
    init_state_priors = [1., 0., 0.]
    state_priors = [1., 1., 1.]  # sum != 1 to simplify values
    hmm = HybridHMMTransitions(A, init_state_priors, state_priors)

    pseudo_lkh = tf.placeholder('float32', [3, 3], 'pseudo_lkh')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        alpha = hmm.forward(pseudo_lkh)
        beta = hmm.backward(pseudo_lkh)

        alpha_value, beta_value = sess.run(
            [alpha, beta],
            feed_dict={
                pseudo_lkh: clipped_log([
                    [1., 0., 0.],
                    [0., 1., 0.],
                    [.0, 0., 1.]])
            })

    np.testing.assert_array_equal(
        np.exp(alpha_value),
        [[1., 0., 0.],
         [0., 1., 0.],
         [0., 0., 1.]])
    np.testing.assert_array_equal(
        np.exp(beta_value),
        [[1., 0., 0.],
         [0., 1., 0.],
         [1., 1., 1.]])


def test_virterbi():
    A = [[.9, .1, 0.],
         [0., .9, .1],
         [0., 0., 1.]]
    init_state_priors = [1., 0., 0.]
    state_priors = [1., 1., 1.]  # sum != 1 to simplify values
    hmm = HybridHMMTransitions(A, init_state_priors, state_priors)

    pseudo_lkh_value = clipped_log([
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.]])
    pseudo_lkh = tf.placeholder('float32', [len(pseudo_lkh_value), 3], 'pseudo_lkh')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ml_state_path = hmm.viterbi(pseudo_lkh)

        ml_state_path_value = sess.run(
            ml_state_path,
            feed_dict={pseudo_lkh: pseudo_lkh_value})

    np.testing.assert_array_equal(
        ml_state_path_value,
        [0, 0, 0, 1, 1, 2, 2, 2])


def test_transition_learning():
    A = [[.9, .1, 0.],
         [0., .9, .1],
         [0., 0., 1.]]
    init_state_priors = [1., 0., 0.]
    state_priors = [1., 1., 1.]  # sum != 1 to simplify values
    hmm = HybridHMMTransitions(A, init_state_priors, state_priors)

    pseudo_lkh_value = clipped_log([
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 0., 1.],
        [0., 0., 1.]])

    pseudo_lkh = tf.placeholder('float32', [len(pseudo_lkh_value), 3], 'pseudo_lkh')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        sess.run(hmm.add_transition_stats(pseudo_lkh),
                 feed_dict={pseudo_lkh: pseudo_lkh_value})

        # print(sess.run(hmm.transition_stats))
        sess.run(hmm.update_transitions())

        init_state_priors, A = sess.run([hmm.init_state_priors, hmm.A])

        np.testing.assert_almost_equal(np.exp(init_state_priors), [1., 0., 0.])
        np.testing.assert_almost_equal(
            np.exp(A),
            [[0.75, 0.25, 0.00],
             [0.00, 0.50, 0.50],
             [0.00, 0.00, 1.00]])


if __name__ == "__main__":
    test_forwward_backward()
    test_virterbi()
    test_transition_learning()
