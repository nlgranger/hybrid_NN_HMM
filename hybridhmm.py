import numpy as np
import tensorflow as tf

from utils import method_scope, clipped_log, logaddexp


class HybridHMMTransitions:
    def __init__(self, transitions, init_state_priors, state_priors, mask=None):
        n_states = len(transitions)

        if not isinstance(transitions, tf.Tensor):
            transitions = tf.constant(transitions)
        if not isinstance(init_state_priors, tf.Tensor):
            init_state_priors = tf.constant(init_state_priors)
        if not isinstance(state_priors, tf.Tensor):
            state_priors = tf.constant(state_priors)
        if mask is None:
            mask = tf.greater(transitions, 0)
        elif not isinstance(mask, tf.Tensor):
            mask = tf.constant(mask, dtype='bool')

        with tf.variable_scope(self.__class__.__name__) as scope:
            self.n_states = n_states

            # Parameters
            self.A = tf.Variable(
                clipped_log(transitions),
                dtype='float32', name="A")
            self.init_state_priors = tf.Variable(
                clipped_log(init_state_priors),
                dtype='float32', name="init_state_priors")
            self.state_priors = tf.Variable(
                clipped_log(state_priors),
                dtype='float32', name="state_priors")
            self.mask = mask

            # Accumulated statistics
            self.transition_stats = tf.Variable(
                np.ones([n_states, n_states]) * -1000,
                dtype='float32', name="transition_stats")
            self.init_state_stats = tf.Variable(
                np.ones([n_states]) * -1000,
                dtype='float32', name="transition_stats")
            self.state_counts = tf.Variable(
                np.zeros([n_states]),
                dtype='int64', name="state_counts")

            self.scope = scope

    @method_scope("pseudo_lkh")
    def pseudo_lkh(self, posteriors):
        return posteriors - tf.expand_dims(self.state_priors, 0)

    @method_scope("forward")
    def forward(self, pseudo_lkh):
        """Compute α terms, including virtual α_0"""
        def forward_step(alpha_t, pseudo_lkh_tm1):
            return tf.reduce_logsumexp(
                tf.expand_dims(alpha_t, 1) + self.A, axis=0) \
                + pseudo_lkh_tm1

        alpha_1 = self.init_state_priors + pseudo_lkh[0] - self.state_priors
        alpha_2_T = tf.scan(
            forward_step,
            pseudo_lkh[1:],
            initializer=alpha_1,
            back_prop=False)
        alpha = tf.concat([tf.expand_dims(alpha_1, 0), alpha_2_T], axis=0)

        return alpha

    @method_scope("backward")
    def backward(self, pseudo_lkh):
        """Compute β terms"""
        def backward_step(beta_tp1, pseudo_lkh_tp1):
            return tf.reduce_logsumexp(
                self.A + tf.expand_dims(pseudo_lkh_tp1 + beta_tp1, 0),
                axis=1)

        beta_T = tf.zeros([self.n_states])
        beta_1_Tm1 = tf.scan(
            backward_step,
            pseudo_lkh[1:],
            reverse=True,
            initializer=beta_T,
            back_prop=False)
        beta = tf.concat([beta_1_Tm1, tf.expand_dims(beta_T, 0)], axis=0)

        return beta

    def viterbi_step(self, rec_term, pseudo_lkh):
        path_proba, path_ptr = rec_term
        probs = self.A + tf.expand_dims(path_proba, 1) + tf.expand_dims(pseudo_lkh, 0)
        path_ptr = tf.argmax(probs, axis=0)
        probs = tf.reduce_max(probs, axis=0)
        return probs, path_ptr

    @method_scope("viterbi")
    def viterbi(self, pseudo_lkh):
        path_proba_1 = self.init_state_priors + pseudo_lkh[0]
        path_ptr_1 = tf.ones([self.n_states], dtype='int64') * -1

        path_proba, path_ptr = tf.scan(
            self.viterbi_step,
            pseudo_lkh,
            initializer=(path_proba_1, path_ptr_1),
            back_prop=False)

        reverse_path_T = tf.argmax(path_proba[-1])

        path = tf.scan(
            lambda i, ptrs: ptrs[i],
            path_ptr[1:],
            initializer=reverse_path_T,
            reverse=True,
            back_prop=False)

        return tf.concat([path, tf.expand_dims(reverse_path_T, 0)], axis=0)

    @method_scope("add_transition_stats")
    def add_transition_stats(self, pseudo_lkh):
        alpha = self.forward(pseudo_lkh)
        beta = self.backward(pseudo_lkh)

        qty = tf.expand_dims(alpha[:-1], 2) \
            + tf.expand_dims(self.A, 0) \
            + tf.expand_dims(pseudo_lkh[1:], 1) \
            - tf.expand_dims(tf.expand_dims(self.state_priors, 0), 0) \
            + tf.expand_dims(beta[1:], 1)
        qty = tf.maximum(qty, -1000)
        qty = qty - tf.reduce_logsumexp(qty, axis=[1, 2], keepdims=True)
        qty = tf.reduce_logsumexp(qty, axis=0)
        transition_stats = logaddexp(self.transition_stats, qty)

        qty = alpha[0] + beta[0]
        qty = qty - tf.reduce_logsumexp(qty)
        init_state_stats = logaddexp(self.init_state_stats, qty)

        return tf.group(
            tf.assign(self.transition_stats, transition_stats),
            tf.assign(self.init_state_stats, init_state_stats))

    @method_scope("update_transitions")
    def update_transitions(self):
        A = self.transition_stats \
            - tf.reduce_logsumexp(self.transition_stats, axis=1, keepdims=True)
        A = tf.where(self.mask, A, -1000 * tf.ones_like(A))
        init_state_priors = self.init_state_stats \
            - tf.reduce_logsumexp(self.init_state_stats)

        with tf.control_dependencies([A, init_state_priors]):
            return tf.group(
                tf.assign(self.A, A),
                tf.assign(self.transition_stats, tf.ones_like(self.transition_stats) * -1000),
                tf.assign(self.init_state_priors, init_state_priors),
                tf.assign(self.init_state_stats, tf.ones_like(self.init_state_stats) * -1000))

    @method_scope("add_priors_stats")
    def add_priors_stats(self, pseudo_lkh):
        """Align states on given sequence and update their frequencies."""
        alignment = self.viterbi(pseudo_lkh)
        matches = tf.equal(tf.expand_dims(alignment, 1),
                           tf.expand_dims(tf.range(self.n_states, dtype='int64'), 0))
        counts = tf.reduce_sum(tf.cast(matches, 'int64'), axis=0)

        return tf.assign_add(self.state_counts, counts)

    @method_scope("update_state_priors")
    def update_state_priors(self):
        state_counts = tf.cast(self.state_counts, 'float32')
        total_count = tf.cast(tf.reduce_sum(self.state_counts), 'float32')
        state_priors = tf.log(state_counts) - tf.log(total_count)

        with tf.control_dependencies([state_priors]):
            return tf.group(
                tf.assign(self.state_priors, state_priors),
                tf.assign(self.state_counts, tf.zeros_like(self.state_counts)))


def heuristic_priors_adjustment(hmm, count_states, sess):
    """Heuristic to find priors which ensures all states are visited."""
    old_priors = sess.run(hmm.state_priors)
    priors = np.exp(old_priors)

    for i in range(50):
        counts = count_states()

        # terminate when less than a fector 5 between state frequencies
        if not any(counts < np.max(counts) / 5):
            break

        # update slowly
        priors = .05 * counts / np.sum(counts) + .95 * priors
        sess.run(tf.assign(hmm.state_priors,
                           tf.constant(np.log(priors), dtype='float32')))

    # clear statistics
    sess.run(tf.assign(hmm.state_counts, tf.zeros_like(hmm.state_counts)))

    return old_priors, priors
