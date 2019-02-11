import numpy as np
import tensorflow as tf

from utils import method_scope


class HybridHMMTransitions:
    def __init__(self, transitions, init_state_priors, state_priors):
        # add transitions from virtual initial state
        n_states = len(transitions)

        with tf.variable_scope(self.__class__.__name__) as scope:
            self.n_states = n_states

            # Parameters
            self.A = tf.Variable(transitions, dtype='float32', name="A")
            self.init_state_priors = tf.Variable(
                init_state_priors, dtype='float32', name="init_state_priors")
            self.state_priors = tf.Variable(
                state_priors, dtype='float32', name="state_priors")

            # Accumulated statistics
            self.transition_stats = tf.Variable(
                np.zeros([n_states, n_states]), dtype='float32', name="transition_stats")
            self.init_state_stats = tf.Variable(
                np.zeros([n_states]), dtype='float32', name="transition_stats")
            self.state_stats = tf.Variable(
                np.zeros([n_states]), dtype='int64', name="state_stats")

            self.scope = scope

    @method_scope("pseudo_lkh")
    def pseudo_lkh(self, posteriors):
        return posteriors / tf.expand_dims(self.state_priors, 0)

    @method_scope("forward")
    def forward(self, pseudo_lkh):
        """Compute α terms, including virtual α_0"""
        def forward_step(alpha_t, pseudo_lkh_tm1):
            return tf.tensordot(tf.transpose(alpha_t), self.A, 1) * pseudo_lkh_tm1

        alpha_1 = self.init_state_priors * pseudo_lkh[0] / self.state_priors
        alpha = tf.scan(
            forward_step,
            pseudo_lkh[1:],
            initializer=alpha_1,
            back_prop=False)
        return tf.concat([tf.expand_dims(alpha_1, 0), alpha], axis=0)

    @method_scope("backward")
    def backward(self, pseudo_lkh):
        """Compute β terms"""
        def backward_step(beta_tp1, pseudo_lkh_tp1):
            return tf.tensordot(self.A, pseudo_lkh_tp1 * beta_tp1, 1)

        beta_T = tf.ones([self.n_states])
        beta = tf.scan(
            backward_step,
            pseudo_lkh[1:],
            reverse=True,
            initializer=beta_T,
            back_prop=False)
        return tf.concat([beta, tf.expand_dims(beta_T, 0)], axis=0)

    def viterbi_step(self, rec_term, pseudo_lkh):
        path_proba, path_ptr = rec_term
        probs = self.A * tf.expand_dims(path_proba, 1) * tf.expand_dims(pseudo_lkh, 0)
        path_ptr = tf.argmax(probs, axis=0)
        probs = tf.reduce_max(probs, axis=0)
        return probs, path_ptr

    @method_scope("viterbi")
    def viterbi(self, pseudo_lkh):
        path_proba_1 = self.init_state_priors * pseudo_lkh[0] / self.state_priors
        path_ptr_1 = tf.ones([self.n_states], dtype='int64') * -1

        path_proba, path_ptr = tf.scan(
            self.viterbi_step,
            pseudo_lkh,
            initializer=(path_proba_1, path_ptr_1))

        reverse_path_T = tf.argmax(path_proba[-1])
        path = tf.scan(
            lambda i, ptrs: ptrs[i],
            path_ptr[1:],
            initializer=reverse_path_T,
            reverse=True)

        return tf.concat([path, tf.expand_dims(reverse_path_T, 0)], axis=0)

    @method_scope("add_transition_stats")
    def add_transition_stats(self, pseudo_lkh):
        alpha = self.forward(pseudo_lkh)
        beta = self.backward(pseudo_lkh)

        qty = tf.expand_dims(alpha[:-1], 2) \
            * tf.expand_dims(self.A, 0) \
            * tf.expand_dims(pseudo_lkh[1:], 1) \
            / tf.expand_dims(tf.expand_dims(self.state_priors, 0), 0) \
            * tf.expand_dims(beta[1:], 1)

        qty = qty / (tf.reduce_sum(qty, [1, 2], keepdims=True) + 1e-4)
        transition_stats_update = tf.reduce_sum(qty, 0)

        init_state_stats_update = alpha[0] * beta[0]

        return tf.group(
            tf.assign_add(self.transition_stats, transition_stats_update),
            tf.assign_add(self.init_state_stats, init_state_stats_update))

    @method_scope("update_transitions")
    def update_transitions(self):
        A = self.transition_stats \
            / (tf.reduce_sum(self.transition_stats, axis=1, keepdims=True) + 1e-4)
        init_state_priors = self.init_state_stats / tf.reduce_sum(self.init_state_stats)
        with tf.control_dependencies([A, init_state_priors]):
            return tf.group(
                tf.assign(self.A, A),
                tf.assign(self.transition_stats, tf.zeros_like(self.transition_stats)),
                tf.assign(self.init_state_priors, init_state_priors),
                tf.assign(self.init_state_stats, tf.zeros_like(self.init_state_stats)))

    @method_scope("add_priors_stats")
    def add_priors_stats(self, pseudo_lkh):
        """Align states on given sequence and update their frequencies."""
        alignment = self.viterbi(pseudo_lkh)
        matches = tf.equal(tf.expand_dims(alignment, 1),
                           tf.expand_dims(tf.range(self.n_states, dtype='int64'), 0))
        counts = tf.reduce_sum(tf.cast(matches, 'int64'), axis=0)

        return tf.assign_add(self.state_stats, counts)

    @method_scope("update_state_priors")
    def update_state_priors(self):
        state_priors = (
            tf.cast(self.state_stats, 'float32')
            / (tf.cast(tf.reduce_sum(self.state_stats), 'float32')) + 1e-4)

        with tf.control_dependencies([state_priors]):
            return tf.group(
                tf.assign(self.state_priors, state_priors),
                tf.assign(self.state_stats, tf.zeros_like(self.state_stats)))
