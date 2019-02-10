import numpy as np
import tensorflow as tf


class HybridHMM:
    def __init__(self, posterior, transitions, state_priors=None):
        self.n_states = len(transitions)
        self.posterior = posterior

        state_priors = state_priors or np.full([self.n_states], 1 / self.n_states)

        with tf.variable_scope("transitions") as scope:
            self.A = tf.Variable(transitions, dtype='float32', name="A")
            # self.A_mask = tf.Variable(transitions != 0, dtype='bool', name="A mask")
            self.state_priors = tf.Variable(
                state_priors, dtype='float32', name="state_priors")
            self.update_statistics = tf.Variable(
                np.zeros([self.n_states, self.n_states]),
                dtype='float32', name="update_statistics")

            self.scope = scope

    def pseudo_lkh(self, x):
        return self.posterior(x) / tf.expand_dims(self.state_priors, 0)

    def forward_step(self, alpha_t, pseudo_lkh_tm1):
        return tf.tensordot(tf.transpose(alpha_t), self.A, 1) * pseudo_lkh_tm1

    def forward(self, pseudo_lkh):
        """Compute α terms, including virtual α_0"""
        alpha_0 = tf.constant([0.] * (self.n_states - 1) + [1.])
        alpha = tf.scan(
            self.forward_step,
            pseudo_lkh,
            initializer=alpha_0,
            back_prop=False)
        return tf.concat([tf.expand_dims(alpha_0, 0), alpha], axis=0)

    def backward_step(self, beta_tp1, pseudo_lkh_tp1):
        return tf.tensordot(self.A, pseudo_lkh_tp1 * beta_tp1, 1)

    def backward(self, pseudo_lkh):
        """Compute β terms"""
        beta_T = tf.ones([self.n_states])
        beta = tf.scan(
            self.backward_step,
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

    def viterbi(self, pseudo_lkh):
        path_proba_0 = tf.ones([self.n_states])
        path_ptr_0 = tf.constant([self.n_states - 1] * self.n_states, 'int64')

        path_proba, path_ptr = tf.scan(
            self.viterbi_step,
            pseudo_lkh,
            initializer=(path_proba_0, path_ptr_0))

        return tf.scan(
            lambda i, ptrs: ptrs[i],
            path_ptr[1:],
            initializer=tf.argmax(path_proba[-1]),
            reverse=True)

    def accumulate_updates(self, pseudo_lkh):
        alpha = self.forward(pseudo_lkh)
        beta = self.backward(pseudo_lkh)

        qty = tf.expand_dims(alpha[:-1], 2) \
            * tf.expand_dims(self.A, 0) \
            * tf.expand_dims(pseudo_lkh, 1) \
            / tf.expand_dims(tf.expand_dims(self.state_priors, 0), 0) \
            * tf.expand_dims(beta, 1)

        qty = qty / tf.reduce_sum(qty, [1, 2], keepdims=True)

        return tf.assign_add(self.update_statistics, tf.reduce_sum(qty, 0))

    def update_transitions(self):
        A = self.update_statistics \
            / tf.reduce_sum(self.update_statistics, axis=1, keepdims=True)
        with tf.control_dependencies([A]):
            update_statistics = tf.zeros_like(self.update_statistics)

        return tf.assign(self.A, A), tf.assign(self.update_statistics, update_statistics)
