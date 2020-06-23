from collections import deque
import random
import numpy as np
import tensorflow as tf


class TransitionTable(object):
    def __init__(self, shape=(84, 84, 4), maxlen=100000):
        self.transitions = deque(maxlen=maxlen)
        self.s_shape = shape

    def sample(self, size=1):
        assert len(self.transitions) >= size
        samples = random.sample(self.transitions, size)

        s = np.empty(shape=(size, *self.s_shape), dtype=np.float32)
        s2 = np.empty(shape=(size, *self.s_shape), dtype=np.float32)
        a = np.empty(shape=size, dtype=np.int32)
        r = np.empty(shape=size, dtype=np.float32)
        term = np.empty(shape=size, dtype=np.float32)

        for i in range(size):
            s[i] = samples[i][0]
            a[i] = samples[i][1]
            r[i] = samples[i][2]
            s2[i] = samples[i][3]
            term[i] = samples[i][4]

        s = tf.Variable(s, dtype=tf.float32)
        a = tf.Variable(a, dtype=tf.int32)
        r = tf.Variable(r, dtype=tf.float32)
        s2 = tf.Variable(s2, dtype=tf.float32)
        term = tf.Variable(term, dtype=tf.float32)

        return s, a, r, s2, term

    def add(self, s, a, r, s2, is_term):
        term = 1 if is_term else 0
        self.transitions.append((s, a, r, s2, term))
