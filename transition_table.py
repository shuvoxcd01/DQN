from collections import deque
import random
import numpy as np


class TransitionTable(object):
    def __init__(self, maxlen=1000000):
        self.transitions = deque(maxlen=maxlen)

    def sample(self, size=1):
        assert len(self.transitions) >= size
        samples = random.sample(self.transitions, size)

        s = np.empty(shape=(size, 84, 84, 4))
        s2 = np.empty(shape=(size, 84, 84, 4))
        a = np.empty(shape=size)
        r = np.empty(shape=size)
        term = np.empty(shape=size)

        for i in range(size):
            s[i] = samples[i][0]
            a[i] = samples[i][1]
            r[i] = samples[i][2]
            s2[i] = samples[i][3]
            term[i] = samples[i][4]

        return s, a, r, s2, term

    def add(self, s, a, r, s2, is_term):
        term = 1 if is_term else 0
        self.transitions.append((s, a, r, s2, term))
