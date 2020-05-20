from collections import deque
import random


class TransitionTable(object):
    def __init__(self, maxlen=1000000):
        self.transitions = deque(maxlen=maxlen)

    def sample(self, size=1):
        assert len(self.transitions) >= size
        return random.sample(self.transitions, size)

    def add(self, s, a, r, s2, is_term):
        term = 1 if is_term else 0
        self.transitions.append((s, a, r, s2, term))