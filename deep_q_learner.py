import random
from datetime import datetime

import numpy as np
import tensorflow as tf

from transition_table import TransitionTable


class DQLAgentArgs(object):
    def __init__(self):
        self.env = None
        self.network = None
        self.total_steps = 50000000
        self.update_freq = 4
        self.learn_start = 50000
        self.epsilon_start = 1.  # Affects epsilon update
        self.epsilon_end = 0.1
        self.epsilon_endt = 1000000
        self.discount = 0.99
        self.rescale_r = None
        self.r_max = None
        self.minibatch_size = 32
        self.target_q = 10000
        # learning reate annealing
        self.lr = None
        self.lr_end = None
        self.lr_endt = None

        assert self.env is not None, "Environment not given."
        assert self.network is not None, "Network not given."
        assert self.r_max is not None if self.rescale_r is not None else True, "R_MAX not defined"


class DeepQLearningAgent(object):
    def __init__(self, args, logdir=None):
        self.env = args.env
        self.network = args.network
        self.target_network = tf.keras.models.clone_model(self.network)
        self.total_steps = args.total_steps
        self.update_freq = args.update_freq
        self.learn_start = args.learn_start
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_endt = args.epsilon_endt
        self.discount = args.discount
        self.num_steps = 0
        self.epsilon = self.epsilon_start
        self.rescale_r = args.rescale_r
        self.r_max = args.r_max
        self.actions = self.env.get_legal_actions()
        self.n_actions = len(self.actions)
        self.minibatch_size = args.minibatch_size
        self.target_q = args.target_q

        # learning rate annealing
        self.lr_start = 0.01 if args.lr is None else args.lr
        self.lr = self.lr_start
        self.lr_end = self.lr if args.lr_end is None else args.lr_end
        self.lr_endt = 1000000 if args.lr_endt is None else args.lr_endt

        self.deltas = tf.zeros_like(self.network.weights)
        self.tmp = tf.zeros_like(self.network.weights)
        self.g = tf.zeros_like(self.network.weights)
        self.g2 = tf.zeros_like(self.network.weights)
        self.experience_replay_memory = TransitionTable()

        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S") if logdir is None else logdir
        self.file_writer = tf.summary.create_file_writer(logdir=logdir)

    def update_epsilon(self):
        self.epsilon = self.epsilon_end + max(0, (self.epsilon_start - self.epsilon_end) * (
                self.epsilon_endt - max(0, self.num_steps - self.learn_start)) / self.epsilon_endt)

    def e_greedy_select_action(self, preprocessed_input):
        if random.random() <= self.epsilon:
            action = self.env.get_random_action()
        else:
            action = np.amax(self.network.predict(np.expand_dims(preprocessed_input, 0))[0])

        self.update_epsilon()

        return action

    def get_q_update(self, s, a, r, s2, term):
        term = tf.add(tf.multiply(tf.cast(tf.identity(term), tf.float32), -1), 1)
        q2_max = tf.reduce_max(tf.cast(tf.identity(self.target_network(s2)), tf.float32), axis=1)
        q2 = tf.multiply(tf.multiply(tf.identity(q2_max), self.discount), term)
        delta = tf.cast(tf.identity(r), tf.float32)
        if self.rescale_r:
            delta = tf.divide(delta, self.r_max)
        delta = tf.add(delta, q2)
        q_all = tf.cast(self.network(s), tf.float32)
        q = np.empty(shape=q_all.shape[0], dtype=np.float32)

        for i in range(q_all.shape[0]):
            q[i] = (q_all[i][a[i]])

        q = tf.Variable(q)
        delta = tf.add(delta, tf.multiply(q, -1))

        # clip delta not applied

        targets = np.zeros(shape=(self.minibatch_size, self.n_actions), dtype=np.float32)

        for i in range(min(self.minibatch_size, a.shape[0])):
            targets[i][a[i]] = delta[i]

        targets = tf.Variable(targets)

        return targets, delta, q2_max

    def q_learn_minibatch(self, s, a, r, s2, term):
        targets, delta, q2_max = self.get_q_update(s, a, r, s2, term)
        with tf.GradientTape() as t:
            y_hat = self.network(s)

        dw = t.gradient(y_hat, self.network.weights, output_gradients=targets)

        # Ignoring weight cost

        # compute linearly annealed learning rate
        t = max(0, self.num_steps - self.learn_start)
        self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t) / self.lr_endt + self.lr_end
        self.lr = max(self.lr, self.lr_end)

        self.g = tf.multiply(self.g, 0.95) + tf.multiply(0.05, dw)
        self.tmp = tf.multiply(dw, dw)
        self.g2 = tf.multiply(self.g2, 0.95) + tf.multiply(0.05, self.tmp)
        self.tmp = tf.multiply(self.g, self.g)
        self.tmp = tf.multiply(self.tmp, -1)
        self.tmp = tf.add(self.tmp, self.g2)
        self.tmp = tf.add(self.tmp, 0.01)
        self.tmp = tf.sqrt(self.tmp)

        self.deltas = tf.multiply(self.deltas, 0) + tf.multiply(tf.divide(dw, self.tmp), self.lr)
        self.network.set_weights(tf.add(self.network.weights, self.deltas))

    def learn_with_experience_replay(self):
        preprocessed_input = self.env.initialize_input_sequence()

        while self.num_steps < self.total_steps:
            self.num_steps += 1

            if self.env.is_game_over():
                preprocessed_input = self.env.initialize_input_sequence()

            action = self.e_greedy_select_action(preprocessed_input)
            reward, next_preprocessed_input = self.env.execute_action(action)

            self.experience_replay_memory.add(
                (preprocessed_input, action, reward, next_preprocessed_input, self.env.is_game_over()))

            preprocessed_input = next_preprocessed_input

            # perform gradient descent step
            if (self.num_steps > self.learn_start) and (self.num_steps % self.update_freq == 0):
                s, a, r, s2, term = self.experience_replay_memory.sample(self.minibatch_size)
                self.q_learn_minibatch(s, a, r, s2, term)

            # update target-q network
            if self.target_q is not None and self.num_steps % self.target_q == 1:
                self.target_network = tf.keras.models.clone_model(self.network)

            with self.file_writer.as_default():
                tf.summary.scalar('Reward', reward, step=self.num_steps)
                tf.summary.flush()
            #     tf.summary.scalar('Return per episode', cumulative_reward, step=self.n_episode)
            #     tf.summary.scalar('Average q_value', avg_q_value_per_action, step=self.n_episode)
            #     tf.summary.scalar('epsilon', self.epsilon, step=self.n_episode)
            #     tf.summary.flush()
            #
            # if ((self.n_episode + 1) % self.save_model_step) == 0:
            #     self.DQN.save_model('-episode:' + str(self.n_episode + 1) + '-epsilon:' + str(self.epsilon))
