import random
import inspect
import numpy as np
from datetime import datetime

import tensorflow as tf
from collections import deque

from ale_manager import ALEManager
from space_Invaders_q_network import DQNSpaceInvaders
from transition_table import TransitionTable


class DQLAgentArgs(object):
    def __init__(self):
        self.ENV = None
        self.NETWORK = None
        self.TOTAL_STEPS = 50000000
        self.UPDATE_FREQ = 4
        self.LEARN_START = 50000
        self.EPSILON_START = 1.  # Affects epsilon update
        self.EPSILON_END = 0.1
        self.EPSILON_ENDT = 1000000
        self.DISCOUNT = 0.99
        self.RESCALE_R = None
        self.R_MAX = None
        self.MINIBATCH_SIZE = 32
        # learning reate annealing
        self.LR = None
        self.LR_END = None
        self.LR_ENDT = None

        assert self.ENV is not None, "Environment not given."
        assert self.NETWORK is not None, "Network not given."
        assert self.R_MAX is not None if self.RESCALE_R is not None else True, "R_MAX not defined"


class DeepQLearningAgent(object):
    def __init__(self, args, save_model_step=100, epsilon=1., logdir=None):
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

        # self.minibatch_size = 32
        # self.experience_replay_memory = deque([], maxlen=1000000)
        # self.env_manager = env_manager() if inspect.isclass(env_manager) else env_manager
        # self.possible_actions = self.env_manager.get_legal_actions()
        # self.input_shape = self.env_manager.get_observation_shape()
        # self.output_units = len(self.possible_actions)
        # self.DQN = q_network(input_shape=self.input_shape, output_units=self.output_units) if inspect.isclass(
        #     q_network) else q_network
        # self.epsilon = float(epsilon)
        # self.gamma = 0.9
        # self.num_total_episode = num_total_episode
        # self.n_episode = episode_starts_from
        # self.epsilon_decay_rate = epsilon_decay_rate
        # logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S") if logdir is None else logdir
        # self.file_writer = tf.summary.create_file_writer(logdir=logdir)
        # self.save_model_step = save_model_step

    def update_epsilon(self):
        self.epsilon = self.epsilon_end + max(0, (self.epsilon_start - self.epsilon_end) * (
                self.epsilon_endt - max(0, self.num_steps - self.learn_start)) / self.epsilon_endt)

    def e_greedy_select_action(self, preprocessed_input):
        if random.random() <= self.epsilon:
            action = self.env_manager.get_random_action()
        else:
            action = self.DQN.get_predicted_action(preprocessed_input)

        self.update_epsilon()

        return action

    # def prepare_minibatch(self, transitions_minibatch):
    #     expected_output_minibatch = []
    #     input_minibatch = []
    #
    #     for current_input, action, reward, next_input, is_terminal_state in transitions_minibatch:
    #         q_value = reward
    #         if not is_terminal_state:
    #             q_value += self.DISCOUNT * np.amax(self.get_prediction(next_input))
    #         prediction = self.get_prediction(current_input)
    #         prediction[action] = q_value
    #         expected_output_minibatch.append(prediction)
    #         input_minibatch.append(current_input)
    #
    #     expected_output_minibatch = np.array(expected_output_minibatch)
    #     input_minibatch = np.array(input_minibatch)
    #
    #     return input_minibatch, expected_output_minibatch

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

            if (self.num_steps > self.learn_start) and (self.num_steps % self.update_freq == 0):
                s, a, r, s2, term = self.experience_replay_memory.sample(self.minibatch_size)
                self.q_learn_minibatch(s, a, r, s2, term)

            # with self.file_writer.as_default():
            #     tf.summary.scalar('Return per episode', cumulative_reward, step=self.n_episode)
            #     tf.summary.scalar('Average q_value', avg_q_value_per_action, step=self.n_episode)
            #     tf.summary.scalar('epsilon', self.epsilon, step=self.n_episode)
            #     tf.summary.flush()
            #
            # if ((self.n_episode + 1) % self.save_model_step) == 0:
            #     self.DQN.save_model('-episode:' + str(self.n_episode + 1) + '-epsilon:' + str(self.epsilon))
