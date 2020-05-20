import random
import inspect
from datetime import datetime

import tensorflow as tf
from collections import deque

from ale_manager import ALEManager
from space_Invaders_q_network import DQNSpaceInvaders


class DeepQLearningAgent(object):
    def __init__(self, env_manager=ALEManager, q_network=DQNSpaceInvaders, num_total_episode=10000,
                 episode_starts_from=0, epsilon_decay_rate=9.000000000000001e-07, save_model_step=100, epsilon=1.,
                 logdir=None):
        self.minibatch_size = 32
        self.experience_replay_memory = deque([], maxlen=1000000)
        self.env_manager = env_manager() if inspect.isclass(env_manager) else env_manager
        self.possible_actions = self.env_manager.get_legal_actions()
        self.input_shape = self.env_manager.get_observation_shape()
        self.output_units = len(self.possible_actions)
        self.DQN = q_network(input_shape=self.input_shape, output_units=self.output_units) if inspect.isclass(
            q_network) else q_network
        self.epsilon = float(epsilon)
        self.gamma = 0.9
        self.num_total_episode = num_total_episode
        self.n_episode = episode_starts_from
        self.epsilon_decay_rate = epsilon_decay_rate
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S") if logdir is None else logdir
        self.file_writer = tf.summary.create_file_writer(logdir=logdir)
        self.save_model_step = save_model_step

    def update_epsilon(self):
        if self.epsilon < 0.1:
            self.epsilon = 0.1
            return
        elif self.epsilon == 0.1:
            return
        else:
            self.epsilon -= self.epsilon_decay_rate

    def e_greedy_select_action(self, preprocessed_input):
        if random.random() <= self.epsilon:
            action = self.env_manager.get_random_action()
        else:
            action = self.DQN.get_predicted_action(preprocessed_input)

        self.update_epsilon()

        return action

    def learn_with_experience_replay(self):
        """vanilla deep_q_learning_with_experience_replay"""
        while self.n_episode < self.num_total_episode:
            preprocessed_input = self.env_manager.initialize_input_sequence()
            cumulative_reward = 0
            episode_q_value_list = []
            while not self.env_manager.is_game_over():
                action = self.e_greedy_select_action(preprocessed_input)
                reward, next_preprocessed_input = self.env_manager.execute_action(action)

                cumulative_reward += reward
                q_value_for_selected_action = self.DQN.get_prediction(preprocessed_input)[action]
                episode_q_value_list.append(q_value_for_selected_action)

                self.experience_replay_memory.append(
                    (preprocessed_input, action, reward, next_preprocessed_input, self.env_manager.is_game_over()))

                preprocessed_input = next_preprocessed_input

                if len(self.experience_replay_memory) > self.minibatch_size:
                    sample_minibatch = random.sample(self.experience_replay_memory, k=self.minibatch_size)
                    _input, _output = self.DQN.prepare_minibatch(sample_minibatch, self.gamma)
                    self.DQN.perform_gradient_descent_step(_input, _output)

            avg_q_value_per_action = sum(episode_q_value_list) / float(len(episode_q_value_list))

            with self.file_writer.as_default():
                tf.summary.scalar('Return per episode', cumulative_reward, step=self.n_episode)
                tf.summary.scalar('Average q_value', avg_q_value_per_action, step=self.n_episode)
                tf.summary.scalar('epsilon', self.epsilon, step=self.n_episode)
                tf.summary.flush()

            if ((self.n_episode + 1) % self.save_model_step) == 0:
                self.DQN.save_model('-episode:' + str(self.n_episode + 1) + '-epsilon:' + str(self.epsilon))

            self.n_episode += 1
