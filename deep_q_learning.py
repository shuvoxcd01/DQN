import random
from collections import deque
from datetime import datetime

import tensorflow as tf
from IPython.display import clear_output


class DeepQLearningAgent(object):
    def __init__(self, env, q_network, num_total_episode=10000, episode_starts_from=0, save_model_step=100, epsilon=1.,
                 logdir=None):
        self.minibatch_size = 32
        self.experience_replay_memory = deque([], maxlen=1000000)
        self.agent_history_length = 4
        # self.target_network_update_frequency = 10000
        self.update_frequency = 4
        self.final_exploration_frame = 1000000
        self.replay_start_size = 50000
        self.env = env
        self.DQN = q_network
        self.epsilon = float(epsilon)
        self.num_total_episode = num_total_episode
        self.n_episode = episode_starts_from
        self.epsilon_decay_rate = 9.000000000000001e-07
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
            action = self.env.get_random_action()
        else:
            action = self.DQN.get_predicted_action(preprocessed_input)

        self.update_epsilon()

        return action

    def initialize_experience_replay_memory(self):
        frame_count = 0
        current_obs = self.env.reset()

        while frame_count < self.replay_start_size:
            action = self.env.get_random_action()
            next_obs, reward, done, info = self.env.act(action)
            self.experience_replay_memory.append((current_obs, action, reward, next_obs, done))
            current_obs = next_obs

            if done:
                current_obs = self.env.reset()

            frame_count += 1
            clear_output(wait=True)
            print("{:.1f}% done.".format(frame_count / float(self.replay_start_size) * 100))

        clear_output(wait=True)
        print("Replay memory initialized.")

    def learn_with_experience_replay(self):
        frame_count = 0
        self.initialize_experience_replay_memory()

        while self.n_episode < self.num_total_episode:
            current_obs = self.env.reset()
            done = False
            reward_per_episode = 0
            episode_q_value_list = []
            while not done:
                action = self.e_greedy_select_action(current_obs)
                next_obs, reward, done, info = self.env.act(action)

                reward_per_episode += reward
                q_value_for_selected_action = self.DQN.get_prediction(current_obs)[action]
                episode_q_value_list.append(q_value_for_selected_action)

                self.experience_replay_memory.append((current_obs, action, reward, next_obs, done))
                frame_count += 1
                current_obs = next_obs

                if frame_count % self.update_frequency == 0:
                    sample_minibatch = random.sample(self.experience_replay_memory, k=self.minibatch_size)
                    _input, _output = self.DQN.prepare_minibatch(sample_minibatch)
                    self.DQN.perform_gradient_descent_step(_input, _output)

            avg_q_value_per_action = sum(episode_q_value_list) / float(len(episode_q_value_list))

            with self.file_writer.as_default():
                tf.summary.scalar('Return per episode', reward_per_episode, step=self.n_episode)
                tf.summary.scalar('Average q_value per action', avg_q_value_per_action, step=self.n_episode)
                tf.summary.scalar('epsilon', self.epsilon, step=self.n_episode)
                tf.summary.flush()

            if ((self.n_episode + 1) % self.save_model_step) == 0:
                self.DQN.save_model('-episode:' + str(self.n_episode + 1) + '-epsilon:' + str(self.epsilon))

            self.n_episode += 1
