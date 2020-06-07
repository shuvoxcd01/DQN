import random
from datetime import datetime
import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

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

        self.save_model_steps = 1000
        self.save_model_path = None

        self.write_weight_histogram = False


class DeepQLearningAgent(object):
    def __init__(self, args):
        self.env = args.env
        assert self.env is not None, "Environment not given."
        self.actions = self.env.get_legal_actions()
        self.n_actions = len(self.actions)

        self.write_weight_histogram = args.write_weight_histogram
        self.network = args.network if args.network is not None else self.create_network()
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

        self.minibatch_size = args.minibatch_size
        self.target_q = args.target_q

        assert self.r_max is not None if self.rescale_r is not None else True, "R_MAX not defined"

        # save model
        self.save_model_steps = args.save_model_steps
        self.save_model_path = args.save_model_path if args.save_model_path is not None else "models/"
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)

        # learning rate annealing
        self.lr_start = 0.01 if args.lr is None else args.lr
        self.lr = self.lr_start
        self.lr_end = self.lr if args.lr_end is None else args.lr_end
        self.lr_endt = 1000000 if args.lr_endt is None else args.lr_endt

        # TODO
        self.deltas = []
        self.tmp = []
        self.g = []
        self.g2 = []
        for i in range(len(self.network.weights)):
            self.deltas.append(tf.zeros_like(self.network.weights[i]))
            self.tmp.append(tf.zeros_like(self.network.weights[i]))
            self.g.append(tf.zeros_like(self.network.weights[i]))
            self.g2.append(tf.zeros_like(self.network.weights[i]))

        # self.deltas = tf.zeros_like(tf.identity(self.network.weights))
        # self.tmp = tf.zeros_like(tf.identity(self.network.weights))
        # self.g = tf.zeros_like(tf.identity(self.network.weights))
        # self.g2 = tf.zeros_like(tf.identity(self.network.weights))

        self.experience_replay_memory = TransitionTable()

        cur_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir_scalar = "logs/scalars/" + cur_datetime
        logdir_histogram = "logs/histograms/" + cur_datetime
        self.scalar_file_writer = tf.summary.create_file_writer(logdir=logdir_scalar)
        self.histogram_file_writer = tf.summary.create_file_writer(logdir=logdir_histogram)

    def create_network(self):
        network = keras.models.Sequential()
        network.add(
            keras.layers.Conv2D(filters=32, kernel_size=(8, 8), strides=4, activation='relu', input_shape=(84, 84, 4)))
        network.add(keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=2, activation='relu'))
        network.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, activation='relu'))
        network.add(keras.layers.Flatten())
        network.add(keras.layers.Dense(units=512, activation='relu'))
        network.add(keras.layers.Dense(units=self.n_actions))

        self.write_weight_histogram = True

        return network

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

        # TODO
        assert len(dw) == len(self.network.weights), "len(dw) and len(network.weights) does not match"

        tmp_weights = []

        for i in range(len(self.network.weights)):
            self.g[i] = tf.multiply(self.g[i], 0.95) + tf.multiply(0.05, dw[i])
            self.tmp[i] = tf.multiply(dw[i], dw[i])
            self.g2[i] = tf.multiply(self.g2[i], 0.95) + tf.multiply(0.05, self.tmp[i])
            self.tmp[i] = tf.multiply(self.g[i], self.g[i])
            self.tmp[i] = tf.multiply(self.tmp[i], -1)
            self.tmp[i] = tf.add(self.tmp[i], self.g2[i])
            self.tmp[i] = tf.add(self.tmp[i], 0.01)
            self.tmp[i] = tf.sqrt(self.tmp[i])
            self.deltas[i] = tf.multiply(self.deltas[i], 0) + tf.multiply(tf.divide(dw[i], self.tmp[i]), self.lr)
            tmp_weights.append(tf.add(self.network.weights[i], self.deltas[i]))

        self.network.set_weights(tmp_weights)

    def learn_with_experience_replay(self):
        try:
            return_val = 0  # for TensorBoard

            preprocessed_input = self.env.initialize_input_sequence()

            while self.num_steps < self.total_steps:
                self.num_steps += 1

                if self.env.is_game_over():
                    preprocessed_input = self.env.initialize_input_sequence()

                    # Write total return of the last episode to TensorBoard
                    with self.scalar_file_writer.as_default():
                        tf.summary.scalar('Return', return_val, step=self.num_steps)
                        tf.summary.flush()

                    return_val = 0  # Reset return value

                action = self.e_greedy_select_action(preprocessed_input)
                reward, next_preprocessed_input = self.env.execute_action(action)

                return_val += reward  # Accumulate returns of the running episode in return_val

                self.experience_replay_memory.add(preprocessed_input, action, reward, next_preprocessed_input,
                                                  self.env.is_game_over())

                preprocessed_input = next_preprocessed_input

                # perform gradient descent step
                if (self.num_steps > self.learn_start) and (self.num_steps % self.update_freq == 0):
                    s, a, r, s2, term = self.experience_replay_memory.sample(self.minibatch_size)
                    self.q_learn_minibatch(s, a, r, s2, term)

                    # Write weight histograms to Tensorboard
                    if self.write_weight_histogram:
                        with self.histogram_file_writer.as_default():
                            tf.summary.histogram('Layer_0 (Conv) Weights', self.network.layers[0].weights[0],
                                                 step=self.num_steps)
                            tf.summary.histogram('Layer_0 (Conv) Bias', self.network.layers[0].weights[1],
                                                 step=self.num_steps)
                            tf.summary.histogram('Layer_1 (Conv) Weights', self.network.layers[1].weights[0],
                                                 step=self.num_steps)
                            tf.summary.histogram('Layer_1 (Conv) Bias', self.network.layers[1].weights[1],
                                                 step=self.num_steps)
                            tf.summary.histogram('Layer_2 (Conv) Weights', self.network.layers[2].weights[0],
                                                 step=self.num_steps)
                            tf.summary.histogram('Layer_2 (Conv) Bias', self.network.layers[2].weights[1],
                                                 step=self.num_steps)
                            tf.summary.histogram('Layer_4 (Dense) Weights', self.network.layers[4].weights[0],
                                                 step=self.num_steps)
                            tf.summary.histogram('Layer_4 (Dense) Bias', self.network.layers[4].weights[1],
                                                 step=self.num_steps)
                            tf.summary.histogram('Layer_5 (Dense) Weights', self.network.layers[5].weights[0],
                                                 step=self.num_steps)
                            tf.summary.histogram('Layer_5 (Dense) Bias', self.network.layers[5].weights[1],
                                                 step=self.num_steps)
                            tf.summary.flush()

                # update target-q network
                if self.target_q is not None and self.num_steps % self.target_q == 1:
                    self.target_network = tf.keras.models.clone_model(self.network)

                with self.scalar_file_writer.as_default():
                    tf.summary.scalar('Reward', reward, step=self.num_steps)
                    tf.summary.scalar('epsilon', self.epsilon, step=self.num_steps)
                    tf.summary.flush()

                if (self.num_steps % self.save_model_steps) == 0:
                    self.network.save(filepath=self.save_model_path + str(self.num_steps))

        except Exception as exception:
            print(exception)
            raise

        finally:
            # Close file writers
            self.scalar_file_writer.close()
            self.histogram_file_writer.close()

            # Save model
            self.network.save(filepath=self.save_model_path + str(self.num_steps))
