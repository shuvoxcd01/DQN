from collections import deque
import  neural_model
import ale_manager
import random

from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
import  numpy as np

class DeepQLearning(object):
    def __init__(self):
        self.minibatch_size = 32
        self.experience_replay_memory = deque([], maxlen=1000000)
        self.env_manager = ale_manager.ALEManager()
        self.possible_actions = self.env_manager.get_legal_actions()
        self.DQN = neural_model.DQN(output_units=len(self.possible_actions))
        self.epsilon = 1.
        self.gamma = 0.9

    def get_q_network(self):
        model = Sequential()
        model.add(
            Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), input_shape=self.input_shape, activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=self.output_units))

        model.compile(loss="mse", optimizer=RMSprop())

        return model

    def update_epsilon(self):
        if self.epsilon < 0.1:
            self.epsilon = 0.1
            return
        elif self.epsilon == 0.1:
            return
        else:
            self.epsilon -= 9.000000000000001e-07  # epsilon -= (1-0.1) / 1000000

    def e_greedy_select_action(self, input):
        if random.random() <= self.epsilon:
            action = self.env_manager.get_random_action()
        else:
            action = DQN.get_predicted_action(input)

    def deep_q_learning_with_experience_replay(self):
        for num_episode in range(10000):
            preprocessed_input = self.env_manager.initialize_input_sequence()
            while not self.env_manager.is_game_over():


