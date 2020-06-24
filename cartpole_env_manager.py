import gym
import numpy as np

from env_manager import EnvManager


class CartPoleEnvManager(EnvManager):
    def __init__(self, show_screen=False):
        self.env = gym.make("CartPole-v0")
        self.done = False
        self.obs = None
        self.reward = 0
        self.show_screen = show_screen
        self.observation_shape = self.env.reset().shape

    def execute_action(self, action):
        if self.show_screen:
            self.env.render()
        self.obs, self.reward, self.done, _ = self.env.step(action)

        return self.reward, self.obs

    def is_game_over(self):
        return self.done

    def get_observation_shape(self):
        return self.observation_shape

    def initialize_input_sequence(self):
        self.obs = self.env.reset()
        self.done = False

        return self.obs

    def get_random_action(self):
        return self.env.action_space.sample()

    def get_legal_actions(self):
        return np.arange(self.env.action_space.n, dtype=np.int32)
