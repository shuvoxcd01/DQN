import gym

from env_manager import EnvManager


class CartPoleEnvManager(EnvManager):

    def __init__(self, display_screen=False):
        self.env = gym.make("CartPole-v0")
        self.done = False

        self.display_screen = display_screen

    def get_legal_actions(self):
        return list(range(self.env.action_space.n))

    def get_random_action(self):
        return self.env.action_space.sample()

    def initialize_input_sequence(self):
        observation = self.env.reset()
        self.done = False
        return observation

    def execute_action(self, action):
        if self.display_screen:
            self.env.render()

        observation, reward, self.done, info = self.env.step(action)
        # reward = reward if not self.done else -10
        return reward, observation

    def is_game_over(self):
        return self.done

    def get_observation_shape(self):
        return (4,)
