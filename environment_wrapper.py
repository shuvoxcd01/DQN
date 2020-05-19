import numpy as np
import gym
from gym.wrappers import AtariPreprocessing, FrameStack


class BreakoutEnv:
    def __init__(self):
        self.noop_max = 30
        self.frame_skip = 4
        self.screen_size = 84
        self.terminal_on_life_loss = True
        self.grayscale_obs = True
        self.num_stack = 4

        env = gym.make('BreakoutNoFrameskip-v4')
        env = AtariPreprocessing(env, noop_max=self.noop_max, frame_skip=self.frame_skip, screen_size=self.screen_size,
                                 terminal_on_life_loss=self.terminal_on_life_loss, grayscale_obs=self.grayscale_obs)
        self.env = FrameStack(env, num_stack=self.num_stack)

    def get_random_action(self):
        return self.env.action_space.sample()

    def act(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = np.moveaxis(np.array(obs), 0, -1)

        reward = 1 if reward > 0 else 0 if reward == 0 else -1

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = np.moveaxis(np.array(obs), 0, -1)
        return obs
