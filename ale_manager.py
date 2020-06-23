import logging
import numpy as np

import gym
from gym.wrappers import AtariPreprocessing, FrameStack

from env_manager import EnvManager


class ALEManagerArgs(object):
    def __init__(self):
        self.ROM_NAME = 'PongNoFrameskip-v4'
        self.SHOW_SCREEN = False
        self.SCREEN_SIZE = 84
        self.FRAME_SKIP = 4  # ACTION_REPEAT
        self.COLOR_AVERAGING = True
        self.NO_OP_MAX = 30
        self.AGENT_HISTORY_LENGTH = 4
        self.TERMINAL_ON_LIFE_LOSS = True
        self.GRAYSCALE_OBS = True
        self.SCALE_OBS = False


class ALEManager(EnvManager):
    def __init__(self, args):
        self.logger = logging.getLogger(__name__)
        self.ROM_NAME = args.ROM_NAME
        self.SHOW_SCREEN = args.SHOW_SCREEN
        self.SCREEN_SIZE = args.SCREEN_SIZE
        self.FRAME_SKIP = args.FRAME_SKIP
        self.COLOR_AVERAGING = args.COLOR_AVERAGING
        self.NO_OP_MAX = args.NO_OP_MAX
        self.AGENT_HISTORY_LENGTH = args.AGENT_HISTORY_LENGTH
        self.TERMINAL_ON_LIFE_LOSS = args.TERMINAL_ON_LIFE_LOSS
        self.GRAYSCALE_OBS = args.GRAYSCALE_OBS
        self.SCALE_OBS = args.SCALE_OBS

        env = gym.make(self.ROM_NAME)
        env = AtariPreprocessing(env, noop_max=self.NO_OP_MAX, frame_skip=self.FRAME_SKIP, screen_size=self.SCREEN_SIZE,
                                 terminal_on_life_loss=self.TERMINAL_ON_LIFE_LOSS, grayscale_obs=self.GRAYSCALE_OBS,
                                 scale_obs=self.SCALE_OBS)

        self.env = FrameStack(env, num_stack=self.AGENT_HISTORY_LENGTH)

        self.cur_obs = None
        self.cur_reward = 0
        self.done = False

    def get_legal_actions(self):
        return np.arange(self.env.action_space.n, dtype=np.int32)

    def get_random_action(self):
        return self.env.action_space.sample()

    def initialize_input_sequence(self):
        self.cur_obs = self.env.reset()
        self.done = False
        return np.moveaxis(self.cur_obs, 0, -1)

    def execute_action(self, action):
        if self.SHOW_SCREEN:
            self.env.render()
        self.cur_obs, self.cur_reward, self.done, info = self.env.step(action)
        return self.cur_reward, np.moveaxis(self.cur_obs, 0, -1)

    def is_game_over(self):
        return self.done

    def get_observation_shape(self):
        return self.SCREEN_SIZE, self.SCREEN_SIZE, self.AGENT_HISTORY_LENGTH
