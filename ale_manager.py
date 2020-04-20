import os
import logging
import sys
import random
import numpy as np
import cv2

from atari_py import ALEInterface, get_game_path, list_games
from env_manager import EnvManager


class ALEManager(EnvManager):

    def __init__(self, rom_name='Space_Invaders.bin', display_screen=False, frame_skip=3, color_averaging=True):
        self.logger = logging.getLogger(__name__)

        self.ale = ALEInterface()
        self.ale.setBool(b'display_screen', display_screen)
        self.ale.setInt(b'frame_skip', frame_skip)
        self.ale.setBool(b'color_averaging', color_averaging)
        self._load_rom(rom_name)
        self.actions = self.ale.getMinimalActionSet()
        self.sequence = np.empty(shape=(84, 84, 4), dtype=np.uint8)

    def _load_rom(self, rom_name):
        if rom_name in list_games():
            self.ale.loadROM(get_game_path(rom_name))
            return

        rom_path = os.path.join(os.path.dirname(os.path.abspath('__file__')), 'ROMs', rom_name)
        if not os.path.exists(rom_path):
            self.logger.error("Invalid ROM path")
            sys.exit(1)

        self.ale.loadROM(bytes(rom_path, encoding='utf-8'))

    def _map_action(self, action):
        return self.actions[action]

    def get_legal_actions(self):
        return np.arange(len(self.actions), dtype=np.int32)

    def get_random_action(self):
        return random.choice(self.get_legal_actions())

    def initialize_input_sequence(self):
        self.ale.reset_game()
        screen = np.empty((210, 160), dtype=np.uint8)
        for i in range(4):
            self.ale.act(self._map_action(self.get_random_action()))
            self.ale.getScreenGrayscale(screen)
            preprocessed_screen = self.preprocess_screen(screen)
            self.sequence[:, :, i] = preprocessed_screen
        return self.sequence

    @staticmethod
    def preprocess_screen(screen):
        resized_screen = cv2.resize(screen, dsize=(84, 110), interpolation=cv2.INTER_AREA)
        cropped_screen = resized_screen[17:110 - 9, :]
        return cropped_screen

    def execute_action(self, action):
        """Executes the action given as parameter and returns a
        reward and a sequence of length 4 containing preprocessed screens."""
        screen = np.empty((210, 160), dtype=np.uint8)
        reward = self.ale.act(self._map_action(action))
        self.ale.getScreenGrayscale(screen)
        preprocessed_screen = self.preprocess_screen(screen)
        self.sequence[:, :, :3] = self.sequence[:, :, 1:]
        self.sequence[:, :, -1] = preprocessed_screen

        return reward, self.sequence

    def is_game_over(self):
        return self.ale.game_over()

    def get_observation_shape(self):
        return (84, 84, 4)
