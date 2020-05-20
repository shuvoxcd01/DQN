from abc import ABC, abstractmethod


class EnvManager(ABC):
    @abstractmethod
    def get_legal_actions(self):
        pass

    @abstractmethod
    def get_random_action(self):
        pass

    @abstractmethod
    def initialize_input_sequence(self):
        pass

    @abstractmethod
    def execute_action(self, action):
        pass

    @abstractmethod
    def is_game_over(self):
        pass
