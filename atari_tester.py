import cv2
import random
from ale_manager import ALEManager
from space_Invaders_q_network import DQNSpaceInvaders


class AtariTestDQN(DQNSpaceInvaders):
    def __init__(self, input_shape, output_units, load_model_dir, load_model_name):
        super().__init__(input_shape=input_shape, output_units=output_units, load_model_dir=load_model_dir,
                         load_model_name=load_model_name)


class AtariTester():
    def __init__(self, model_dir, model_name, rom_name):
        self.env_manager = ALEManager(rom_name=rom_name, display_screen=True, frame_skip=4, color_averaging=True)
        self.input_shape = self.env_manager.get_observation_shape()
        self.output_units = len(self.env_manager.get_legal_actions())
        self.q_network = AtariTestDQN(input_shape=self.input_shape, output_units=self.output_units,
                                      load_model_dir=model_dir, load_model_name=model_name)
        self.epsilon = 0.05
        self.return_list = []

    def test_performance(self, episodes=10):
        # cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        for _ in range(episodes):
            print("episode: " + str(_))
            total_reward_episode = 0
            _input_sequence = self.env_manager.initialize_input_sequence()
            while not self.env_manager.is_game_over():
                # cv2.imshow('frame', _input_sequence[:, :, 0])
                # cv2.waitKey(500) & 0xFF
                if random.random() <= self.epsilon:
                    action = self.env_manager.get_random_action()
                else:
                    action = self.q_network.get_predicted_action(_input_sequence)
                # print(action)
                reward, _input_sequence = self.env_manager.execute_action(action)
                total_reward_episode += reward
                print(self.env_manager.is_game_over())

            self.return_list.append(total_reward_episode)
        # cv2.destroyAllWindows()
        return self.return_list


model_dir = 'models/breakout'
model_name = '-episode_7900-epsilon_0.1--breakout_dqn.h5'

tester = AtariTester(model_dir=model_dir, model_name=model_name, rom_name='breakout')
return_list = tester.test_performance()

print(return_list)
