from collections import deque
import ale_manager
import random
import deep_q_network


class DeepQLearning(object):
    def __init__(self):
        self.minibatch_size = 32
        self.experience_replay_memory = deque([], maxlen=1000000)
        self.env_manager = ale_manager.ALEManager()
        self.possible_actions = self.env_manager.get_legal_actions()
        self.input_shape = (84, 84, 4)
        self.output_units = len(self.possible_actions)
        self.DQN = deep_q_network.DQN(input_shape=self.input_shape, output_units=self.output_units)
        self.epsilon = 1.
        self.gamma = 0.9

    def update_epsilon(self):
        if self.epsilon < 0.1:
            self.epsilon = 0.1
            return
        elif self.epsilon == 0.1:
            return
        else:
            self.epsilon -= 9.000000000000001e-07  # epsilon -= (1-0.1) / 1000000

    def e_greedy_select_action(self, preprocessed_input):
        if random.random() <= self.epsilon:
            action = self.env_manager.get_random_action()
        else:
            action = self.DQN.get_predicted_action(preprocessed_input)

        self.update_epsilon()

        return action

    def deep_q_learning_with_experience_replay(self):
        for num_episode in range(10000):
            preprocessed_input = self.env_manager.initialize_input_sequence()
            while not self.env_manager.is_game_over():
                action = self.e_greedy_select_action(preprocessed_input)
                reward, next_preprocessed_input = self.env_manager.execute_action(action)
                self.experience_replay_memory.append(
                    (preprocessed_input, action, reward, next_preprocessed_input, self.env_manager.is_game_over()))
                if len(self.experience_replay_memory) > self.minibatch_size:
                    sample_minibatch = random.choices(self.experience_replay_memory, k=self.minibatch_size)
                    _input, _output = self.DQN.prepare_minibatch(sample_minibatch, self.gamma)
                    self.DQN.perform_gradient_descent_step(_input, _output)



if __name__ == '__main__':
    DeepQLearning().deep_q_learning_with_experience_replay()