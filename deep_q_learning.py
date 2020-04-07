from collections import deque
import ale_manager
import random
import deep_q_network


class DeepQLearning(object):
    def __init__(self, env_manager=ale_manager.ALEManager(), input_shape=(84, 84, 4), q_network=deep_q_network.DQN,
                 num_episode=10000):
        self.minibatch_size = 32
        self.experience_replay_memory = deque([], maxlen=1000000)
        self.env_manager = env_manager
        self.possible_actions = self.env_manager.get_legal_actions()
        self.input_shape = input_shape
        self.output_units = len(self.possible_actions)
        self.DQN = q_network(input_shape=self.input_shape, output_units=self.output_units)
        self.epsilon = 1.
        self.gamma = 0.9
        self.num_episode = num_episode

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
        for episode in range(self.num_episode):
            preprocessed_input = self.env_manager.initialize_input_sequence()
            while not self.env_manager.is_game_over():
                action = self.e_greedy_select_action(preprocessed_input)
                reward, next_preprocessed_input = self.env_manager.execute_action(action)
                self.experience_replay_memory.append(
                    (preprocessed_input, action, reward, next_preprocessed_input, self.env_manager.is_game_over()))
                preprocessed_input = next_preprocessed_input
                if len(self.experience_replay_memory) > self.minibatch_size:
                    sample_minibatch = random.sample(self.experience_replay_memory, k=self.minibatch_size)
                    _input, _output = self.DQN.prepare_minibatch(sample_minibatch, self.gamma)
                    self.DQN.perform_gradient_descent_step(_input, _output)


if __name__ == '__main__':
    dql_agent = DeepQLearning(env_manager=env_manager_cartpole.EnvManager(), input_shape=4,
                              q_network=deep_q_network_cartpole.DQN_CartPole, num_episode=100)
    dql_agent.deep_q_learning_with_experience_replay()
