from cartpole_env_manager import CartPoleEnvManager
from cartpole_q_network import DQNCartPole
from deep_q_learning import DeepQLearningAgent

if __name__ == '__main__':
    # dql_agent = DeepQLearningAgent(env_manager=CartPoleEnvManager, q_network=DQNCartPole, num_episode=200,
    #                                epsilon_decay_rate=0.000099)
    dql_agent = DeepQLearningAgent()
    dql_agent.learn_with_experience_replay()
