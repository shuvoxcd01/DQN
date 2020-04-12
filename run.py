from cartpole_env_manager import CartPoleEnvManager
from cartpole_q_network import DQNCartPole
from deep_q_learning import DeepQLearningAgent

if __name__ == '__main__':
    env_manager = CartPoleEnvManager()
    load_model_dir = 'models/cart_pole'
    load_model_name = '199--model.h5'
    input_shape = env_manager.get_observation_shape()
    output_units = len(env_manager.get_legal_actions())
    q_network = DQNCartPole(input_shape=input_shape, output_units=output_units, load_model_dir=load_model_dir,
                            load_model_name=load_model_name)
    dql_agent = DeepQLearningAgent(env_manager=env_manager, q_network=q_network, num_total_episode=20,
                                   episode_starts_from=18, epsilon_decay_rate=0.000099, save_model_step=1)
    # dql_agent = DeepQLearningAgent()
    dql_agent.learn_with_experience_replay()
