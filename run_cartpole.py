import json

import tensorflow as tf
from tensorflow import keras
#import keras
from cartpole_env_manager import CartPoleEnvManager
from deep_q_learner import DeepQLearningAgent, DQLAgentArgs

if __name__ == '__main__':
    # network = tf.keras.models.load_model('models/815864saved_from_finally_block-h5')
    #
    # with open('models/agent_state.json', 'r') as f:
    #     agent_state = json.load(f)
    #
    # num_steps = agent_state["num_steps"]
    # epsilon = agent_state["epsilon"]
    # lr = agent_state["lr"]
    # learn_start = num_steps + 50000
    # write_weight_histogram = True
    #
    # print("num_steps: ", num_steps)
    # print("epsilon_start : ", epsilon)
    # print("lr: ", lr)
    # print("learn_start: ", learn_start)
    # print("write_weight_histogram: ", write_weight_histogram)
    #

    env = CartPoleEnvManager()
    agent_args = DQLAgentArgs()
    agent_args.env = env

    network = keras.models.Sequential()
    network.add(keras.layers.Dense(units=24, activation='relu', input_shape=env.get_observation_shape()))
    network.add(keras.layers.Dense(units=24, activation='relu'))
    network.add(keras.layers.Dense(units=len(env.get_legal_actions()), activation='linear'))
    # agent_args.num_steps = num_steps
    # agent_args.epsilon_start = epsilon
    # agent_args.lr = lr
    # agent_args.learn_start = learn_start

    agent_args.total_steps = 25000 * 2
    agent_args.epsilon_endt = 10000 * 2
    agent_args.learn_start = 64
    agent_args.minibatch_size = 32
    agent_args.target_q = 200
    agent_args.update_freq = 4
    agent_args.save_model_steps = 500
    agent_args.save_model_path = 'models/cartpole/'
    agent_args.logdir_scalar = 'logs/cartpole/scalar'
    agent_args.logdir_histogram = 'logs/cartpole/histogram'

    agent_args.network = network
    agent_args.write_weight_histogram = False

    dql_agent = DeepQLearningAgent(args=agent_args)
    dql_agent.learn_with_experience_replay()
