import json

import tensorflow as tf
from ale_manager import ALEManager, ALEManagerArgs
from deep_q_learner import DeepQLearningAgent, DQLAgentArgs

# tf.debugging.set_log_device_placement(True)
# tf.keras.backend.set_floatx('float32')

if __name__ == '__main__':
    network = tf.keras.models.load_model('models/815864saved_from_finally_block-h5')

    with open('models/agent_state.json', 'r') as f:
        agent_state = json.load(f)

    num_steps = agent_state["num_steps"]
    epsilon = agent_state["epsilon"]
    lr = agent_state["lr"]
    learn_start = num_steps + 50000
    write_weight_histogram = True

    print("num_steps: ", num_steps)
    print("epsilon_start : ", epsilon)
    print("lr: ", lr)
    print("learn_start: ", learn_start)
    print("write_weight_histogram: ", write_weight_histogram)

    env = ALEManager(ALEManagerArgs())
    agent_args = DQLAgentArgs()
    agent_args.env = env
    agent_args.num_steps = num_steps
    agent_args.epsilon_start = epsilon
    agent_args.lr = lr
    agent_args.learn_start = learn_start

    agent_args.network = network
    agent_args.write_weight_histogram = write_weight_histogram

    dql_agent = DeepQLearningAgent(args=agent_args)
    dql_agent.learn_with_experience_replay()
