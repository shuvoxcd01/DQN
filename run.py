import tensorflow as tf
from ale_manager import ALEManager, ALEManagerArgs
from deep_q_learner import DeepQLearningAgent, DQLAgentArgs

tf.debugging.set_log_device_placement(True)
tf.keras.backend.set_floatx('float32')

if __name__ == '__main__':
    env = ALEManager(ALEManagerArgs())
    agent_args = DQLAgentArgs()
    agent_args.env = env

    dql_agent = DeepQLearningAgent(args=agent_args)
    dql_agent.learn_with_experience_replay()
