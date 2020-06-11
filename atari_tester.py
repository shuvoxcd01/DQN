import random
import numpy as np
import tensorflow as tf
from ale_manager import ALEManager, ALEManagerArgs
from deep_q_learner import DeepQLearningAgent, DQLAgentArgs
import time

tf.debugging.set_log_device_placement(True)
tf.keras.backend.set_floatx('float32')

if __name__ == '__main__':
    env_args = ALEManagerArgs()
    env_args.SHOW_SCREEN = True
    env = ALEManager(env_args)
    agent_args = DQLAgentArgs()
    agent_args.env = env
    epsilon = 0.05

    try:
        dql_agent = tf.keras.models.load_model('./models/2162784saved_from_finally_block-h5')
    except Exception as e:
        print(e)
        raise

    for i in range(10):
        obs = env.initialize_input_sequence()
        return_val = 0
        actions = []
        while True:
            if env.is_game_over():
                print("Return of Episode " + str(i) + " is: ", return_val)
                # print("actions:", np.unique(np.array(actions)))
                # print(env.env.unwrapped.get_action_meanings())
                # time.sleep(3)
                break

            if random.random() <= epsilon:
                action = env.get_random_action()
            else:
                action = np.argmax(dql_agent.predict(np.expand_dims(obs, 0))[0])
                # actions.append(action)
                # print(action)
                # print(dql_agent.predict(np.expand_dims(obs, 0)))
                # time.sleep(1)

            # print(action)

            reward, obs = env.execute_action(action)
            return_val += reward

            # time.sleep(0.5)
