from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
import numpy as np


class DQN(object):
    def __init__(self, input_shape, output_units):
        self.input_shape = input_shape
        self.output_units = output_units
        self.model = self.get_q_network()

    def get_q_network(self):
        model = Sequential()
        model.add(
            Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), input_shape=self.input_shape, activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=self.output_units))

        model.compile(loss="mse", optimizer=RMSprop())

        return model

    def get_prediction(self, preprocessed_input):
        return self.model.predict(np.expand_dims(preprocessed_input, 0))[0]

    def get_predicted_action(self, preprocessed_input):
        return np.argmax(self.get_prediction(preprocessed_input))

    def prepare_minibatch(self, transitions_minibatch, gamma):
        expected_output_minibatch = []
        input_minibatch = []

        for current_input, action, reward, next_input, is_terminal_state in transitions_minibatch:
            q_value = reward
            if not is_terminal_state:
                q_value += gamma * np.amax(self.get_prediction(next_input))
            prediction = self.get_prediction(current_input)
            prediction[action] = q_value
            expected_output_minibatch.append(prediction)
            input_minibatch.append(current_input)

        expected_output_minibatch = np.array(expected_output_minibatch)
        input_minibatch = np.array(input_minibatch)

        return input_minibatch, expected_output_minibatch

    def perform_gradient_descent_step(self, _input, _output):
        self.model.fit(x=_input, y=_output, epochs=1)
