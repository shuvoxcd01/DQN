import os
from abc import abstractmethod

import numpy as np
from keras.models import load_model


class DQN(object):
    def __init__(self, input_shape, output_units, save_model_dir='models', save_model_name='model.h5',
                 load_model_dir=None, load_model_name=None):
        self.input_shape = input_shape
        self.output_units = output_units
        self.save_model_dir = save_model_dir
        self.save_model_name = save_model_name
        self.load_model_dir = load_model_dir
        self.load_model_name = load_model_name
        self.model = self._load_model()

        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)

    def _load_model(self):
        if self.load_model_dir is None and self.load_model_name is None:
            return self.get_q_network()

        model_name = os.path.join(self.load_model_dir, self.load_model_name)

        if os.path.exists(model_name):
            return load_model(model_name)

        raise Exception("Model could not be loaded.")

    @abstractmethod
    def get_q_network(self):
        pass

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

    def save_model(self, step=''):
        model_name = os.path.join(self.save_model_dir, (str(step) + '--' + self.save_model_name))
        self.model.save(model_name)
