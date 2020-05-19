import os
from abc import abstractmethod
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop
import numpy as np
from keras.models import load_model


class DQNBreakout(object):
    def __init__(self, save_model_dir='models', save_model_name='model.h5',
                 load_model_dir=None, load_model_name=None):
        self.input_shape = (84, 84, 4)
        self.output_units = 4
        self.save_model_dir = save_model_dir
        self.save_model_name = save_model_name
        self.load_model_dir = load_model_dir
        self.load_model_name = load_model_name
        self.gamma = 0.99
        self.learning_rate = 0.00025
        self.gradient_momentum = 0.95
        # self.squared_gradient_momentum = 0.95
        # self.min_squared_gradient = 0.01
        self.model = self._load_model()

        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)

    def _load_model(self):
        if self.load_model_dir is None or self.load_model_name is None:
            print("Creating new neural-network")
            return self.get_q_network()

        model_name = os.path.join(self.load_model_dir, self.load_model_name)

        if os.path.exists(model_name):
            print("Loading existing model, " + str(model_name))
            return load_model(model_name)

        raise Exception("Model could not be loaded.")

    @abstractmethod
    def get_q_network(self):
        model = Sequential()
        model.add(
            Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), input_shape=self.input_shape, activation='relu'))
        model.add(Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=256, activation='relu'))
        model.add(Dense(units=self.output_units))

        model.compile(loss="mse", optimizer=RMSprop(learning_rate=self.learning_rate, momentum=self.gradient_momentum))

        return model

    def get_prediction(self, preprocessed_input):
        return self.model.predict(np.expand_dims(preprocessed_input, 0))[0]

    def get_predicted_action(self, preprocessed_input):
        return np.argmax(self.get_prediction(preprocessed_input))

    def prepare_minibatch(self, transitions_minibatch):
        expected_output_minibatch = []
        input_minibatch = []

        for current_input, action, reward, next_input, is_terminal_state in transitions_minibatch:
            q_value = reward
            if not is_terminal_state:
                q_value += self.gamma * np.amax(self.get_prediction(next_input))
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
