from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop

from deep_q_network import DQN


class DQNCartPole(DQN):
    def __init__(self, input_shape, output_units, model_dir='models/cart_pole', model_name='model.h5'):
        super().__init__(input_shape, output_units, model_dir, model_name)

    def get_q_network(self):
        model = Sequential()
        model.add(Dense(units=24, input_shape=self.input_shape, activation='relu'))
        model.add(Dense(units=24, activation='relu'))
        model.add(Dense(units=self.output_units, activation='linear'))

        model.compile(loss="mse", optimizer=RMSprop())

        return model
