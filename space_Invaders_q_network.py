from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop

from deep_q_network import DQN


class DQNSpaceInvaders(DQN):

    def __init__(self, input_shape, output_units, model_dir="models/space_invaders", model_name='model.h5'):
        super().__init__(input_shape, output_units, model_dir, model_name)

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
