import gym
import random

import tensorflow as tf
import numpy as np
import pandas as pd
import keras
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed, Input, Activation, Embedding
from keras.preprocessing.sequence import TimeseriesGenerator

filename = "dataset.csv"
data_train = pd.read_csv(filename)

data_features = data_train.copy()

data_features = np.array(data_features)
print(data_features)

env = gym.make("CartPole-v1")
states = env.observation_space.shape[0]
actions = env.action_space.n

def create_model(state_size, action_size):
    model = Sequential()

    model.add(LSTM(32, input_shape=(state_size, 2)))
    model.add(Dense(action_size))

    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=0.001))
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    return model

model = build_model(states, actions)

steps_per_epoch = 20
history = model.fit(data_features, data_features, steps_per_epoch=steps_per_epoch)


