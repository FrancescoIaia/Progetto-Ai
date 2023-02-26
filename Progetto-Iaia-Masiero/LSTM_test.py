import gym
import random

import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, LSTM, Embedding

env = gym.make("CartPole-v1")
states = env.observation_space.shape[0]
actions = env.action_space.n

def build_model(states, actions):
    additional_metrics = ['accuracy']
    batch_size = 128
    embedding_output_dims = 15
    loss_function = BinaryCrossentropy()
    max_sequence_length = 300
    num_distinct_words = 5000
    number_of_epochs = 5
    optimizer = tf.keras.optimizers.Adam()
    validation_split = 0.20
    verbosity_mode = 1

    model = Sequential()
    #model.add(Embedding(, , input_length=max_sequence_length))
    model.add(LSTM(10))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss=loss_function, metrics=additional_metrics)
    model.summary()

    return model

model = build_model(states, actions)
