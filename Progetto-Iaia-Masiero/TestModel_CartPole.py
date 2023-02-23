import gym
import random

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, LSTM

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make("CartPole-v1")
states = env.observation_space.shape[0]
actions = env.action_space.n

def build_model(states, actions):
    """
    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))

    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(actions))
    model.add(Activation('linear'))
    """
    n_input = 4
    n_hidden = 10
    n_output = 1
    initial_state = 0.1
    training_threshold = 1.5
    step_threshold = 0.5

    model = Sequential()

    model.add(LSTM(n_hidden, input_shape=(1, states)))
    model.add(Activation('tanh'))
    model.add(Dense(n_output))
    model.add(Activation('sigmoid'))
    model.add(Dense(actions))
    model.add(Activation('linear'))

    print(model.summary())

    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=1000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn


model = build_model(states, actions)
model.summary()

dqn = build_agent(model, actions)
dqn.compile(keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])

dqn.load_weights('dqn_weights_2.h5f')

dqn.test(env, nb_episodes=10, visualize=True)