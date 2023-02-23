import gym
import random

import tensorflow as tf
import numpy as np
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation, LSTM, Embedding

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

env = gym.make("CartPole-v1")
states = env.observation_space.shape[0]
actions = env.action_space.n

#print(actions)

#episodes = 10
#for episode in range(1, episodes+1):
#    state = env.reset();
#    done = False
#    score = 0

#    while not done:
#        env.render()
#        action = random.choice([0,1])
#        n_state, reward, done, info = env.step(action)
#        score+=reward
#    print('Episode:{} Score:{}'.format(episode, score))

def build_model(states, actions):
    n_input = 4
    n_hidden = 10
    n_output = actions
    model = Sequential()

    model.add(LSTM(n_hidden, input_shape=(1, states)))
    model.add(Activation('tanh'))
    model.add(Dense(actions))
    model.add(Activation('sigmoid'))

    return model

model = build_model(states, actions)

model.summary()

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=5000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=5000, visualize=True, verbose=1)

scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))

dqn.save_weights('dqn_weights_2.h5f', overwrite=True)