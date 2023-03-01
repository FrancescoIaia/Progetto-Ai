import gym
import random
import pandas as pd
import numpy as np

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
    model = Sequential()

    model.add(Flatten(input_shape=(1, states)))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
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

def create_dataset(dqn):
    trainingX, trainingY = [], []
    min_score = 50
    episode = 50
    step = 500
    scores = []
    for i in range(episode):
        print("episode: " + str(i) + "/" + str(episode))
        score = 0
        obs = env.reset()
        training_sampleX, training_sampleY = [], []
        for s in range(step):
            action = dqn.forward(obs)
            obs, reward, done, info = env.step(action)
            env.render()

            one_hot_action = [0., 0.]
            if action == 1:
                one_hot_action[1] = 1.
            else:
                one_hot_action[0] = 1.
            training_sampleX.append(obs.tolist())
            training_sampleY.append(one_hot_action)
            score += reward
            if done:
                break

        if score > min_score:
            scores.append(score)
            trainingX += training_sampleX
            trainingY += training_sampleY

    print("Average: {}".format(np.mean(scores)))
    print("Median: {}".format(np.median(scores)))
    df = pd.DataFrame({'observation': trainingX, 'action': trainingY})
    df.to_csv("dataset_keras.csv", index=False)
    env.close()


model = build_model(states, actions)
model.summary()

dqn = build_agent(model, actions)
dqn.compile(keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])
dqn.load_weights('weights/dqn_weights_2.h5f')

scores = dqn.test(env, nb_episodes=5, visualize=True)
print(scores.history['episode_reward'])

create_dataset(dqn)