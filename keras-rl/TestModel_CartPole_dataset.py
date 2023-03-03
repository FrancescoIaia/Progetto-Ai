import gym
import pandas as pd
import numpy as np

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Activation

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
    observations_array, actions_array = [], []
    min_score = 50
    episode = 10
    step = 500
    scores = []

    for i in range(episode):
        print("episode: " + str(i) + "/" + str(episode))
        score = 0
        obs = env.reset()
        training_obs, training_act = [], []
        for s in range(step):
            action = dqn.forward(obs) #np.random.randint(0, 2)
            one_action = [0., 0.]
            if action == 1:
                one_action[1] = 1.
            else:
                one_action[0] = 1.

            training_obs.append(obs.tolist())
            training_act.append(one_action)

            obs, reward, done, info = env.step(action)
            score += reward
            if done:
                break

        if score > min_score:
            scores.append(score)
            observations_array += training_obs
            actions_array += training_act

    print("Average: {}".format(np.mean(scores)))
    print("Median: {}".format(np.median(scores)))
    df = pd.DataFrame({'observation': observations_array, 'action': actions_array})
    df.to_csv("dataset_keras_cartpole.csv", index=False)
    env.close()


model = build_model(states, actions)
model.summary()

dqn = build_agent(model, actions)
dqn.compile(keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])
dqn.load_weights('weights/dqn_weights_2.h5f')

scores = dqn.test(env, nb_episodes=3, visualize=True)
print(scores.history['episode_reward'])

create_dataset(dqn)