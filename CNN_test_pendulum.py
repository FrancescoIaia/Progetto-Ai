import gym
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout
from ast import literal_eval

filename = 'Dataset/dataset_keras_pendulum.csv'

def get_data():
    data_train = pd.read_csv(filename)
    data_train["observation"] = data_train["observation"].apply(lambda x: literal_eval(x))
    data_train["action"] = data_train["action"].apply(lambda x: literal_eval(x))

    observations = data_train['observation']
    actions = data_train['action']

    obs, act = [], []

    for i in observations:
        obs.append(np.array(i))

    for i in actions:
        act.append(np.array(i))

    obs = np.array(obs)
    act = np.array(act)
    return act, obs


def create_model(obs, state):
    model = Sequential()

    model.add(Dense(128, input_shape=(obs,), activation="relu"))
    model.add(Dropout(0.6))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.6))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.6))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.6))

    model.add(Dense(state, activation="linear"))

    model.compile(
        loss="mse",
        optimizer="adam",
    )

    return model


act_data, obs_data = get_data()

env = gym.make("Pendulum-v1")
obs = env.observation_space.shape[0]
action = env.action_space.shape[0]

print(act_data)
print(obs_data)

model = create_model(obs, action)

model.fit(obs_data, act_data, epochs=50)

scores = []
episode = 10
steps = 200
for i in range(episode):
    print("Episode: " + str(i) + "/" + str(episode))
    score = 0
    observation = env.reset()
    for step in range(steps):
        print("Step: " + str(step) + "/" + str(steps))
        action = model.predict(observation.reshape(1, obs))
        observation, reward, done, _ = env.step(action)
        env.render()
        score += reward
        if done:
            break
    scores.append(score)

print("Average: {}".format(np.mean(scores)))
print("Median: {}".format(np.median(scores)))
