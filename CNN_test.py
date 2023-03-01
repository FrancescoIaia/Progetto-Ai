import gym
import numpy as np
import pandas as pd
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from ast import literal_eval

filename = 'keras-rl/dataset_keras.csv'

def get_data():
    data_train = pd.read_csv(filename)
    data_train["observation"] = data_train["observation"].apply(lambda x: literal_eval(x))
    data_train["action"] = data_train["action"].apply(lambda x: literal_eval(x))

    observations = data_train['observation']
    actions = data_train['action']

    obs, act = [],  []

    for i in observations:
        obs.append(np.array(i))

    for i in actions:
        act.append(np.array(i))
    
    obs = np.array(obs)
    act = np.array(act)
    return act, obs


def create_model():
    model = Sequential()
    model.add(Dense(128, input_shape=(4,), activation="relu"))
    model.add(Dropout(0.6))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.6))

    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.6))

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.6))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.6))
    model.add(Dense(2, activation="softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])
    return model

env = gym.make("CartPole-v1")
act_data, obs_data = get_data()
print(act_data)
print(obs_data)

model = create_model()

model.fit(obs_data, act_data, epochs=5)

scores = []
num_trials = 50
sim_steps = 500
for i in range(num_trials):
    print(i)
    score = 0
    observation = env.reset()
    for step in range(sim_steps):
        action = np.argmax(model.predict(observation.reshape(1, 4)))
        observation, reward, done, _ = env.step(action)
        env.render()
        score += reward
        if done:
            break
    scores.append(score)

print(np.mean(scores))