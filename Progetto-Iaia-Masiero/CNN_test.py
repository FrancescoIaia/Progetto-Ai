import gym
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from ast import literal_eval

def gather_data(env):
    num_trials = 10000
    min_score = 50
    sim_steps = 500
    trainingX, trainingY = [], []

    scores = []
    for _ in range(num_trials):
        observation = env.reset()
        score = 0
        training_sampleX, training_sampleY = [], []
        for step in range(sim_steps):
            # action corresponds to the previous observation so record before step
            action = np.random.randint(0, 2)
            one_hot_action = np.zeros(2)
            one_hot_action[action] = 1
            training_sampleX.append(observation)
            training_sampleY.append(one_hot_action)
            observation, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        if score > min_score:
            scores.append(score)
            trainingX += training_sampleX
            trainingY += training_sampleY

    trainingX, trainingY = np.array(trainingX), np.array(trainingY)
    print("Average: {}".format(np.mean(scores)))
    print("Median: {}".format(np.median(scores)))

    return trainingX, trainingY

def gather_data1():
    filename = "dataset.csv"

    data_train = pd.read_csv('dataset.csv')
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

env = gym.make("CartPole-v0")
trainingX, trainingY = gather_data1()
print(trainingY)
print(trainingX)
model = create_model()
model.fit(trainingY, trainingX, epochs=15)

scores = []
num_trials = 50
sim_steps = 500
for i in range(num_trials):
    print(i)
    observation = env.reset()
    score = 0
    for step in range(sim_steps):
        env.render()
        action = np.argmax(model.predict(observation.reshape(1, 4)))
        observation, reward, done, _ = env.step(action)
        score += reward
        if done:
            break
    scores.append(score)

print(np.mean(scores))