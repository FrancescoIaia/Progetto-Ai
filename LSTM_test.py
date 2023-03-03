import gym
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, LSTM
from ast import literal_eval

from keras.utils import plot_model

filename = 'keras-rl/dataset_random.csv'


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


def create_model(states, actions):
    model = Sequential()

    model.add(LSTM(32, input_shape=(states, 1)))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(actions, activation="softmax"))

    model.compile(
        loss="mse",
        optimizer="adam",
        metrics=["accuracy"])
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    return model


env = gym.make("CartPole-v1")
states = env.observation_space.shape[0]
actions = env.action_space.n

act_data, obs_data = get_data()

print(act_data)
print(obs_data)

model = create_model(states, actions)

model.fit(obs_data, act_data, epochs=5)

scores = []
episode = 50
steps = 500
for i in range(episode):
    print("Episode: " + str(i) + "/" + str(episode))
    score = 0
    observation = env.reset()
    for step in range(steps):
        print("Step: " + str(step) + "/" + str(steps))
        action = np.argmax(model.predict(observation.reshape(1, states)))
        observation, reward, done, _ = env.step(action)
        #env.render()
        score += reward
        if done:
            break
    scores.append(score)

print("Average: {}".format(np.mean(scores)))
print("Median: {}".format(np.median(scores)))