import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

env = gym.make('Pendulum-v1', render_mode="human")
observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    env.render()
    observation, reward, terminated, truncated, info = env.step(action)
    states = env.observation_space.shape
    model = Sequential()
    model.add(Dense(24, activation='relu', input_shape=states))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.summary()
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    if terminated or truncated:
        observation, info = env.reset()
env.close()
