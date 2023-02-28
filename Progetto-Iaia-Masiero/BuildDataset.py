import gym
import numpy as np
import pandas as pd
from keras.dtensor.integration_test_utils import train_step
from keras.utils.version_utils import training

from stable_baselines3 import PPO

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)


trainingX, trainingY = [], []
vec_env = model.get_env()
obs = vec_env.reset()

for i in range(2000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    one_hot_action = [0., 0.]
    if action == 1:
        one_hot_action[1] = 1.
    else:
        one_hot_action[0] = 1.
    trainingX.append(obs[0].tolist())
    trainingY.append(one_hot_action)

    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

df = pd.DataFrame({'observation': trainingX, 'action': trainingY})
df.to_csv("dataset.csv", index=False)
env.close()


