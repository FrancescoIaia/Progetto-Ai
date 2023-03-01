import gym
import pandas as pd
import numpy as np
from stable_baselines3 import DQN, PPO

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

trainingX, trainingY = [], []
vec_env = model.get_env()

min_score = 50
episode = 50
step = 500
scores = []
obs = vec_env.reset()
for i in range(episode):
    print("episode: "+ str(i) + "/" + str(episode))
    score = 0
    training_sampleX, training_sampleY = [], []
    for s in range(step):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()

        one_hot_action = [0., 0.]
        if action == 1:
            one_hot_action[1] = 1.
        else:
            one_hot_action[0] = 1.
        training_sampleX.append(obs[0].tolist())
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
df.to_csv("dataset.csv", index=False)
env.close()


