import gym
import pandas as pd
import numpy as np
from stable_baselines3 import DDPG

env = gym.make("Pendulum-v1")

model = DDPG("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

trainingX, trainingY = [], []
vec_env = model.get_env()

episode = 5
step = 200
scores = []
for i in range(episode):
    print("episode: " + str(i) + "/" + str(episode))
    obs = vec_env.reset()
    score = 0
    training_sampleX, training_sampleY = [], []
    for s in range(step):
        action, _states = model.predict(obs, deterministic=True)

        training_sampleX.append(obs[0].tolist())
        training_sampleY.append(action[0])

        obs, reward, done, info = vec_env.step(action)
        vec_env.render()
        score += reward
        if done:
            break

        scores.append(score)
        trainingX += training_sampleX
        trainingY += training_sampleY

print("Average: {}".format(np.mean(scores)))
print("Median: {}".format(np.median(scores)))

df = pd.DataFrame({'observation': trainingX, 'action': trainingY})
df.to_csv("dataset_sb_pendulum.csv", index=False)
env.close()
