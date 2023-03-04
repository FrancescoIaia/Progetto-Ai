import gym
import pandas as pd
import numpy as np
from stable_baselines3 import DDPG

env = gym.make("Pendulum-v1")

model = DDPG("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=5_000)

observations_array, actions_array = [], []
vec_env = model.get_env()

episode = 5
step = 200
scores = []
for i in range(episode):
    print("episode: " + str(i) + "/" + str(episode))
    obs = vec_env.reset()
    score = 0
    training_obs, training_act = [], []
    for s in range(step):
        action, _states = model.predict(obs, deterministic=True)

        training_obs.append(obs[0].tolist())
        training_act.append(action[0])

        obs, reward, done, info = vec_env.step(action)
        vec_env.render()
        score += reward
        if done:
            break

        scores.append(score)
        observations_array += training_obs
        actions_array += training_act

print("Average: {}".format(np.mean(scores)))
print("Median: {}".format(np.median(scores)))

df = pd.DataFrame({'observation': observations_array, 'action': actions_array})
df.to_csv("dataset_sb_pendulum.csv", index=False)
env.close()
