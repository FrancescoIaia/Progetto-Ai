import gym
import pandas as pd
import numpy as np
from stable_baselines3 import DQN, PPO

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

observations_array, actions_array = [], []
vec_env = model.get_env()

min_score = 50
episode = 5
step = 500
scores = []
for i in range(episode):
    print("episode: " + str(i) + "/" + str(episode))
    obs = vec_env.reset()
    score = 0
    training_obs, training_act = [], []
    for s in range(step):
        action, _states = model.predict(obs, deterministic=True)

        one_action = [0., 0.]
        if action == 1:
            one_action[1] = 1.
        else:
            one_action[0] = 1.
        training_obs.append(obs[0].tolist())
        training_act.append(one_action)

        obs, reward, done, info = vec_env.step(action)
        vec_env.render()
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
df.to_csv("dataset.csv", index=False)
env.close()


