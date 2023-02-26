import gym
import csv

from stable_baselines3 import PPO

memory = []
f = open('dataset.csv', 'w')
writer = csv.writer(f)

env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(20):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    memory = [action, obs, reward]
    writer.writerow(memory)

    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()


