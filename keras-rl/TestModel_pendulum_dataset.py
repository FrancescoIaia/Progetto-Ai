import numpy as np
import gym
import keras

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate
import pandas as pd
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess


def create_dataset(agent):
    observations_array, actions_array = [], []
    episode = 100
    step = 200
    scores = []
    for i in range(episode):
        print("episode: " + str(i) + "/" + str(episode))
        score = 0
        obs = env.reset()
        training_obs, training_act = [], []
        for s in range(step):
            action = agent.forward(obs)
            obs, reward, done, info = env.step(action)
            env.render()

            training_obs.append(obs.tolist())
            training_act.append(action)
            score += reward
            if done:
                break

        scores.append(score)
        observations_array += training_obs
        actions_array += training_act

    print("Average: {}".format(np.mean(scores)))
    print("Median: {}".format(np.median(scores)))
    df = pd.DataFrame({'observation': observations_array, 'action': actions_array})
    df.to_csv("dataset_keras_pendulum.csv", index=False)
    env.close()


ENV_NAME = 'Pendulum-v1'

env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(16))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)

agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(keras.optimizers.Adam(learning_rate=.001, clipnorm=1.), metrics=['mae'])

agent.fit(env, nb_steps=15000, visualize=False, verbose=1, nb_max_episode_steps=200)

agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)

create_dataset(agent)
