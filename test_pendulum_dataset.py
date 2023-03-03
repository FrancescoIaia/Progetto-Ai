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
    trainingX, trainingY = [], []
    episode = 100
    step = 200
    scores = []
    for i in range(episode):
        print("episode: " + str(i) + "/" + str(episode))
        score = 0
        obs = env.reset()
        training_sampleX, training_sampleY = [], []
        for s in range(step):
            action = agent.forward(obs)
            obs, reward, done, info = env.step(action)
            # env.render()


            training_sampleX.append(obs.tolist())
            training_sampleY.append(action)
            score += reward
            if done:
                break

        scores.append(score)
        trainingX += training_sampleX
        trainingY += training_sampleY

    print("Average: {}".format(np.mean(scores)))
    print("Median: {}".format(np.median(scores)))
    df = pd.DataFrame({'observation': trainingX, 'action': trainingY})
    df.to_csv("dataset_keras_pendulum.csv", index=False)
    env.close()


ENV_NAME = 'Pendulum-v1'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Next, we build a very simple model.
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

# Finally, we configure and compile our agent. You can use every built-in tensorflow.keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
                  random_process=random_process, gamma=.99, target_model_update=1e-3)
agent.compile(keras.optimizers.Adam(learning_rate=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=10000, visualize=False, verbose=1, nb_max_episode_steps=200)

# After training is done, we save the final weights.
#agent.save_weights(f'ddpg_{ENV_NAME}_weights.h5f', overwrite=True)
#agent.load_weights("ddpg_Pendulum-v1_weights.h5f")

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)

create_dataset(agent)
