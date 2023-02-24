# Relazione-WIP

Da questo [Git](https://github.com/tahmiid/DQNCartPoleAI) abbiamo preso un modello pre-addestrato e l’abbiamo tesato con pesi diversi. 

Per valutare le performance abbiamo usato i reward, confrontandoli variavano in base ai pesi caricati. Ed in generale i reward erano sempre superiori ai 400 su 500 di max

```python
#file del git TestModel.py

env = gym.make("CartPole-v1")
states = env.observation_space.shape[0]
actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,states)))
model.add(Dense(24, activation='relu'))
model.add(Dense(24, activation='relu'))
model.add(Dense(actions, activation='linear'))
```

Da questo [Git](https://github.com/matthiasplappert/keras-rl-weights) abbiamo provato il seguente modello pre-addestrato con vari pesi. Per valutare le performance abbiamo usato i reward e venivano 500 con questi pesi `dqn.load_weights('dqn_CartPole-v0_weights (2).h5f')`

[dqn_CartPole-v0_weights (2).h5f](Relazione%204b914fbdb58d4d789cd0b129dd906494/dqn_CartPole-v0_weights_(2).h5f)

```python
#file del git TestModel.py

model = Sequential()
model.add(Flatten(input_shape=(1,states)  ))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(actions))
model.add(Activation('linear'))
```

Cambiando env abbiamo testato `env = gym.make('Pendulum-v1')` usando una architettura, invece che *DQNAgent*, *DDPGAgent.* i pesi pretreinati si basavano su una versione deprecata di `Pendulum-v0` 

Abbiamo eseguito un ciclo di train con risultati scadenti. Quindi lo abbiamo eseguito un ulteriore ciclo caricando i pesi dell’iterazione precedente e i risultati sono migliorati notevolmente. 

[pendulum.py](/Progetto-Iaia-Masiero/TestPendulum.py/)

Cercando altri modelli pre-addestrati abbiamo trovato questa libreria 

[Stable-Baselines3 Docs - Reliable Reinforcement Learning Implementations — Stable Baselines3 1.8.0a6 documentation](https://stable-baselines3.readthedocs.io/en/master/index.html)

Che ha al suo interno dei modelli pre-addestrati associati ai vari algoritmi di RL.

Abbiamo testato un po’ di esempi nella documentazione modificando anche degli iperparametri, però i reward non erano soddisfacenti.

### Next Step

- Implementare nel modello una LSTM o GRU
- Migliorare le performance di Stable
