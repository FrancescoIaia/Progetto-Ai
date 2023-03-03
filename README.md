# Progetto AI Reinforcement Learning 
### Emanuele Masiero 872695
### Iaia Francesco 869373
La releazione del progetto è all'interno del [file](Relazione/Relazione.md)
# Struttura Git
- [Dataset](Dataset): Cartella contenente tutti i dataset commentati nella relazione
- [CNN_test_cartpole.py](CNN_test_cartpole.py): 
- [LSTM_test.py](LSTM_test.py): 
- [README.md](README.md): Questo Readme
## Keras-rl
Dentro questa cartella ci sono tutti i file che utilizzano la libreria Keras-rl
- [BuildModel_CartPole](keras-rl/BuildModel_CartPole.py): File usato per creare i pesi, con problema cartpole e DQNAgent
- [TestModel_CartPole_dataset](keras-rl/TestModel_CartPole_dataset.py): File usato per creare il dataset di cartpole 
con i pesi creati in precedenza e presenti nella [cartella weights](keras-rl/weights)
- [TestModel_pendulum_dataset](keras-rl/TestModel_pendulum_dataset.py): File usato per creare il dataset. A differenza 
di cartpole qua creiamo due modelli uno agente e uno critic e facciamo direttamente il fit senza salvare i pesi

## Stable-baselines
Dentro questa cartella ci sono i file che usano la cartella stable-baseline
- [BuildDataset_cartpole_sb](stable-baselines/BuildDataset_cartpole_sb.py): File che crea il dataset con sb3 per il problema
cartpole
- [BuildDataset_pendulum_sb](stable-baselines/BuildDataset_pendulum_sb.py): File che crea il dataset con sb3 per il problema
pendulum 