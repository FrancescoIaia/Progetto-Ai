# Progetto AI Reinforcement Learning 
### Emanuele Masiero 
### Iaia Francesco 
# La releazione del progetto è all'interno del [file](Relazione/Relazione.md)
# Struttura Git

- [Dataset](Dataset): Cartella contenente tutti i dataset commentati nella relazione
- [CNN_test_cartpole](CNN_test_cartpole.py): File che esegue un test di una cnn sul problema cartpole 
- [CNN_test_pendulum](CNN_test_pendulum.py): File che esegue un test di una cnn sul problema pendulum 
- [LSTM_test_cartpole](LSTM_test_cartpole.py): File che esegue un test di una LSTM sul problema cartpole 
- [LST_test_pendulum](LSTM_test_pendulum.py): File che esegue un test di una LSTM sul problema pendulum 
- [Relazione](Relazione/Relazione.md): Relazione dettagliata del progetto con discussione risultati

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
