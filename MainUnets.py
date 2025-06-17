import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from OscillationMaps.Trainer.TrainFNN import TrainModel

if __name__ == '__main__':
    model = TrainModel(plotR=True, verbose=True)
    model.train_ensemble(ensemble_size=1, epochs=1000, batch_size=1024)
