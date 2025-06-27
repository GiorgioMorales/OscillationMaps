import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from OscillationMaps.Trainer.TrainFNN import TrainModel

if __name__ == '__main__':
    model = TrainModel(verbose=True, model_i=4)
    model.train_ensemble(ensemble_size=1, epochs=200, batch_size=1024, scratch=True)
