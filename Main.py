import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from OscillationMaps.Trainer.TrainGenerator2 import TrainModel

if __name__ == '__main__':
    complexities = [2]
    for c in complexities:
        print("Training ensemble of models of complexity ", c)
        model = TrainModel(complexity=c, plotR=True, verbose=True)
        model.train_ensemble(ensemble_size=1, epochs=1000)