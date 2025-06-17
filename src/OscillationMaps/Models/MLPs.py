import torch
import numpy as np
from abc import ABC
import torch.nn as nn
from torch import stack, cat
# from EquationLearning.models.functions import get_function


class MLP(nn.Module, ABC):
    """Defines conventional NN architecture"""

    def __init__(self, input_features: int = 10, output_size: int = 1):
        """
        Initialize NN
        :param input_features: Input shape of the network.
        :param output_size: Output shape of the network.
        """
        super(MLP, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=100), nn.ReLU())
        self.drop1 = nn.Dropout(p=0.1)
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(in_features=100, out_features=500), nn.ReLU())
        self.drop2 = nn.Dropout(p=0.1)
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(in_features=500, out_features=100), nn.ReLU())
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(in_features=100, out_features=50), nn.ReLU())

        # Number of outputs depends on the method
        self.out = nn.Linear(50, output_size)

    def forward(self, x):
        x = self.hidden_layer1(x)
        # x = self.drop1(x)
        x = self.hidden_layer2(x)
        # x = self.drop2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        return self.out(x)


class MLP2(nn.Module, ABC):
    """Defines conventional NN architecture"""

    def __init__(self, input_features: int = 10, output_size: int = 1):
        """
        Initialize NN
        :param input_features: Input shape of the network.
        :param output_size: Output shape of the network.
        :param n_layers: Number of hidden layers.
        """
        super(MLP2, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=500), nn.ReLU())
        self.drop1 = nn.Dropout(p=0.01)
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(in_features=200, out_features=500), nn.ReLU())
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(in_features=200, out_features=500), nn.ReLU())
        self.drop2 = nn.Dropout(p=0.05)
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(in_features=500, out_features=100), nn.ReLU())
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(in_features=100, out_features=50), nn.ReLU())

        # Number of outputs depends on the method
        self.out = nn.Linear(50, output_size)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        return self.out(x)


class MLP3(nn.Module, ABC):
    """Defines conventional NN architecture"""

    def __init__(self, input_features: int = 10, output_size: int = 1):
        """
        Initialize NN
        :param input_features: Input shape of the network.
        :param output_size: Output shape of the network.
        """
        super(MLP3, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=200), nn.ReLU())
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(in_features=200, out_features=500), nn.ReLU())
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(in_features=500, out_features=500), nn.ReLU())
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(in_features=500, out_features=100), nn.ReLU())
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(in_features=100, out_features=50), nn.ReLU())

        # Number of outputs depends on the method
        self.out = nn.Linear(50, output_size)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        return self.out(x)


class MLP4(nn.Module, ABC):
    """Defines conventional NN architecture with dropout"""

    def __init__(self, input_features: int = 10, output_size: int = 1, dropout_prob: float = 0.3):
        """
        Initialize NN
        :param input_features: Input shape of the network.
        :param output_size: Output shape of the network.
        :param dropout_prob: Probability of dropping units during training.
        """
        super(MLP4, self).__init__()
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=200),
            nn.ReLU()
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(in_features=200, out_features=500),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(in_features=500, out_features=500),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(in_features=500, out_features=500),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob)
        )
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(in_features=500, out_features=100),
            nn.ReLU()
        )
        self.hidden_layer6 = nn.Sequential(
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU()
        )

        self.out = nn.Linear(50, output_size)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.hidden_layer6(x)
        return self.out(x)
