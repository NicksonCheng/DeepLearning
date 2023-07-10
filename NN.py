import torch
import torch.nn as nn
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = (nn.Flatten(),)
        self.linear = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        # x = self.flatten(x)
        output = self.linear(x)
        return output
