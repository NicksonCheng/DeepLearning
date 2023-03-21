import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy
from tqdm import tqdm


class FCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FCN, self).__init__()
        # dimension for input, hiiden and output layer
        self.input_dim = input_dim
        self.hidden_dim = 32
        self.output_dim = output_dim

        # laerning rate
        self.learning_rate = 0.001

        # weight between input and hidden layer
        self.w1 = torch.randn(self.input_dim, self.hidden_dim)

        # weight between hidden and output layer
        self.w2 = torch.randn(self.hidden_dim, self.output_dim)

    def sigmoid(self, s):
        return 1/1 + torch.exp(-s)

    def forward(self, X):
        # first lienar combination
        self.y1 = torch.matmul(X, self.w1)

        # first non-linear activate function
        self.y2 = self.sigmoid(self.y1)

        # second lienar combination
        self.y3 = torch.matmul(self.y2, self.w2)

        # second non-linear activate function
        y4 = self.sigmoid(self.y3)
        print(y4)
        return y4

    def backward():
        pass

    def train(self, X, l):
        output_y = self.forward(X)
        print(output_y)
        return output_y


transform = transforms.Compose([
    transforms.ToTensor()
])
epochs = 10
batch_size = 64

train_data = datasets.MNIST(root='./mnist', train=True,
                            transform=transform, download=True)
test_data = datasets.MNIST(root='./mnist', train=False,
                           transform=transform, download=True)


train_size = train_data.train_data.size()

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


for data, label in tqdm(train_loader):
    output = FCN(data)

    print(output)
