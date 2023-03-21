import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy
from tqdm import tqdm


def cross_entropy_loss(outputs, labels):

    label_one_hot = torch.zeros(outputs.size())
    label_one_hot[torch.arange(outputs.size()[0]), labels] = 1

    # Compute the loss
    loss = -(label_one_hot * torch.log(outputs))
    loss = loss.sum(dim=1).mean()
    return loss.item()


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
        return 1/(1 + torch.exp(-s))

    def forward(self, X):
        # first lienar combination
        X = torch.reshape(X, (X.shape[0], -1))
        self.y1 = torch.matmul(X, self.w1)
        # first non-linear activate function
        self.y2 = self.sigmoid(self.y1)

        # second lienar combination
        self.y3 = torch.matmul(self.y2, self.w2)
        # second non-linear activate function
        y4 = self.sigmoid(self.y3)
        return y4

    def backward(self, X,):
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


train_size = train_data.train_data.shape
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
model = FCN(train_size[1]**2, 10)
total_loss = []
for data, labels in tqdm(train_loader):
    outputs = model(data)
    loss = cross_entropy_loss(outputs, labels)
    total_loss.append(loss)
    model.train(data, labels)
    break
