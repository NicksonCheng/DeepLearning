import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


def cross_entropy_loss(outputs, labels):
    # label_one_hot = torch.zeros(outputs.size())
    # label_one_hot[torch.arange(outputs.size()[0]), labels] = 1

    # Compute the loss
    loss = -(labels * torch.log(outputs))
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
        self.b1 = torch.randn(self.hidden_dim)
        # weight between hidden and output layer
        self.w2 = torch.randn(self.hidden_dim, self.output_dim)
        self.b2 = torch.randn(self.output_dim)

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def sigmoid_derivative(self, s):
        return s * (1 - s)

    def softmax(self, x):
        exp_x = torch.exp(x)

        return exp_x / torch.sum(exp_x, dim=1, keepdim=True)

    def forward(self, X):
        # first lienar combination
        self.y1 = torch.matmul(X, self.w1) + self.b1
        # first non-linear activate function
        self.y2 = self.sigmoid(self.y1)

        # second lienar combination
        self.y3 = torch.matmul(self.y2, self.w2) + self.b2
        # second non-linear activate function
        y4 = self.softmax(self.y3)
        print(self.w1, self.w2, self.b1, self.b2)
        return y4

    def backward(self, X, y_hat, y_true):
        # Calculate gradients of the output layer
        dL_dy4 = y_hat - y_true  # Gradient of loss w.r.t. y4
        dL_dy3 = dL_dy4  # softmax BP

        # Calculate gradients of the second layer weights and biases
        dL_dw2 = torch.matmul(self.y2.T, dL_dy3)  # Gradient of loss w.r.t. w2
        dL_db2 = torch.sum(dL_dy3, dim=0)  # Gradient of loss w.r.t. b2

        # Update the parameters of the second layer
        self.w2 -= self.learning_rate * dL_dw2
        self.b2 -= self.learning_rate * dL_db2

        # Calculate gradients of the first layer
        dL_dy2 = torch.matmul(dL_dy3, self.w2.T)  # Gradient of loss w.r.t. y2
        dL_dy1 = dL_dy2 * self.sigmoid_derivative(self.y1)  # Gradient of loss w.r.t. y1

        # Calculate gradients of the first layer weights and biases
        dL_dw1 = torch.matmul(X.T, dL_dy1)  # Gradient of loss w.r.t. w1
        dL_db1 = torch.sum(dL_dy1, dim=0)  # Gradient of loss w.r.t. b1

        # Update the parameters of the first layer
        self.w1 -= self.learning_rate * dL_dw1
        self.b1 -= self.learning_rate * dL_db1

    def train(self, X, y_true):
        y_hat = self.forward(X)
        self.backward(X, y_hat, y_true)
