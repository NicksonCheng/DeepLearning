import numpy as np
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from Backpropagation import FCN, cross_entropy_loss
from NN import NeuralNetwork


def one_hot_encoder(label):
    # One-hot encode the labels
    one_hot_labels = torch.zeros((len(label), 10))
    one_hot_labels[torch.arange(len(label)), label] = 1
    return one_hot_labels


transform = transforms.Compose([transforms.ToTensor()])
epochs = 10
batch_size = 64
train_dataset = datasets.MNIST(
    root="./mnist", train=True, transform=transform, download=True
)
test_dataset = datasets.MNIST(
    root="./mnist", train=False, transform=transform, download=True
)

train_size = train_dataset.data.shape
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


## initial model and parameter

# model = FCN(train_size[1] ** 2, 10)
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork()
model.to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
for epoch in range(epochs):
    total_loss = 0.0
    start = time.time()
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        X = torch.reshape(images, (images.shape[0], -1))
        y = one_hot_encoder(labels).to(device)
        ## get prediction
        y_hat = model(X)
        # cross entropy loss
        # loss = cross_entropy_loss(y_hat, y)
        loss = criterion(y_hat, y)
        total_loss += loss.item()

        # backward and optimized
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # model.train(X, y)
    average_loss = total_loss / len(train_loader)
    end = time.time()
    print(f"epoches {epoch} loss: {average_loss}  time: {end-start}")

model.eval()


## no gradient descent and no backpropagation
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = torch.reshape(images, (images.shape[0], -1)).to(device)
        labels = labels.to(device)
        one_hot_labels = one_hot_encoder(labels).to(device)

        output = model(images)

        predicted = torch.argmax(output.data, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test Accuracy {(100*correct/total):.2f}%")
