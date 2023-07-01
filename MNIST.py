import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from Backpropagation import FCN, cross_entropy_loss


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
model = FCN(train_size[1] ** 2, 10)


for epoch in range(epochs):
    total_loss = 0.0
    for images, labels in tqdm(train_loader):
        X = torch.reshape(images, (images.shape[0], -1))
        y = one_hot_encoder(labels)

        ## get prediction
        y_hat = model(X)

        # cross entropy loss
        loss = cross_entropy_loss(y_hat, y)
        # print(loss)
        total_loss += loss

        model.train(X, y)
    print(f"epoches {epoch} loss: {total_loss}")
