import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import torch.nn as nn
from opacus import PrivacyEngine  # pip install opacus


def loadData():
    """
    Load Alzheimer's train and test set
    """
    transform = transforms.Compose([
        transforms.CenterCrop(180),
        transforms.Resize(255),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = ImageFolder(
        "C:/Datasets/Alzheimer_s Dataset/train", transform=transform)
    testset = ImageFolder(
        "C:/Datasets/Alzheimer_s Dataset/test", transform=transform)
    trainloader = DataLoader(trainset, batch_size=50, shuffle=True)
    testloader = DataLoader(testset, batch_size=50, shuffle=True)
    return trainloader, testloader


def train(net, trainLoader, epochs, device):
    """
    Train the network on the training set.
    """
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    privacy_engine = PrivacyEngine(net,
                                   # batchS-size/length of dataset
                                   sample_rate=50/len(trainLoader.dataset),
                                   # length of data
                                   sample_size=len(trainLoader.dataset),
                                   noise_multiplier=1.5,  # amount of noise
                                   max_grad_norm=2.0)
    privacy_engine.attach(optimizer)
    loss = 0.0
    for _ in range(epochs):
        correct, total = 0, 0
        for images, labels in trainLoader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
        accuracy = correct/total
        print("Epoch Accuracy:", accuracy)


def test(net, testloader, device):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy
