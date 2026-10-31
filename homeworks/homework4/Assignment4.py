import torch
import torchvision
import cv2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.transforms import v2


def CIFAR10_dataset_a():
    """write the code to grab a single mini-batch of 4 images from the training set, at random. 
   Return:
    1. A batch of images as a torch array with type torch.FloatTensor. 
    The first dimension of the array should be batch dimension, the second channel dimension, 
    followed by image height and image width. 
    2. Labels of the images in a torch array

    """
    transform = v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32, scale=True),v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4
    train_data = torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform = transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True, num_workers=0)
    images, labels = next(iter(trainloader))
    return images, labels

class Net(nn.Module):
    # Use this function to define your network
    # Creates the network
    def __init__(self):
        super().__init__()
        # Inits the model layers
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)


    def forward(self, x):
        # Defines forward apth
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

def train_classifier():
    # Creates dataset
    transform = v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32, scale=True),v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = torchvision.datasets.CIFAR10(root='./data',train = True, download = True,transform = transform)
    trainloader = torch.utils.data.DataLoader(train_data,batch_size=4,shuffle=True,num_workers=0)

    # Creates Network 
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr = 0.001, momentum = 0.9)

    # Defines loss function and optimizer
    for epoch in range(2):  # loop over the dataset for 2 iteration
        running_loss = 0.0
        for i, data in enumerate(trainloader,0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print("Finish Training")

    # Saves the model weights after training
    PATH = './cifar_net_2epoch.pth'
    torch.save(net.state_dict(), PATH)

def evalNetwork():
    # Initialized the network and load from the saved weights
    PATH = './cifar_net_2epoch.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH, map_location="cpu"))
    net.eval()
    # Loads dataset
    batch_size=4
    transform = v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32, scale=True),v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    testset = torchvision.datasets.CIFAR10(root='./data',train = False,download = True,transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')

def get_first_layer_weights():
    net = Net()
    # TODO: load the trained weights
    net.load_state_dict(torch.load('./cifar_net_2epoch.pth',map_location = 'cpu',weights_only = True))
    first_weight = net.conv1.weight.detach().clone()  # TODO: get conv1 weights (exclude bias)
    return first_weight

def get_second_layer_weights():
    net = Net()
    # TODO: load the trained weights
    net.load_state_dict(torch.load('./cifar_net_2epoch.pth',map_location = 'cpu',weights_only = True))
    second_weight = net.conv2.weight.detach().clone()  # TODO: get conv2 weights (exclude bias)
    return second_weight

def sampled(net,dataset):
    indices = torch.randperm(len(dataset))[:1000].tolist()
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(subset,batch_size=100,num_workers = 0)
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            predict = net(images).argmax(dim=1)
            correct += (predict == labels).sum().item()
            total += labels.size(0)
    net.train()
    return correct / total

def hyperparameter_sweep():
    '''
    Reuse the CNN and training code from Question 2
    Train the network three times using different learning rates: 0.01, 0.001, and 0.0001
    During training, record the training loss every 2000 iterations
    compute and record the training and test errors every 2000 iterations by randomly sampling 1000 images from each dataset
    After training, plot three curves
    '''
    transform = v2.Compose([v2.ToImage(),v2.ToDtype(torch.float32, scale=True),v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = torchvision.datasets.CIFAR10(root='./data',train = True, download = True,transform = transform)
    test_data = torchvision.datasets.CIFAR10(root='./data',train=False,download=True,transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data,batch_size=4,shuffle=True,num_workers=0)
    
    res = {}
    for lr in [0.01,0.001,0.0001]:
        net = Net()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(),lr=lr,momentum=0.9)
        iteration = 0
        running_loss = 0.0
        record = {'iterations':[], 'loss':[], 'train_error':[], 'test_error':[]}
        for epoch in range(2):
            for inputs, labels in trainloader:
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs,labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                iteration += 1
                if iteration % 2000 == 0:
                    train_acc = sampled(net,train_data)
                    test_acc = sampled(net,test_data)
                    record['iterations'].append(iteration)
                    record['loss'].append(running_loss/2000)
                    record['train_error'].append(100 * (1-train_acc))
                    record['test_error'].append(100 * (1-test_acc))
                    print(
                        f'lr={lr}, iteration={iteration}, '
                        f'loss={running_loss / 2000:.3f}, '
                        f'train accuracy={100 * train_acc:.1f}%, '
                        f'test accuracy={100 * test_acc:.1f}%',
                        flush=True
                    )
                    running_loss = 0.0

        res[lr] = record

    torch.save(res,'results')
    for key, ylabel,filename in [
        ('loss', 'Training loss', 'training_loss.png'),
        ('train_error', 'Training error (%)', 'training_error.png'),
        ('test_error', 'Test error (%)', 'test_error.png')
    ]:
        plt.figure()
        for lr, record in res.items():
            plt.plot(record['iterations'], record[key], label=f'lr={lr}')
        plt.xlabel('Iteration')
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(filename,dpi=300)
        plt.close()
    return res

    

if __name__ == "__main__":
    #train_classifier()
    #evalNetwork()
    hyperparameter_sweep()