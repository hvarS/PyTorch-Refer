# -*- coding: utf-8 -*-
"""CNN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Tq6HUya2PrC0SmyOIFo2c_eVtguRED2q
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class CNN(nn.Module):
  def __init__(self,in_channels = 1,num_classes = 10):
    super(CNN,self).__init__()
    self.conv1 = nn.Conv2d(in_channels= in_channels,out_channels = 8,kernel_size =(3,3),stride = (1,1),padding = (1,1))
    self.pool1 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
    self.conv2 = nn.Conv2d(in_channels= 8,out_channels = 16,kernel_size =(3,3),stride = (1,1),padding = (1,1))
    self.pool2 = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
    self.fc1 = nn.Linear(16*7*7,num_classes)

  def forward(self,x):
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = x.reshape(x.shape[0],-1)
    x = self.fc1(x)
    return x

model = CNN(1,10)

x = torch.randn((64,1,28,28))

print(model(x).shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device

in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 4

train_dataset = datasets.MNIST(root = "dataset/",train = True,transform = transforms.ToTensor(),download = True)
train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True)
test_dataset = train_dataset = datasets.MNIST(root = "dataset/",train = False,transform = transforms.ToTensor(),download = True)
test_loader = DataLoader(dataset = test_dataset,batch_size = batch_size,shuffle = True)

model = CNN(1,10).to(device = device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

for epoch in range(num_epochs):
  for batch_idx,(data,targets) in enumerate(train_loader):
    #get data to cuda if possible 
    data = data.cuda()
    targets = targets.cuda()

    scores = model(data)
    loss = criterion(scores,targets)

    #backward
    optimizer.zero_grad()
    loss.backward()


    #gradient_descent or adam-step
    optimizer.step()

# Check the accuracy for the training step
def check_accuracy(loader,model):
  if loader.dataset.train:
    print("Checking accuracy on training data")
  else:
    print("Checking accuracy on test data")

  num_correct = 0
  num_samples = 0
  model.eval()

  with torch.no_grad():
    for x,y in loader:
      x = x.cuda()
      y = y.cuda()

      scores = model(x)
      _,predictions = scores.max(1)
      num_correct += (predictions == y).sum()
      num_samples += predictions.size(0)
    print(f' Got {num_correct}/{num_samples} with accuracy ={float(num_correct)/float(num_samples)*100:.2f} ')
    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)

