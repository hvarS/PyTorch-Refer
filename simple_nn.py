import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, dataloader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 

#Creating a fully connected network 

class NN(nn.Module):
    def __init__(self,input_size,num_classes):
        '''
        input_size: Flattened Image size ; For ex 28*28 greyscale image to 
                                            768
        num_classes: Final Number of Classes that you want to classify into
        '''
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,num_classes)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#HyperParameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 1




#Load Dataset
train_data = datasets.MNIST(root='datasets/',train=True,transform = transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset =train_data,batch_size=batch_size ,shuffle=True)

test_data = datasets.MNIST(root='datasets/',train=False,transform = transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset =test_data,batch_size=batch_size ,shuffle=True)

#Define Model
model = NN(input_size=input_size,num_classes=num_classes).to(device=device)

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

#Train Network
#Input format of tensor will look like = (64,1,28,28)
for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        #Data Shape = (64,1,28,28)

        #Flatten the images
        data = data.reshape(data.shape[0],-1) #Data Shap3 = (64,784)

        #Forward 
        logits = model(data)
        loss = criterion(logits,targets)

        #Backward
        optimizer.zero_grad()
        loss.backward()

        #Gradient Descent
        optimizer.step()
    print(f'Epoch {epoch} Completed')


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
      x = x.to(device = device)
      y = y.to(device = device)
      x = x.reshape(x.shape[0],-1)

      scores = model(x)
      _,predictions = scores.max(1)
      num_correct += (predictions == y).sum()
      num_samples += predictions.size(0)
    print(f' Got {num_correct}/{num_samples} with accuracy ={float(num_correct)/float(num_samples)*100:.2f} ')
    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)



