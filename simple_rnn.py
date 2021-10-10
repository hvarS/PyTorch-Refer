from typing import Sequence
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, dataloader
import torchvision.datasets as datasets
import torchvision.transforms as transforms 

#Creating a fully connected network 

#For Changing to LSTM , GRU just change to nn.LSTM or nn.GRU instead of nn.RNN

#Input Form is like : (batches,sequence_length,feature_size)

class RNN(nn.Module):
    def __init__(self,input_size,num_classes,hidden_size,num_layers):
        '''
        input_size: Input Sentence Words ; For ex 28 words in a sentence
        hidden_size: Hidden size of RNN
        num_layers: Stack Size of RNN
        num_classes: Final Number of Classes that you want to classify into
        '''
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first = True)
        self.fc = nn.Linear(hidden_size*sequence_length,num_classes)

    def forward(self,x):
        #x[0] : Batch_size
        h_0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(device)
        x ,_ = self.rnn(x,h_0) #Input to RNN = input tensor, Hidden State 0 (Initial) , Outputs : output, h_n
        out = x.reshape(x.shape[0],-1)
        out = self.fc(out)
        return out





device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#HyperParameters
input_size = 28            #28 Sentences
sequence_length = 28       #28 words
num_layers = 2             #RNN 
num_classes = 10
hidden_size = 250
learning_rate = 0.001
batch_size = 64
num_epochs = 1




#Load Dataset
train_data = datasets.MNIST(root='datasets/',train=True,transform = transforms.ToTensor(),download=True)
train_loader = DataLoader(dataset =train_data,batch_size=batch_size ,shuffle=True)

test_data = datasets.MNIST(root='datasets/',train=False,transform = transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset =test_data,batch_size=batch_size ,shuffle=True)

#Define Model
model = RNN(input_size=input_size,num_classes=num_classes,hidden_size=hidden_size,num_layers=num_layers).to(device=device)

#Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = learning_rate)

#Train Network
#Input format of tensor will look like = (64,1,28,28)
for epoch in range(num_epochs):
    for batch_idx,(data,targets) in enumerate(train_loader):
        
        #Data Shape = (64,1,28,28)

        data = data.to(device).squeeze(1)
        targets = targets.to(device)

        #Data Shape = (64,28,28) 

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
      x = x.squeeze(1)

      scores = model(x)
      _,predictions = scores.max(1)
      num_correct += (predictions == y).sum()
      num_samples += predictions.size(0)
    print(f' Got {num_correct}/{num_samples} with accuracy ={float(num_correct)/float(num_samples)*100:.2f} ')
    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)



