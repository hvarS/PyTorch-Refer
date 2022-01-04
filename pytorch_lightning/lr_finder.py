import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F

class LitModel(LightningModule):
    def __init__(self,learning_rate):
        super().__init__()

        self.lr = learning_rate
        self.l1 = nn.Linear(28*28,128)
        self.l2 = nn.Linear(128,256)
        self.l3 = nn.Linear(256,10)

    def forward(self,x):

        batch_size,channels,height,width = x.size()

        x = x.view(batch_size,-1)
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        x = self.l3(x)
        
        x = F.log_softmax(x,dim=1)
        return x
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))
    def training_step(self, batch,batch_idx):
        x,y = batch
        logits = self(x)
        loss = F.nll_loss(logits,y)

        #Log data
        self.log("_nll loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

class mnistData(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
    def train_dataloader(self):
        data = torch.randn(100,1,28,28)
        ground_truth = torch.randint(0,1,size=(100,))
        return DataLoader(list(zip(data,ground_truth)),batch_size=64,num_workers=4)
    def val_dataloader(self):
        data = torch.randn(100,1,28,28)
        ground_truth = torch.randint(0,1,size=(100,))
        return DataLoader(list(zip(data,ground_truth)),batch_size=64,num_workers=4)
    def test_dataloader(self):
        data = torch.randn(100,1,28,28)
        ground_truth = torch.randint(0,1,size=(100,))
        return DataLoader(list(zip(data,ground_truth)),batch_size=64,num_workers=4)
    

trainer = Trainer(auto_lr_find = True)
model = LitModel(1)
dataModule = mnistData()
trainer.tune(model,datamodule=dataModule)