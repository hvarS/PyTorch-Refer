import torch
from torch.nn import functional as F
from torch import nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule


class mnistModel(LightningModule):
    def __init__(self):
        super().__init__()

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
    def training_step(self, batch,batch_idx):
        x,y = batch
        logits = self(x)
        loss = F.nll_loss(logits,y)

        #Log data
        self.log("_nll loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr = 1e-3)

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
    

model = mnistModel()
dataModule = mnistData()
trainer = Trainer()
trainer.fit(model,datamodule=dataModule)
trainer.test(model,dataModule)
