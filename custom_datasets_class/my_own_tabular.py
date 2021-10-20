import torch
class CustomDataset:
    def __init__(self,data,targets):
        self.data = data
        self.targets = targets
    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        current_sample = self.data[idx,:]
        current_target = self.targets[idx]
        return {
            "sample":torch.tensor(current_sample,dtype=torch.float),
            "target":torch.tensor(current_target,dtype=torch.int64)
        }

