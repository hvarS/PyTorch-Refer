from engine import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import optuna
from optuna.trial import TrialState
import torch.optim as optim

class CNN(nn.Module):
  def __init__(self):
    super(CNN,self).__init__()
    self.fc1 = nn.Linear(3*224*224,1)
    self.softmax = nn.Softmax(dim=0)

  def forward(self,x):
    x = x.view(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3])
    x = self.fc1(x)
    return self.softmax(x)

class CustomDataset(data.Dataset):
    def __init__(self):

        self.images = torch.randn(300,3,224,224)
        self.regress = torch.randn(300,1)

        
    def __getitem__(self, index):
        img = self.images[index]
        target = self.regress[index]
        return (img, target)

    def __len__(self):
        return len(self.images)


def objective(trial):

    # Generate the model.
    model = CNN()

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    engine = Engine()
    train_dataset = CustomDataset()
    val_dataset = CustomDataset()

    criterion = nn.L1Loss()
    acc = engine.learn(model,train_dataset,val_dataset,criterion,optimizer)

    # Handle pruning based on the intermediate value.
    if trial.should_prune():
      raise optuna.exceptions.TrialPruned()

    return acc


    



state = {'learning_rate':1e-2}

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=5, timeout=600)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    