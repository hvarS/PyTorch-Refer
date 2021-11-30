import torch 
import tqdm
import time
import torchnet as tnt

class Engine(object):
    """Base Class to contain the fundamentals required in any kind of training instance """
    def __init__(self,state = {}) -> None:
        """Initialize Engine object with a state variable(dict Format)"""
        
        self.state = state
        
        if self._state('use_gpu') is None:
            self.state['use_gpu'] = torch.cuda.is_available()

        if self._state('batch_size') is None:
            self.state['batch_size'] = 16
        
        if self._state('start_epoch') is None:
            self.state['start_epoch'] = 0

        if self._state('max_epochs') is None:
            self.state['max_epochs'] = 100

        if self._state('learning_rate') is None:
            self.state['learning_rate'] = 1e-2
        
        if self._state('optimizer') is None:
            self.state['optimizer'] = torch.optim.Adam(lr = self.state['learning_rate'])

        if self._state('epoch_print_freq') is None:
            self.state['epoch_print_freq'] = 1
        
        if self._state('data_iteration') is None:
            self.state['data_iteration'] = 0

        if self.state('mode') is None:
            self.state['mode'] = 'train'

        ## Data Point , Batch point and Time meters
        # meters
        self.state['meter_loss'] = tnt.meter.AverageValueMeter()
        # time measure
        self.state['batch_time'] = tnt.meter.AverageValueMeter()
        self.state['data_time'] = tnt.meter.AverageValueMeter()
    
    def _state(self,key):
        """Internal Method only to be accesed during initialization"""
        if key in self.state:
            return self.state[key]

    def on_forward(self,model)-> None:
        self.state['output'] = model(self.state['input'])

    def on_start_batch(self):
        pass

    def on_end_batch(self):
        self.state['loss_batch'] = self.state['loss']
        self.state['meter_loss'].add(self.state['loss_batch'])

    def on_start_epoch(self):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()
    
    def on_end_epoch(self):
        loss = self.state['meter_loss'].value()
        if self.state['mode'] == 'train':
                print('Epoch: [{0}]\t'
                      'Loss {loss:.4f}'.format(self.state['epoch'], loss=loss))
        elif self.state['mode'] == 'val':
            print('Validation: \t Loss {loss:.4f}'.format(loss=loss))
        else:
            print('Test: \t Loss {loss:.4f}'.format(loss=loss))
        
        return loss

    def train(self,train_loader,model,criterion,optimizer):
        
        self.state['mode'] = 'train'
        train_start = time.time()

        self.on_start_epoch()

        for i,(input,target) in enumerate(train_loader):
            
            self.state['data_iteration'] = i
            self.state['data_iteration_time'] = time.time()-train_start

            self.state['input'] = input
            self.state['target'] = target

            if self.state['use_gpu']:
                self.state['input'] = self.state['input'].cuda()
                self.state['target'] = self.state['output'].cuda()

            self.on_start_batch()
            
            self.on_forward(model)
            self.state['loss'] = criterion(self.state['output'],self.state['target'])

            optimizer.zero_grad()
            self.state['loss'].backward()
            optimizer.step()

            self.on_end_batch()
        
        self.on_end_epoch()

    def validate(self,val_loader,model,criterion,optimizer):
        
        self.state['mode'] = 'val'

        self.on_start_epoch()
        
        train_start = time.time()
        
        for i,(input,target) in enumerate(val_loader):
            
            self.state['data_iteration'] = i
            self.state['data_iteration_time'] = time.time()-train_start

            self.state['input'] = input
            self.state['target'] = target

            if self.state['use_gpu']:
                self.state['input'] = self.state['input'].cuda()
                self.state['target'] = self.state['output'].cuda()
            
            self.on_start_batch()

            self.on_forward(model)
            self.state['loss'] = criterion(self.state['output'],self.state['target'])

            self.on_end_batch()

        return self.on_end_epoch()


    def learn(self,model,train_dataset,val_dataset,criterion,optimizer,train = True):

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.state['batch_size'], shuffle=True)

        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=self.state['batch_size'], shuffle=False)

        if self.state['use_gpu']:
            criterion = criterion.cuda()
        
        if train :
            for epoch in range(self.state['start_epoch'],self.state['max_epochs']):
                self.state['current_epoch'] = epoch
                self.train(train_loader=train_loader,model=model,criterion=criterion,optimizer=optimizer)
                epoch_performance = self.validate(val_loader=val_loader,model = model,criterion=criterion,optimizer=optimizer)
                self.state['best_score'] = max(self.state['best_score'],epoch_performance)

        return self.state['best_score']




            


