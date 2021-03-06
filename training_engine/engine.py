import torch 
from tqdm import tqdm
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
        
        if self._state('print_freq') is None:
            self.state['print_freq'] = 0
        
        if self._state('use_pb') is None:
            self.state['use_pb'] = True

        if self._state('data_iteration') is None:
            self.state['data_iteration'] = 0

        if self._state('mode') is None:
            self.state['mode'] = 'train'

        self.state['best_score'] = 0

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


    def on_start_epoch(self):
        self.state['meter_loss'].reset()
        self.state['batch_time'].reset()
        self.state['data_time'].reset()
    
    def on_end_epoch(self):
        loss = self.state['meter_loss'].value()[0]
        loss_value = self.state['loss']
        epoch_num = self.state['current_epoch']
        if self.state['mode'] == 'train':
                print(f'Epoch: [{epoch_num}] ::: Loss {loss_value}')
        elif self.state['mode'] == 'val':
            print(f'Validation: \t Loss {loss_value}')
        else:
            print(f'Test: \t Loss {loss_value}')
        
        return loss
    
    def on_start_batch(self):
        pass

    def on_end_batch(self,data_loader):
        self.state['loss_batch'] = self.state['loss']
        self.state['meter_loss'].add(self.state['loss_batch'].data)

        if self.state['print_freq'] != 0 and self.state['iteration'] % self.state['print_freq'] == 0:
            loss = self.state['meter_loss'].value()
            batch_time = self.state['batch_time'].value()
            data_time = self.state['data_time'].value()
            if self.state['mode'] == 'train':
                print('Epoch: [{0}]'
                      'Training : [{1}/{2}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['current_epoch'], self.state['iteration'], len(data_loader),
                    batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))
            elif self.state['mode'] == 'val':
                print('Validation: [{0}/{1}]\t'
                      'Time {batch_time_current:.3f} ({batch_time:.3f})\t'
                      'Data {data_time_current:.3f} ({data_time:.3f})\t'
                      'Loss {loss_current:.4f} ({loss:.4f})'.format(
                    self.state['iteration'], len(data_loader), batch_time_current=self.state['batch_time_current'],
                    batch_time=batch_time, data_time_current=self.state['data_time_batch'],
                    data_time=data_time, loss_current=self.state['loss_batch'], loss=loss))

    def train(self,train_loader,model,criterion,optimizer):
        
        self.state['mode'] = 'train'
        end = time.time()
        model.train()
        self.on_start_epoch()

        if self.state['use_pb']:
            train_loader = tqdm(train_loader, desc='Training')

        for i,(input,target) in enumerate(train_loader):
            
            self.state['iteration'] = i

            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

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
            
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()

            self.on_end_batch(train_loader)
        
        self.on_end_epoch() 

    def validate(self,val_loader,model,criterion,optimizer):
        
        self.state['mode'] = 'val'
        model.eval()
        self.on_start_epoch()

        if self.state['use_pb']:
            val_loader = tqdm(val_loader, desc='Validation')

        end = time.time()

        for i,(input,target) in enumerate(val_loader):
            
            self.state['iteration'] = i

            self.state['data_time_batch'] = time.time() - end
            self.state['data_time'].add(self.state['data_time_batch'])

            self.state['input'] = input
            self.state['target'] = target

            if self.state['use_gpu']:
                self.state['input'] = self.state['input'].cuda()
                self.state['target'] = self.state['output'].cuda()
            
            self.on_start_batch()

            self.on_forward(model)
            self.state['loss'] = criterion(self.state['output'],self.state['target'])

            # measure elapsed time
            self.state['batch_time_current'] = time.time() - end
            self.state['batch_time'].add(self.state['batch_time_current'])
            end = time.time()

            self.on_end_batch(val_loader)

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

    def learn_tune(self,trial,model,train_dataset,val_dataset,criterion,optimizer,train = True):

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
                trial.report(epoch_performance, epoch)
                self.state['best_score'] = max(self.state['best_score'],epoch_performance)

        return self.state['best_score']






            


