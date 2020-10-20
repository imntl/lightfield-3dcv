import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

class Trainer(object):
    def __init__(self, model,
                 optimizer, loss_function,
                 loader_train, loader_val,
                 dtype, device, **in_params):
        """
        :param model: PyTorch model of the neural network

        :param optimizer: PyTorch optimizer

        :param print_every: How often should we print the loss during training
        """
        # Create attributes:
        self.device = device
        self.model = model.to(device=self.device)  # move the model parameters to CPU/GPU
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.loader_train = loader_train
        self.loader_val = loader_val
        self.print_every = in_params["print_every"]
        self.dtype = dtype
        self.batch_size = in_params["batch_size"]
        self.input_size = in_params["input_size"]
        self.path = in_params["path"]
        self.collect_test_loss=[]
        self.collect_train_loss=[]


    def train_model(self, epoch):
        """
        - epoch: An integer giving the epoch
        """
        train_loss = 0
        self.model.train()  # put model to training mode
        for t, input in enumerate(self.loader_train):
            
            input = input.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
            
            # do a step in training
            args = self.model(input)
            loss = self.loss_function(*args,**{'M_N':1e-4*self.batch_size/len(self.loader_train)})['loss']
            self.optimizer.zero_grad()
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),1)
            train_loss += loss.item() # accumulate for average loss
            self.optimizer.step()

            # print loss
            if t % self.print_every == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, t * len(args[1]), len(self.loader_train.dataset),
                    100. * t / len(self.loader_train),
                    loss.item() / len(args[1])))
        # print average loss
        self.collect_train_loss.append(train_loss / len(self.loader_train.dataset))
        print('====> Epoch: {} Average loss: {:.6f}'.format(
              epoch, train_loss / len(self.loader_train.dataset)))

    def test_model(self, epoch):
        self.model.eval() # Put model to evaluation mode
        test_loss = 0.

        with torch.no_grad():
            # During validation, we accumulate these values across the whole dataset and then average at the end:
            for i, input in enumerate(self.loader_val):
                input = input.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
       
                # compute loss and accumulate
                args = self.model(input)
                test_loss += self.loss_function(*args,**{'M_N':1e-4*self.batch_size/len(self.loader_val)})['loss'].item()
                if i == 0 and epoch%10 == 0:
                    n = min(args[1].size(0), 6)
                    comparison = torch.cat([args[1][:n],
                                          args[0].view(self.batch_size, self.model.out_channels, self.input_size, self.input_size)[:n]])
                    save_image(comparison.cpu(),
                             self.path + '/reconstruction_' + str(epoch) + '.png', nrow=n)

        # print average loss
        test_loss /= len(self.loader_val.dataset)
        self.collect_test_loss.append(test_loss)
        print('====> Test set loss: {:.6f}'.format(test_loss))
        
    def train_and_test(self, epochs, path1, path2):

        for e in range(1,epochs+1):
            self.train_model(e)
            self.test_model(e)
            if e%10 == 0:
                with torch.no_grad():
                    sample = self.model.sample(64, device)
                    save_image(sample[:, :3, :, :], self.path + '/sample_' + str(e) + '.png')
        # Print and save loss plots
        with torch.no_grad():
          trloss = self.collect_train_loss
          teloss = self.collect_test_loss
          print(np.min(trloss),np.min(teloss))

          n1 = len(trloss)
          n2 = len(teloss)

          plt.yscale('log')
          plt.plot(np.arange(n1), trloss)
          plt.grid()
          plt.title('average train loss')
          plt.savefig(path1)
          plt.show()

          plt.yscale('log')
          plt.plot(np.arange(n2), teloss)
          plt.grid()
          plt.title('test loss')
          plt.savefig(path2)
          plt.show()