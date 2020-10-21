import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from torchvision.utils import save_image

class ViewSynthesis:
    def __init__(self,
                 model,
                 data,
                 reverse = False):
        self.model = model
        if isinstance(data, list) and len(data)==6:
            if reverse:
                self.data = data[-3:]
            else: self.data = data[:3]
        else: raise ValueError('Input data should be list of length 6.')
        
        self.target = self.data[-1].unsqueeze(0)
        self.input = [self.data[0].unsqueeze(0), self.data[1].unsqueeze(0)]
        
        self.h = self.target.shape[-2]
        self.w = self.target.shape[-1]
        
        
    def combine(self,weight=0.5):
        '''
        
        Parameters
        ----------
        weight : float between 0 and 1. The default is 0.5.

        Returns
        -------
        Tensor : New interpolated views
            
        '''
        if weight<0 or weight >1:
            raise ValueError('Weight should be float between 0 and 1.')
        mu1, log_var1 = self.model.encode(self.input[0])
        mu2, log_var2 = self.model.encode(self.input[1])
        z1 = self.model.reparameterize(mu1,log_var1)
        z2 = self.model.reparameterize(mu2,log_var2)
        return self.model.decode(weight*z1+(1-weight)*z2).detach()
    
    def compare(self, weight = 0.5):
        '''
        Measures how well the model interpolates the Diagonal. Also Returns loss
        of reconstructions.

        Parameters
        ----------
        weight : float. The default is 0.5.

        Returns
        -------
        list

        '''
        new_views = self.combine(weight)
        syn_loss = F.mse_loss(new_views, self.target).item()
     
        rec1 = self.model(self.input[0])[0].detach()
        rec_loss1 = F.mse_loss(rec1, self.input[0]).item()
        
        rec2 = self.model(self.input[1])[0].detach()
        rec_loss2 = F.mse_loss(rec2, self.input[1]).item()
        
        return [syn_loss, rec_loss1, rec_loss2]
    
    def save_synthesized_views(self, path, weight=0.5):
        new_views = self.combine(weight)
        new_views = torch.reshape(new_views,(9, 3, self.h, self.w))
        orig_views1 = torch.reshape(self.input[0],(9, 3, self.h, self.w))
        orig_views2 = torch.reshape(self.input[1],(9, 3, self.h, self.w))
        target = torch.reshape(self.target,(9, 3, self.h, self.w))
        comparison = torch.cat([orig_views1, orig_views2, target, new_views],-1)
        save_image(comparison, path  + '.png', nrow=1)
    
    
class Evaluation:
    def __init__(self,
                 model,
                dataset,
                reverse = False):
        self.model = model
        self.dataset = dataset
        self.reverse = reverse
        self.collect_syn_loss=[]
        self.collect_rec1_loss=[]
        self.collect_rec2_loss=[]
     
    def evaluate_view_synthesis(self, weight=0.5):
        '''
        
   
        Parameters
        ----------
        weight : float. The default is 0.5.
   
        Returns
        -------
        None.
   
        '''
        for data in self.dataset:
            view_synthesis = ViewSynthesis(self.model, data, self.reverse)
            loss = view_synthesis.compare(weight)
            self.collect_syn_loss.append(loss[0])
            self.collect_rec1_loss.append(loss[1])
            self.collect_rec2_loss.append(loss[2])
             
    def plot_evaluation(self, weight=0.5):
        self.evaluate_view_synthesis(weight)
        syn_loss = self.collect_syn_loss
        rec1_loss = self.collect_rec1_loss
        rec2_loss = self.collect_rec2_loss
        
        n = len(syn_loss)
        
        plt.yscale('log')
        plt.plot(np.arange(n), syn_loss, label='View Synthesis')
        plt.plot(np.arange(n), rec1_loss, label='Reconstruction hstack')
        plt.plot(np.arange(n), rec2_loss, label='Reconstruction vstack')
        plt.ylabel('mse')
        plt.grid()
        plt.legend()
        plt.title('Comparison of Reconstruction Losses')
        plt.show()
        
    def save_synthesized_views(self, path, nb, weight=0.5):
        for i in range(nb):
          path_new = os.path.join(path, 'synthesized_view' + str(i))
          data = self.dataset[i]
          view_synthesis = ViewSynthesis(self.model, data, self.reverse)
          view_synthesis.save_synthesized_views(path_new, weight)
           
           
           
       