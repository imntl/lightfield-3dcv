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
        self.reverse = reverse
        if isinstance(data, list) and len(data)==6:
            if reverse:
                self.data = data[-3:]
                self.target2 = data[2].unsqueeze(0)
            else: 
                self.data = data[:3]
                self.target2 = data[-1].unsqueeze(0)
            
        else: raise ValueError('Input data should be list of length 6.')
        
        self.target = self.data[-1].unsqueeze(0)
        self.input = [self.data[0].unsqueeze(0), self.data[1].unsqueeze(0)]
        
        self.h = self.target.shape[-2]
        self.w = self.target.shape[-1]
        
        self.full_data = []
        for stack in data:
            self.full_data.append(torch.reshape(stack,(9, 3, self.h, self.w)))
        self.full_data = torch.cat([self.full_data[0], self.full_data[1], self.full_data[2], self.full_data[3], self.full_data[4], self.full_data[5]],-1)
            
            
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
        # calculate loss of interpolated views
        new_views = self.combine(weight)
        syn_loss = F.mse_loss(new_views, self.target).item() 
        syn_loss2 = F.mse_loss(new_views, self.target2).item()
        
        # calculate reconstruction losses
        rec1 = self.model(self.input[0])[0].detach()
        rec_loss1 = F.mse_loss(rec1, self.input[0]).item()
        
        rec2 = self.model(self.input[1])[0].detach()
        rec_loss2 = F.mse_loss(rec2, self.input[1]).item()
        
        return [rec_loss1, rec_loss2, syn_loss, syn_loss2]
    
    def save_synthesized_views(self, path1, path2, path3, weight=0.5):
        
        save_image(self.full_data, path1 + 'full_data.png', nrow=1)
        
        # get interpolated views and reconstructions
        new_views = self.combine(weight)
        new_views = torch.reshape(new_views,(9, 3, self.h, self.w))
        orig_views1 = torch.reshape(self.input[0],(9, 3, self.h, self.w))
        rec1 = self.model(self.input[0])[0].detach()
        rec1 = torch.reshape(rec1,(9, 3, self.h, self.w))
        orig_views2 = torch.reshape(self.input[1],(9, 3, self.h, self.w))
        rec2 = self.model(self.input[1])[0].detach()
        rec2 = torch.reshape(rec2,(9, 3, self.h, self.w))
        target = torch.reshape(self.target,(9, 3, self.h, self.w))
        target2 = torch.reshape(self.target2,(9, 3, self.h, self.w))
        
        # save original and constructed views
        #comparison = torch.cat([orig_views1, rec1, orig_views2, rec2, target, new_views, target2],-1)
        comparison = torch.cat([orig_views2, rec2],-1)
        save_image(comparison, path1  + '.png', nrow=1)
        
       
            
        
        
        # save difference map
        diff = abs(target-new_views)
        diff2 = abs(target2-new_views)
        
        diff = torch.sum(diff, dim=1) #.unsqueeze(1)
        thresh = (torch.min(diff).item()+torch.max(diff).item())/2
        diff2 = torch.sum(diff2, dim=1) #.unsqueeze(1)
        diff = torch.where(diff>thresh,diff,torch.tensor([0.]))
        diff2 = torch.where(diff2>thresh,diff2,torch.tensor([0.]))
        
        
        if not self.reverse:
            img_h1 = orig_views1[0] #lm
            img_h2 = orig_views1[-1] #rm
            diff_map1 = torch.stack((diff[0],diff[0],diff[0]),dim=0) #ol
            diff_map2 = torch.stack((diff[-1],diff[-1],diff[-1]),dim=0) #ur
            diff2_map1 = torch.stack((diff2[0],diff2[0],diff2[0]),dim=0) #or
            diff2_map2 = torch.stack((diff2[-1],diff2[-1],diff2[-1]),dim=0) #ul
        else:
            img_h1 = orig_views1[-1] #lm
            img_h2 = orig_views1[0] #rm
            diff_map1 = torch.stack((diff2[0],diff2[0],diff2[0]),dim=0) #ol
            diff_map2 = torch.stack((diff2[-1],diff2[-1],diff2[-1]),dim=0) #ur
            diff2_map1 = torch.stack((diff[0],diff[0],diff[0]),dim=0) #or
            diff2_map2 = torch.stack((diff[-1],diff[-1],diff[-1]),dim=0) #ul
        img_v1 = orig_views2[0] #om
        img_v2 = orig_views2[4] #m
        img_v3 = orig_views2[-1] #um
        
        map_views = torch.ones((3,3*self.h,3*self.w))
        
        map_views[:,:self.h,:self.w] = diff_map1
        map_views[:,self.h:2*self.h,:self.w] = img_h1
        map_views[:,2*self.h:,:self.w] = diff2_map2
        map_views[:,:self.h,self.w:2*self.w] = img_v1
        map_views[:,self.h:2*self.h,self.w:2*self.w] = img_v2
        map_views[:,2*self.h:,self.w:2*self.w] = img_v3
        map_views[:,:self.h,2*self.w:] = diff2_map1
        map_views[:,self.h:2*self.h,2*self.w:] = img_h2
        map_views[:,2*self.h:,2*self.w:] = diff_map2
        
        
        #color_maps = torch.cat([diff, diff2],-1)
        if self.reverse:
            path2 = path2 + '_rev'
        save_image(map_views, path2 + '.png', nrow=1)
        
        
         # save 3x3 views
        img_h1 = orig_views1[0]
        img_h2 = orig_views1[-1]
        img_v1 = orig_views2[0]
        img_v2 = orig_views2[-1]
        img_new1 = new_views[0]
        img_new2 = new_views[4]
        img_new3 = new_views[-1]
        small_comparison = torch.ones((3,3*self.h,3*self.w))
        
        if not self.reverse:
            small_comparison[:,:self.h,:self.w] = img_new1
            small_comparison[:,self.h:2*self.h,:self.w] = img_h1
            small_comparison[:,:self.h,self.w:2*self.w] = img_v1
            small_comparison[:,self.h:2*self.h,self.w:2*self.w] = img_new2
            small_comparison[:,2*self.h:,self.w:2*self.w] = img_v2
            small_comparison[:,self.h:2*self.h,2*self.w:] = img_h2
            small_comparison[:,2*self.h:,2*self.w:] = img_new3
            
            
        else:
            small_comparison[:,2*self.h:,:self.w] = img_new3
            small_comparison[:,self.h:2*self.h,:self.w] = img_h2
            small_comparison[:,:self.h,self.w:2*self.w] = img_v1
            small_comparison[:,self.h:2*self.h,self.w:2*self.w] = img_new2
            small_comparison[:,2*self.h:,self.w:2*self.w] = img_v2
            small_comparison[:,self.h:2*self.h,2*self.w:] = img_h1
            small_comparison[:,:self.h,2*self.w:] = img_new1
            
        save_image(small_comparison, path3 + '.png', nrow=1)
    
    
class Evaluation:
    def __init__(self,
                 model,
                dataset,
                reverse = False):
        self.model = model
        self.dataset = dataset
        self.reverse = reverse
        self.collect_syn_loss=[]
        self.collect_syn_loss2=[]
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
        # collect losses for whole dataset
        for data in self.dataset:
            view_synthesis = ViewSynthesis(self.model, data, self.reverse)
            loss = view_synthesis.compare(weight)
            self.collect_rec1_loss.append(loss[0])
            self.collect_rec2_loss.append(loss[1])
            self.collect_syn_loss.append(loss[2])
            self.collect_syn_loss2.append(loss[3])
             
    def plot_evaluation(self, weight=0.5):
        '''
        

        Parameters
        ----------
        weight : float, The default is 0.5.

        Returns
        -------
        None.

        '''
        self.evaluate_view_synthesis(weight)
        syn_loss = self.collect_syn_loss
        syn_loss2 = self.collect_syn_loss2
        rec1_loss = self.collect_rec1_loss
        rec2_loss = self.collect_rec2_loss
        
        n = len(syn_loss)
        
        plt.yscale('log')
        plt.plot(np.arange(n), syn_loss, label='View Synthesis')
        plt.plot(np.arange(n), syn_loss2, label='View Synthesis_wrong')
        plt.plot(np.arange(n), rec1_loss, label='Reconstruction hstack')
        plt.plot(np.arange(n), rec2_loss, label='Reconstruction vstack')
        plt.ylabel('mse')
        plt.grid()
        plt.legend()
        plt.title('Comparison of Reconstruction Losses')
        plt.show()
        
    def save_synthesized_views(self, path, nb, weight=0.5):
        '''

        Parameters
        ----------
        path : string, path where images should be saved.
        nb : int. Number of different light fields for which images will be saved
        weight : float. The default is 0.5.

        Returns
        -------
        None.

        '''
        for i in range(nb):
          path_new1 = os.path.join(path, 'synthesized_view' + str(i))
          path_new2 = os.path.join(path, 'color_maps' + str(i))
          path_new3 = os.path.join(path, 'small_comparison' + str(i))
          data = self.dataset[i]
          view_synthesis = ViewSynthesis(self.model, data, self.reverse)
          view_synthesis.save_synthesized_views(path_new1, path_new2, path_new3, weight)
           
           
           
       