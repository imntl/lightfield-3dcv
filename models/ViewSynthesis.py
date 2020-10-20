import torch
from torch import nn
from torch.nn import functional as F

class ViewSynthesis(object):
    def __init__(self,
                 model,
                 data):
        self.model = model
        if isinstance(data, list):
            self.target = data[-1]
            self.input = data[:-1]
        else: raise ValueError('Input data should be list object.')
        
    def combine(self,weight=0.5):
        if weight<0 or weight >0:
            raise ValueError('Weight should be float between 0 and 1.')
        mu1, log_var1 = self.model.encode(self.input[0])
        mu2, log_var2 = self.model.encode(self.input[1])
        z1 = self.model.reparameterize(mu1,log_var1)
        z2 = self.model.reparameterize(mu2,log_var2)
        return self.model.decode(weight*z1+(1-weight)*z2).detach()
    def compare(self, weight = 0.5):
        res = self.combine(weight)
        err = F.mse_loss(res, self.target)
        return err
        