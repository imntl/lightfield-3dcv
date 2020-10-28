import numpy as np
import random
import skimage.color as color
import torch

class Brightness:
    '''
    Randomly change Brightness
        
    Parameters
    ----------
    probability : float between 0 and 1. The default is 0.5.

    '''
    def __init__(self, probability=0.5, level=0.9):
  
        assert probability >= 0 and probability <= 1
        assert isinstance(level, float)
        self.probability = probability
        self.level = level

    def __call__(self, data):
      
        if np.random.rand() <= self.probability:
          alpha = random.uniform(-self.level, self.level) + 1.0
          data = data*alpha

        return data



class Contrast:
    '''
    Randomly change Contrast
    
    Parameters
    ----------
    probability : float between 0 and 1. The default is 0.5.

    '''

    def __init__(self, probability = 0.5, level=0.9):
    
        assert probability >= 0 and probability <= 1
        assert isinstance(level, float)
        self.probability = probability
        self.level = level

    def __call__(self, data):
      
        if np.random.rand() <= self.probability:
          alpha = random.uniform(-self.level, self.level) + 1.0
          mean = data.mean()

          data = data * alpha + mean * (1.0 - alpha)

        return data


class Noise:
    '''
    Add random Gaussian noise
    
    Parameters
    ----------
    probability : float between 0 and 1. The default is 0.5.

    '''

    def __init__(self, probability = 0.5, stdev=0.01):
      
        assert probability >= 0 and probability <= 1
        assert isinstance(stdev, float)
        self.probability = probability
        self.stdev = stdev

    def __call__(self, data):
       
        if np.random.rand() <= self.probability:
            noise = np.random.normal(scale=self.stdev, size=data.shape)
            data += noise
            data = np.clip(data, 0, 1)

        return data


class RandomCrop:
    '''
    Crop patches randomly to a given size
    
    Parameters
    ----------
    size : height and width of quadratic patch

    '''

    def __init__(self, size, pad=0):
       
        assert isinstance(size, int) or (
            isinstance(size, tuple) and len(size) == 2)

        self.size = size
        if isinstance(size, int):
            self.size = (size, size)

        assert isinstance(pad, int)
        self.pad = pad

    def __call__(self, data):
       
        h = data[0].shape[-2]
        w = data[0].shape[-1]

        assert h > self.size[0]
        assert w > self.size[1]

        y = random.randint(self.pad, h - self.size[0] - self.pad)
        x = random.randint(self.pad, w - self.size[1] - self.pad)

        return data[:, y:y+self.size[0], x:x+self.size[1]]
    
    
class ToTensor(object):
    '''
    Converts a numpy.ndarray (C x H x W) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    '''

    def __call__(self, pic):
        
        img = torch.from_numpy(pic)
        if torch.max(img)>1:
          img = img.float().div(255)
        else: img = img.float()
        return img
