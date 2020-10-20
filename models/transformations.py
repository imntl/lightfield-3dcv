import numpy as np
import random
import skimage.color as color

class Brightness:
    """
    Randomly change Brightness
    """
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

class Color_jitter:
    """ 
    Randomly jitter the saturation, hue and brightness of the image.
    """
    def __init__(self, probability=0.5, level=0.9):
  
        assert probability >= 0 and probability <= 1
        self.probability = probability

    def __call__(self, data):

        if np.random.rand() <= self.probability:

          # skimage expects WHC instead of CHW
          data = data.transpose((1, 2, 0))

          # transform image to hsv color space to apply jitter
          n, m, l = data.shape
          data = data.reshape((n,m*(l//3),3))
          data = color.rgb2hsv(data)

          # compute jitter factors in range 0.66 - 1.5  
          jitter_factors = 1.5 * np.random.rand(3)
          jitter_factors = np.clip(jitter_factors, 0.66, 1.5)

          # apply the jitter factors, making sure to stay in correct value range
          data *= jitter_factors
          data = np.clip(data, 0, 1)

          # transform back to rgb and CHW    
          data = color.rgb2hsv(data)
            
          data = data.reshape((n,m,l))
          data = data.transpose((2, 0, 1))
        return data


class Contrast:
    """
    Randomly change Contrast
    """

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
    """
    Add random Gaussian noise
    """

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
    """
    Crop patches randomly to a given size
    """

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

