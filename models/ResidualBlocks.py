import torch
from torch import nn

class ResidualBlockEncoder(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 **kwargs) -> None:
        super(ResidualBlockEncoder, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                              kernel_size= 3, stride= stride, padding  = 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels,
                              kernel_size= 3, stride= stride, padding  = 1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.norm(x)
        if self.stride != 1 or self.in_channels != self.out_channels:
            temp = self.conv2(x)
        else: temp = 1*x
        x = self.relu(self.conv1(x))
        x = x + temp
        return x
    
    
class ResidualBlockDecoder(nn.Module):
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 **kwargs) -> None:
        super(ResidualBlockDecoder, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        self.conv1 = nn.ConvTranspose2d(in_channels, 
                                        out_channels,
                                        kernel_size= 3, 
                                        stride= stride, 
                                        padding  = 1,
                                        output_padding=stride-1)
        
        self.conv2 = nn.ConvTranspose2d(in_channels, 
                                        out_channels,
                                        kernel_size= 3, 
                                        stride= stride, 
                                        padding  = 1,
                                        output_padding=stride-1)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.norm(x)
        if self.stride != 1 or self.in_channels != self.out_channels:
            temp = self.conv2(x)
        else: temp = 1*x
        x = self.relu(self.conv1(x))
        x = x + temp
        return x