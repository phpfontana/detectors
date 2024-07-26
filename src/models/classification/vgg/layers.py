import torch
import torch.nn as nn
import yaml


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class MaxPoolBlock(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(MaxPoolBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.pool(x)
    
class DropoutBlock(nn.Module):
    def __init__(self, p):
        super(DropoutBlock, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, x):
        return self.dropout(x)

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, last_layer=False):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.act = nn.Softmax(dim=1) if last_layer else nn.SiLU()

    def forward(self, x):
        return self.act(self.fc(x))
    
