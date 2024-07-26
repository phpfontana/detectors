import torch
import torch.nn as nn
import yaml
from layers import *
from torchvision.models import vgg16_bn, VGG16_BN_Weights

def create_conv_layer(layer_type, in_channels, params):
    if layer_type == 'Conv':
        out_channels, kernel_size, stride, padding = params
        return ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), out_channels
    elif layer_type == 'MaxPool':
        kernel_size, stride, padding = params
        return MaxPoolBlock(kernel_size=kernel_size, stride=stride, padding=padding), in_channels

class VGG(nn.Module):
    def __init__(self, weights=None, in_channels=3, num_classes=1000, config=None, model='vgg16'):
        super(VGG, self).__init__()
        self.in_channels = in_channels # input_shape: (batch_size, channels, height, width
        self.num_classes = num_classes
        self.history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }
        
        self.features = self.create_conv_layers(config, model)

        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=4096),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes),
            nn.Softmax(dim=1)
        )
        
    def create_conv_layers(self, config, model):
        layers = []
        in_channels = self.in_channels

        for layer_config in config[model]:
            layer_type, params = layer_config
            layer, in_channels = create_conv_layer(layer_type, in_channels, params)
            layers.append(layer)

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x
    
    def compile(self, optimizer, loss_fn, metrics=None):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics
    
    def fit(self, train_loader, epochs):
        pass

    def summary(self):
        pass

    def evaluate(self, val_loader):
        pass

    def predict(self, x):
        return self.forward(x)

def main():
    pass

if __name__ == '__main__':
    main()