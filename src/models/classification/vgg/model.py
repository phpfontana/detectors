import torch
import torch.nn as nn
import yaml
from layers import *

# Helper function to create a layer from the config
def create_layer(layer_type, in_channels, params):
    if layer_type == 'Conv':
        out_channels, kernel_size, stride, padding = params
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding), out_channels
    elif layer_type == 'MaxPool':
        kernel_size, stride, padding = params
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding), in_channels
    elif layer_type == 'Linear':
        out_features = params[0]
        return nn.Linear(in_features=in_channels, out_features=out_features), out_features
    elif layer_type == 'ReLU':
        return nn.ReLU(inplace=True), in_channels
    elif layer_type == 'Dropout':
        p = params[0]
        return nn.Dropout(p=p), in_channels
    elif layer_type == 'Softmax':
        return nn.Softmax(dim=1), in_channels
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

# VGG16 Model
class VGG(nn.Module):
    def __init__(self, config):
        super(VGG, self).__init__()
        self.layers, self.in_features = self._create_layers(config)
        self.num_classes = config['nc']
        self.fc_layers = self._create_fc_layers(config['head'], self.num_classes)

    def _create_layers(self, config):
        layers = []
        in_channels = 3  # Initial input channels (e.g., RGB image)

        # Backbone layers
        for layer_config in config['backbone']:
            idx, repeat, layer_type, params = layer_config
            layer, in_channels = create_layer(layer_type, in_channels, params)
            layers.append(layer)

        # Flatten layer before fully connected layers
        layers.append(nn.Flatten())
        in_features = in_channels * 7 * 7  # Assuming input image size is 224x224 and feature maps are downsampled by 32

        return nn.Sequential(*layers), in_features

    def _create_fc_layers(self, head_config, num_classes):
        layers = []
        in_features = self.in_features  # Use in_features calculated from _create_layers
        for layer_config in head_config:
            idx, repeat, layer_type, params = layer_config
            if layer_type == 'Linear' and params[0] == 'nc':
                params[0] = num_classes  # Convert 'nc' to actual number of classes
            print(f"Creating {layer_type} layer with in_features: {in_features}, params: {params}")  # Debug statement
            layer, in_features = create_layer(layer_type, in_features, params)
            layers.append(layer)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        x = self.fc_layers(x)
        return x

# Load config
with open('/Users/fontana/Desktop/research/deep-learning-cv/src/models/classification/vgg/vgg16.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model
model = VGG(config)
print(model)

# Example usage
x = torch.randn(1, 3, 224, 224)  # Example input
output = model(x)
print(output.shape)
