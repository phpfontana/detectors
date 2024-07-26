import os
import yaml
import torch
import torch.nn as nn
from torchvision.models import VGG16_BN_Weights

def make_layer(layer, in_channels, params):
    if layer == 'Conv':
        out_channels, kernel_size, stride, padding = params
        return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding), out_channels
    elif layer == 'MaxPool':
        kernel_size, stride, padding = params
        return nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding), in_channels
    elif layer == 'ReLU':
        return nn.ReLU(inplace=True), in_channels
    elif layer == 'SiLU':
        return nn.SiLU(inplace=True), in_channels
    elif layer == 'Dropout':
        p = params[0]
        return nn.Dropout(p=p), in_channels
    elif layer == 'Linear':
        out_features = params[0]
        return nn.Linear(in_features=in_channels, out_features=out_features), out_features
    elif layer == 'Softmax':
        return nn.Softmax(dim=1), in_channels
    elif layer == 'BatchNorm':
        num_features = params[0]
        return nn.BatchNorm2d(num_features), in_channels
    elif layer == 'AdaptiveAvgPool':
        output_size = params
        return nn.AdaptiveAvgPool2d(output_size), in_channels
    else:
        raise ValueError(f"Unsupported layer type: {layer}")

class VGG(nn.Module):
    def __init__(self, model='vgg16', num_classes=1000, in_channels=3):
        super(VGG, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        config_path = os.path.join(os.path.dirname(__file__), 'config', f'{model}.yaml')
        
        try:
            with open(config_path, 'r') as file:
                self.config = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file for model '{model}' not found at path: {config_path}")
        except yaml.YAMLError as exc:
            raise ValueError(f"Error parsing YAML file for model '{model}': {exc}")

        self.features = self.make_layers(self.config, 'features')
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512*7*7, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes)
        )

    def make_layers(self, config, layer):
        layers = []
        in_channels = self.in_channels

        for layer_config in config[layer]:
            layer_type, params = layer_config
            layer, in_channels = make_layer(layer_type, in_channels, params)
            layers.append(layer)
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

    def compile(self):
        pass

    def fit(self):
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass

def main():
    model = VGG(model='vgg16', num_classes=1000, in_channels=3)
    print(model)

    weights = VGG16_BN_Weights.IMAGENET1K_V1
    model.load_state_dict(weights.get_state_dict())
        
    x = torch.randn(1, 3, 224, 224)  # Example input
    output = model(x)
    print(output.shape)

if __name__ == '__main__':
    main()