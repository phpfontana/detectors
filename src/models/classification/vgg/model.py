import os
import yaml
import torch
import torch.nn as nn
from torchvision.models import VGG16_BN_Weights, vgg16_bn
from PIL import Image

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
        output_size = params[0]
        return nn.AdaptiveAvgPool2d(output_size), in_channels
    else:
        raise ValueError(f"Unsupported layer type: {layer}")

def _make_layers(config, layer, in_channels):
    layers = []
    for layer_config in config[layer]:
        layer_type, params = layer_config
        layer, in_channels = make_layer(layer_type, in_channels, params)
        layers.append(layer)
    return nn.Sequential(*layers), in_channels

def load_yaml_config(config_path):
    try:
        with open(config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration not found at path: {config_path}")
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing YAML file: {exc}")
    
    return config

class VGG(nn.Module):
    def __init__(self, model='vgg16', num_classes=1000, in_channels=3):
        super(VGG, self).__init__()
        self.model = model
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Load configuration
        config_path = os.path.join(os.path.dirname(__file__), 'config', f'{model}.yaml')
        self.config = load_yaml_config(config_path)

        # Update num_classes in config
        for idx, (layer_type, params) in enumerate(self.config['classifier']):
            if layer_type == 'Linear' and params[0] == 'nc':
                self.config['classifier'][idx][1][0] = self.num_classes

        # Build layers
        self.features, in_channels = _make_layers(self.config, 'features', self.in_channels)
        self.avgpool, in_channels = _make_layers(self.config, 'avgpool', in_channels)
        self.classifier, _ = _make_layers(self.config, 'classifier', in_channels*7*7)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def compile(self, optimizer, loss_fn, metrics):
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metrics = metrics

    def fit(self, train_loader, val_loader, epochs):
        for epoch in range(epochs):
            for x, y in train_loader:
                self.optimizer.zero_grad()
                output = self(x)
                loss = self.loss_fn(output, y)
                loss.backward()
                self.optimizer.step()
            
            for x, y in val_loader:
                output = self(x)
                loss = self.loss_fn(output, y)
                self.metrics.update(output, y)

            print(f"Epoch: {epoch}, Loss: {loss.item()}, Metrics: {self.metrics}")

    def evaluate(self, test_loader):
        for x, y in test_loader:
            output = self(x)
            loss = self.loss_fn(output, y)
            self.metrics.update(output, y)
        
        print(f"Test Metrics: {self.metrics}")

def main():
    weights = VGG16_BN_Weights.IMAGENET1K_V1
    img_path = os.path.join(os.path.dirname(__file__), 'test.jpeg')
    img = Image.open(img_path)
    
    model = VGG(model='vgg16', num_classes=1000, in_channels=3)
    model.load_state_dict(weights.get_state_dict())
    print(model)

    torch_model = vgg16_bn(weights=weights)
    print(torch_model)

    transforms = weights.transforms()

    img_transformed = transforms(img)

    x = img_transformed.unsqueeze(0) # (1, 3, 224, 224)
    
    output = model(x)
    torch_output = torch_model(x)
    print(output.shape)
    print(torch_output.shape)

    class_idx = torch.argmax(output, dim=1).item()
    torch_class_idx = torch.argmax(torch_output, dim=1).item()

    print(f"Class index: {class_idx}")
    print(f"Torch class index: {torch_class_idx}")

if __name__ == '__main__':
    main()
