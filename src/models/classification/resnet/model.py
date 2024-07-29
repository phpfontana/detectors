import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels*4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], 64, 1)
        self.layer2 = self._make_layer(block, layers[1], 128, 2)
        self.layer3 = self._make_layer(block, layers[2], 256, 2)
        self.layer4 = self._make_layer(block, layers[3], 512, 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * (4 if block == BottleNeck else 1), num_classes)

    def _make_layer(self, block, blocks, out_channels, stride):
        downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * (4 if block == BottleNeck else 1):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * (4 if block == BottleNeck else 1), kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * (4 if block == BottleNeck else 1))
            )
        
        layers.append(block(self.in_channels, out_channels, stride, downsample) if block == BottleNeck else block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * (4 if block == BottleNeck else 1)

        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)

        return out

def ResNet18(in_channels, num_classes):
    return ResNet(ResidualBlock, [2, 2, 2, 2], in_channels, num_classes)

def ResNet34(in_channels, num_classes):
    return ResNet(ResidualBlock, [3, 4, 6, 3], in_channels, num_classes)

def ResNet50(in_channels, num_classes):
    return ResNet(BottleNeck, [3, 4, 6, 3], in_channels, num_classes)

def ResNet101(in_channels, num_classes):
    return ResNet(BottleNeck, [3, 4, 23, 3], in_channels, num_classes)

def ResNet152(in_channels, num_classes):
    return ResNet(BottleNeck, [3, 8, 36, 3], in_channels, num_classes)


def main():
    resnet18 = ResNet18(3, 1000)
    resnet34 = ResNet34(3, 1000)
    resnet50 = ResNet50(3, 1000)
    resnet101 = ResNet101(3, 1000)
    resnet152 = ResNet152(3, 1000)

    resnet18_weights = ResNet18_Weights.IMAGENET1K_V1
    resnet34_weights = ResNet34_Weights.IMAGENET1K_V1
    resnet50_weights = ResNet50_Weights.IMAGENET1K_V1
    resnet101_weights = ResNet101_Weights.IMAGENET1K_V1
    resnet152_weights = ResNet152_Weights.IMAGENET1K_V1

    input = torch.randn(1, 3, 224, 224)

    models = [resnet18, resnet34, resnet50, resnet101, resnet152]
    weights = [resnet18_weights, resnet34_weights, resnet50_weights, resnet101_weights, resnet152_weights]

    for model, weights in zip(models, weights):
        model.load_state_dict(weights.get_state_dict())
        print(model(input).shape)


if __name__ == '__main__':
    main()
