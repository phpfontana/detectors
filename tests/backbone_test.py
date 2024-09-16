import torch
import torchvision
import torch.nn as nn
from typing import Any, Callable, List, Optional, Type, Union
from torchvision.models import ResNet50_Weights
from pprint import pprint

def main():
    model = torchvision.models.resnet18()
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model.load_state_dict(weights.get_state_dict())

    backbone = nn.Sequential(*list(model.children())[:-1])  # Remove the classifier

    pprint(backbone)
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        out = backbone(x)

    print(out.shape)

if __name__ == '__main__':
    main()