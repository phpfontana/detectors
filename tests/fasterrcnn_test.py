import torch
from detectors.models.backbones import resnet50
from detectors.models.detection import faster_rcnn


def main():
    backbone = resnet50()
    model = faster_rcnn(backbone, 1000)
    
    # Set the model to evaluation mode to avoid needing targets
    model.eval()

    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        out = model(x)

    # The output is a dictionary, typically with 'boxes', 'labels', and 'scores'
    print(out)

if __name__ == '__main__':
    main()
