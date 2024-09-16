import torch
import torchvision
from detectors.models.backbones import resnet50
from detectors.models.detection import faster_rcnn


def main():
    # load weights
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    
    # load backbone
    backbone = resnet50()

    # load backbone weights
    backbone.load_state_dict(weights.get_state_dict())

    # load mask rcnn model
    model = faster_rcnn(backbone, 1000)
    
    # set the model to evaluation mode to avoid needing targets
    model.eval()

    # warmup pass
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        out = model(x)

    print(out)

if __name__ == '__main__':
    main()
