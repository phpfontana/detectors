from pprint import pprint
import torch
import torchvision
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from detectors.models.backbones import resnet50
from detectors.models.detection import faster_rcnn


def main():
    # load weights
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    
    # load backbone
    backbone = resnet50()

    # load backbone weights
    # backbone.load_state_dict(weights.get_state_dict())

    # load mask rcnn model
    model = faster_rcnn(backbone, 1000)
    pprint(model)
    
    # set the model to evaluation mode to avoid needing targets
    model.eval()

    # warmup pass
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        out = model(x)

    # draw bounding boxes
    boxes = out[0]['boxes']
    labels = out[0]['labels']
    scores = out[0]['scores']

    print(boxes)
    print(labels)
    print(scores)

if __name__ == '__main__':
    main()
