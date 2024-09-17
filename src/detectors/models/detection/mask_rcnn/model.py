import torch
import torchvision
from typing import Tuple
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def mask_rcnn(backbone: torch.nn.Module, num_classes: int = 91, min_size: int = 800, max_size: int = 1333, anchor_sizes: Tuple[Tuple[int, ...], ...] = ((32, 64, 128, 256, 512),), aspect_ratios: Tuple[Tuple[float, ...], ...] = ((0.5, 1.0, 2.0),)) -> MaskRCNN:
    """
    Create a Mask R-CNN model with a custom backbone.

    Args:
        backbone (torch.nn.Module): The backbone network to use in the Mask R-CNN model.
        num_classes (int): Number of output classes of the model (including the background).
        min_size (int): Minimum size of the image to be rescaled before feeding it to the backbone.
        max_size (int): Maximum size of the image to be rescaled before feeding it to the backbone.
        anchor_sizes (Tuple[Tuple[int, ...], ...]): Sizes of anchors for each feature map.
        aspect_ratios (Tuple[Tuple[float, ...], ...]): Aspect ratios of anchors for each feature map.

    Returns:
        MaskRCNN: The constructed Mask R-CNN model.
    """

    # remove the last layer of the backbone
    backbone = torch.nn.Sequential(*list(backbone.children())[:-1])

    # pass a dummy input through the backbone to get the number of output channels
    feature_maps = backbone(torch.randn(1, 3, 224, 224))

    # set the number of output channels in the backbone
    backbone.out_channels = feature_maps.shape[1]   

    # create the anchor generator
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    # create the ROI pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    # create the mask ROI pooler
    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
    
    return MaskRCNN(backbone, 
                    num_classes=num_classes,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler,
                    mask_roi_pool=mask_roi_pooler,
                    min_size=min_size,
                    max_size=max_size)

def main():
    # load backbone
    backbone = torchvision.models.resnet50(weights="DEFAULT")

    # load mask rcnn model
    model = mask_rcnn(backbone, 91)
    print(model)
    
    # set the model to evaluation mode to avoid needing targets
    model.eval()

    # warmup pass
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        out = model(x)

    # draw bounding boxes and masks
    boxes = out[0]['boxes']
    labels = out[0]['labels']
    scores = out[0]['scores']
    masks = out[0]['masks']

    print(boxes)
    print(labels)
    print(scores)
    print(masks)

if __name__ == '__main__':
    main()
