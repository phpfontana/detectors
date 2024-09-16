import torch
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator

def mask_rcnn(backbone: torch.nn.Module, num_classes: int = 1000) -> torchvision.models.detection.MaskRCNN:
    """
    Args:
        backbone (torch.nn.Module): 
        num_classes (int): 

    Returns:
        torchvision.models.detection.MaskRCNN: 
    """

    backbone = backbone.feature_extractor
    
    feature_maps = backbone(torch.randn(1, 3, 224, 224))

    backbone.out_channels = feature_maps.shape[1]

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)

    mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
    
    return torchvision.models.detection.MaskRCNN(backbone, num_classes, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler, mask_roi_pool=mask_roi_pooler)

def main():
    pass

if __name__ == '__main__':
    main()