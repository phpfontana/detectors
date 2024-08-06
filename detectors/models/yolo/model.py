import torch.nn as nn
from torch import Tensor

class YOLO(nn.Module):
    def __init__(self):
        super(YOLO, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        pass