import sys
from pathlib import Path
import torch
import torch.nn as nn

VENDOR_PATH = Path(__file__).resolve().parents[2] / "vendor" / "crossvit_ibm"
sys.path.insert(0, str(VENDOR_PATH))

from models.crossvit import crossvit_9_224 

class CrossViTO1A(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        self.backbone = crossvit_9_224(pretrained=pretrained, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
