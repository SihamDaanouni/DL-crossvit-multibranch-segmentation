import sys
from pathlib import Path
import torch
import torch.nn as nn

VENDOR_PATH = Path(__file__).resolve().parents[2] / "vendor" / "crossvit_ibm"
sys.path.insert(0, str(VENDOR_PATH))

from models.crossvit import VisionTransformer

class CrossViTC1(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()

        # Branch small: pour les images normales
        self.branch_small = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=128,
            depth=[1, 3, 0],
            num_heads=4,
            mlp_ratio=[3, 3, 1],
            qkv_bias=True,
        )

        # Branch large: pour les images segmentées
        self.branch_large = VisionTransformer(
            img_size=224,
            patch_size=12,
            embed_dim=256,
            depth=[1, 3, 0],
            num_heads=4,
            mlp_ratio=[3, 3, 1],
            qkv_bias=True,
        )

        # Addition fes dimensions d'embedding des deux branches
        self.head = nn.Linear(
            self.branch_small.embed_dim + self.branch_large.embed_dim,
            num_classes
        )

    def forward(self, x_small: torch.Tensor, x_large: torch.Tensor) -> torch.Tensor:
        # 1. Extraction des caractéristiques
        feat_small = self.branch_small(x_small)
        feat_large = self.branch_large(x_large)

        # 2. Concaténation sur la dimension des features (dim=1)
        combined = torch.cat([feat_small, feat_large], dim=1)

        # 3. Projection vers les classes
        return self.head(combined)
