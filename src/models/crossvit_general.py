from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

VENDOR_PATH = Path(__file__).resolve().parents[2] / "vendor" / "crossvit_ibm"
sys.path.insert(0, str(VENDOR_PATH))

from models.crossvit import VisionTransformer as IBMVisionTransformer  



def compute_patch_weights(mask_tensor, patch_size=16, gamma=1.0, epsilon=0.01):
    """
    Calcule wp = (epsilon + rp)^gamma avec normalisation unitaire[cite: 48, 51, 52].
    rp est le ratio de pixels plante dans le patch[cite: 48].
    """
    # 1. Masque binaire : 1 si pixel > 0 (plante), 0 sinon
    binary_mask = (mask_tensor.mean(dim=1, keepdim=True) > 0.05).float()
    
    # 2. Calcul du ratio rp par patch (aligné avec la grille ViT) [cite: 47, 48]
    rp = F.avg_pool2d(binary_mask, kernel_size=patch_size, stride=patch_size)
    
    # 3. Redimensionnement en [B, N, 1] pour la multiplication des tokens
    rp = rp.flatten(2).transpose(1, 2)
    
    # 4. Fonction puissance f(rp) [cite: 51]
    wp = torch.pow(epsilon + rp, gamma)
    
    # 5. Normalisation unitaire par image [cite: 52]
    wp = wp / (wp.mean(dim=1, keepdim=True) + 1e-8)
    
    return wp



class CrossViTBackbone(IBMVisionTransformer):
    """
    CrossViT IBM adapté pour :
      - Accepter un Tensor unique (dupliqué sur les branches)
      - Accepter une liste de Tensors [x0, x1] (routing externe)
      - Gérer l'interpolation automatique si la taille d'entrée != taille attendue (ex: 224 -> 240)
    Retourne les tokens CLS par branche.
    """
    def forward_features(self, x: Union[torch.Tensor, List[torch.Tensor]], weights: torch.Tensor = None):
        # 1. Gestion des entrées (Routing)
        if isinstance(x, torch.Tensor):
            # Cas A ou B : même image envoyée aux deux branches
            xs_in = [x for _ in range(self.num_branches)]
        else:
            # Cas C1 ou C2 : images différentes par branche
            if len(x) != self.num_branches:
                raise ValueError(f"Expected {self.num_branches} inputs, got {len(x)}")
            xs_in = x

        B = xs_in[0].shape[0]
        xs = []
        
        # 2. Préparation par branche (Interpolation + Patch Embed + Pos Embed)
        for i in range(self.num_branches):
            xi = xs_in[i]
            _, _, H, W = xi.shape
            
            # Si la branche attend 240px (Small) et reçoit 224px, on interpole.
            if H != self.img_size[i]:
                xi = torch.nn.functional.interpolate(
                    xi, size=(self.img_size[i], self.img_size[i]), mode="bicubic", align_corners=False
                )

	    # Patch Embedding
            tmp = self.patch_embed[i](xi)

            # --- PARTIE 3 : Application des poids par patch [cite: 19, 54] ---
            if weights is not None:
                tmp = tmp * weights # Propagation aux patches co-localisés
            cls_tokens = self.cls_token[i].expand(B, -1, -1)
            tmp = torch.cat((cls_tokens, tmp), dim=1)
            tmp = tmp + self.pos_embed[i]
            tmp = self.pos_drop(tmp)
            xs.append(tmp)

        # 3. Passage dans les blocs (inclus la Cross-Attention définie dans self.blocks)
        for blk in self.blocks:
            xs = blk(xs)

        # 4. Normalisation finale
        xs = [self.norm[i](t) for i, t in enumerate(xs)]
        
        # Retourne uniquement le token CLS (index 0) de chaque branche
        return [t[:, 0] for t in xs]

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]], weights: torch.Tensor = None):
        return self.forward_features(x, weights=weights)


def build_backbone(*, same_resolution: bool, img_size: int = 224) -> CrossViTBackbone:
    """
    Construction du Backbone selon les configurations du projet.
    
    Args:
        same_resolution: 
            False -> Partie 1 (Small/Large type CrossViT-Small)
            True  -> Partie 2 (Même patch size, même nb tokens)
        img_size: Taille de l'image d'entrée (généralement 224)
    """
    if same_resolution:
        # --- PARTIE 2 : CONFIGURATION ISO-RÉSOLUTION ---
        # On force les deux branches à être identiques structurellement
        img_size_list = (img_size, img_size)
        patch_size = (16, 16)
        embed_dim = (192, 192) # On aligne sur la dimension Small
        num_heads = (3, 3)
        # Profondeur légère pour l'expérience
        depth = ([1, 3, 1], [1, 3, 1], [1, 3, 1])
        mlp_ratio = (2.0, 2.0, 4.0)
        
    else:
        # --- PARTIE 1 : CONFIGURATION REFERENCE (CrossViT-Small) ---
        # Basé strictement sur 'crossvit_small_224' d'IBM
        
        # Branche 0 (Small): Patch 12 => Image 240 (240 / 12 = 20 tokens)
        # Branche 1 (Large): Patch 16 => Image 224 (224 / 16 = 14 tokens)
        img_size_list = (240, 224) 
        
        patch_size = (12, 16)
        embed_dim = (192, 384)
        num_heads = (6, 6) # Selon config officielle Small
        
        # Configuration officielle de la profondeur pour Small
        depth = ([1, 4, 0], [1, 4, 0], [1, 4, 0])
        mlp_ratio = (4.0, 4.0, 1.0)

    backbone = CrossViTBackbone(
        img_size=img_size_list,
        patch_size=patch_size,
        in_chans=3,
        num_classes=2,  # Pour éviter le 1000 par défaut dans ibm
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        multi_conv=False,
    )

    backbone.embed_dim = embed_dim
    return backbone



class CrossViTClassifier(nn.Module):
    """
    Wrapper générique pour la classification.
    Gère le routing des données (Non-Segmentée / Segmentée) vers les branches.
    """
    def __init__(
        self,
        *,
        same_resolution: bool,
        route: Sequence[int],          # ex: [0, 0] pour A, [0, 1] pour C1
        num_classes: int = 2,
        img_size: int = 224,
        patch_weighting: bool = False, # Optionnel pour O3
    ):
        
        super().__init__()
        self.patch_weighting = patch_weighting
        # Vérifications de sécurité
        if len(route) != 2:
            raise ValueError("route must have length 2 (for 2 branches)")
        if any(r not in (0, 1) for r in route):
            raise ValueError("route values must be 0 or 1 (0=nonseg, 1=seg)")
        
        self.route = list(route)
        self.backbone = build_backbone(same_resolution=same_resolution, img_size=img_size)
        
        # La tête de classification concatène les sorties des deux branches
        in_dim = sum(self.backbone.embed_dim)
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x_nonseg: torch.Tensor, x_seg: torch.Tensor) -> torch.Tensor:
        
        # Calcul des poids optionnel (Partie 3) [cite: 46]
        weights = None
        if self.patch_weighting:
            # On utilise le masque segmenté (x_seg) pour calculer les poids [cite: 46]
            weights = compute_patch_weights(x_seg, patch_size=16, gamma=1.0)
        
        # Préparation des sources
        sources = (x_nonseg, x_seg)
        
        # Routing dynamique selon la configuration (A, B, C1, C2)
        # xs est une liste [Tensor_pour_Branche0, Tensor_pour_Branche1]
        xs = [sources[self.route[0]], sources[self.route[1]]]
        
        # Passage dans le backbone
        # Le backbone gère l'interpolation si une branche (ex: Small) a besoin de 240px
        # alors que l'entrée est en 224px.
        feats = self.backbone(xs, weights=weights) # -> [CLS_Br0, CLS_Br1]
        
        # Concaténation et Classification
        x = torch.cat(feats, dim=1)
        return self.head(x)
