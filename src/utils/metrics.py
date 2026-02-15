import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from typing import Dict

def compute_metrics(preds: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Computes Accuracy, F1-score and the confusion matrix.

    Args:
        preds: The prediction of the model as `torch.Tensor`
        labels: The ground truth as `torch.Tensor`
    """
    preds = preds.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
        "conf_matrix": confusion_matrix(labels, preds).tolist()
    }

def compute_rollout(all_attn_layers, branch:int = 0):
    """
    Computes the rollout after a forward_attention from RolloutVisionTransformer

    Args:
        all_attn_layers: list of branches -> list of blocks -> attn_map (B x heads x N x N)
        branch: index of branch for rollout - default is `0`
    """
    attn_mats = []

    # Boucle sur les blocks de la branche sélectionnée
    for attn in all_attn_layers[branch]:
        A = attn.mean(dim=1) # Moyenne sur les tetes
        I = torch.eye(A.size(-1)).to(A.device)
        A = A + I # Ajout de l'identité
        A = A / A.sum(dim=-1, keepdim=True) # Normalisation
        attn_mats.append(A)

    # Produit cumulatif
    rollout = attn_mats[0]
    for A in attn_mats[1:]:
        rollout = A @ rollout
    return rollout

def compute_heatmap(rollout_map, images, alpha:float=0.6):
    """
    Computes the heatmap from a rollout/attention map for a batch of images (segmented).

    Args:
        rollout_map: Attention map (Batch x N x N). `N` = nb_patches + 1 (for CLS token)
        images: The batch of images corrolated with `rollout_map`
        alpha: Percentage between 0.0 and 1.0 for heatmap blending - default is `0.6`
    """
    assert 0.0 <= alpha <= 1.0, "Parameter 'alpha' should be a percentage between 0.0 and 1.0"

    B, N, _ = rollout_map.shape
    H_img, W_img = images.shape[2:]

    cls_influence = rollout_map[:, 0, 1:]  # (B, N-1)
    H_patches = W_patches = int((N-1)**0.5)

    heatmap = cls_influence.reshape(B, H_patches, W_patches)

    # Normalisation par image
    min_val = heatmap.amin(dim=(1,2), keepdim=True)
    max_val = heatmap.amax(dim=(1,2), keepdim=True)
    heatmap = (heatmap - min_val) / (max_val - min_val + 1e-6)

    # Interpolation pour chaque élément du batch
    heatmap = heatmap.unsqueeze(1)  # B, H_patches, W_patches -> B, 1, H_patches, W_patches
    heatmap = F.interpolate(heatmap, size=(H_img, W_img), mode='bilinear', align_corners=False)

    return alpha * heatmap + (1-alpha) * images

def compute_iou(heatmap, M_plant, quantile:float=0.8):
    """
    Computes the intersection over union for a batch of heatmaps and masks.

    Args:
        heatmap: Attention maps (Batch x H x W)
        M_plant: Binary masks (Batch x H x W)
        quantile: Used for calculating the binary version of each heatmap
    """
    assert 0.0 <= quantile <= 1.0, "Parameter 'quantile' should be a percentage between 0.0 and 1.0"

    # Calcul du seuil pour chaque heatmap du batch et conversion en carte binaire
    threshold = torch.quantile(heatmap, q=quantile, dim=(1, 2), keepdim=True)
    M_pred = (heatmap >= threshold)

    # Conversion en booléen
    M_plant = M_plant.bool()

    # Calcul de l'intersection et de l'union pour chaque paire
    intersection = (M_pred & M_plant).sum(dim=(1, 2))
    union = (M_pred | M_plant).sum(dim=(1, 2))
    iou = intersection.float() / (union.float() + 1e-6)

    return iou

def compute_iou_loss(heatmap, M_plant):
    """
    Computes the IoU loss for a batch of heatmaps and masks.

    Args:
        heatmap: Attention maps (Batch x H x W)
        M_plant: Binary masks (Batch x H x W)
    """
    M_plant = M_plant.float()

    intersection = (heatmap * M_plant).sum(dim=(1,2))
    union = (heatmap + M_plant - heatmap*M_plant).sum(dim=(1,2))

    loss = 1 - (intersection / (union + 1e-6))

    return loss.mean()