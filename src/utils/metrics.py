import torch
import numpy as np
import torch.nn.functional as F
def compute_iou(heatmap, M_plant, quantile=0.5):
    """
    Computes IoU between heatmap and binary mask.
    """

    assert 0.0 <= quantile <= 1.0, "Parameter 'quantile' should be between 0.0 and 1.0"

    # Ensure heatmap is (B,H,W)
    if heatmap.dim() == 4:
        heatmap = heatmap.squeeze(1)

    # Convert mask to (B,H,W)
    if M_plant.dim() == 4:
        M_plant = M_plant.mean(dim=1)

    # Binarize mask
    M_plant = (M_plant > 0.05).bool()

    B = heatmap.shape[0]

    # ✅ Flatten spatial dims
    heatmap_flat = heatmap.view(B, -1)

    threshold = torch.quantile(
        heatmap_flat,
        q=quantile,
        dim=1,
        keepdim=True
    )

    threshold = threshold.view(B, 1, 1)

    # Binary prediction
    M_pred = heatmap >= threshold

    # IoU
    intersection = (M_pred & M_plant).sum(dim=(1, 2))
    union = (M_pred | M_plant).sum(dim=(1, 2))

    iou = intersection.float() / (union.float() + 1e-6)

    return iou

def compute_heatmap(rollout, seg, alpha=0.6):
    """
    Compute heatmap from attention rollout.
    
    Args:
        rollout: Attention rollout (Batch x H x W) or (Batch x 1 x H x W)
        seg: Segmented images (Batch x C x H x W)
        alpha: Blending factor
    
    Returns:
        heatmap: Normalized attention heatmap (Batch x H x W)
    """
    # Get device from input
    device = rollout.device
    
    # Ensure rollout is (Batch x H x W)
    if rollout.dim() == 4:
        rollout = rollout.squeeze(1)
    
    # Normalize rollout to [0, 1] for each image in batch
    batch_size = rollout.shape[0]
    heatmap = torch.zeros_like(rollout, device=device)  # Ensure same device
    
    for i in range(batch_size):
        r = rollout[i]
        r_min = r.min()
        r_max = r.max()
        if r_max > r_min:
            heatmap[i] = (r - r_min) / (r_max - r_min)
        else:
            heatmap[i] = r
    
    return heatmap


def compute_iou_loss(heatmap, M_plant):
    """
    Computes the IoU loss for a batch of heatmaps and masks.
    """
    # Get device from input
    device = heatmap.device
    
    # Convert M_plant to (Batch x H x W) if needed
    if M_plant.dim() == 4:
        M_plant = M_plant.mean(dim=1)  # (Batch x H x W)
    
    # Ensure both tensors are on the same device
    if M_plant.device != device:
        M_plant = M_plant.to(device)
    
    # Ensure heatmap is also (Batch x H x W)
    if heatmap.dim() == 4:
        heatmap = heatmap.squeeze(1)
    
    # Binarize the plant mask
    M_plant = (M_plant > 0.05).float()

    # Normalize heatmap to [0, 1] per image for stability
    batch_size = heatmap.shape[0]
    heatmap_normalized = torch.zeros_like(heatmap, device=device)
    
    for i in range(batch_size):
        h = heatmap[i]
        h_min = h.min()
        h_max = h.max()
        if h_max > h_min:
            heatmap_normalized[i] = (h - h_min) / (h_max - h_min)
        else:
            heatmap_normalized[i] = h

    # Compute soft IoU (differentiable)
    intersection = (heatmap_normalized * M_plant).sum(dim=(1, 2))
    union = (heatmap_normalized + M_plant - heatmap_normalized * M_plant).sum(dim=(1, 2))

    # IoU per sample
    iou = intersection / (union + 1e-6)
    
    # Loss is 1 - IoU
    loss = 1 - iou.mean()

    return loss


def compute_rollout(attention_maps, branch_idx):

    if not attention_maps:
        raise ValueError("Empty attention maps")

    # Flatten structure to get tensors of chosen branch
    attn_list = []

    for layer in attention_maps:
        if not isinstance(layer, list):
            continue

        if branch_idx >= len(layer):
            continue

        branch_data = layer[branch_idx]

        # branch_data can be tensor OR list of tensors
        if isinstance(branch_data, list):
            for item in branch_data:
                if torch.is_tensor(item):
                    attn_list.append(item)
        elif torch.is_tensor(branch_data):
            attn_list.append(branch_data)

    if not attn_list:
        raise ValueError("No valid attention tensors found")

    device = attn_list[0].device

    rollout = None

    for attn in attn_list:

        # (B, Heads, Tokens, Tokens)
        attn_avg = attn.mean(dim=1)

        I = torch.eye(attn_avg.size(-1), device=device).unsqueeze(0)
        attn_avg = attn_avg + I
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)

        if rollout is None:
            rollout = attn_avg
        else:
            rollout = torch.bmm(attn_avg, rollout)

    # CLS → patches
    cls_attn = rollout[:, 0, 1:]

    B = cls_attn.shape[0]
    num_patches = cls_attn.shape[1]
    side = int(num_patches ** 0.5)

    rollout_spatial = cls_attn.reshape(B, side, side)

    rollout_spatial = rollout_spatial.unsqueeze(1)  # (B,1,14,14)

    rollout_upsampled = F.interpolate(
        rollout_spatial,
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    )

    return rollout_upsampled.squeeze(1)  # (B,224,224)

def compute_metrics(predictions, labels):
    """
    Compute classification metrics.
    
    Args:
        predictions: Model predictions (numpy array or tensor)
        labels: Ground truth labels (numpy array or tensor)
    
    Returns:
        dict: Dictionary with accuracy and F1 score
    """
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted', zero_division=0),
        'classification_report': classification_report(labels, predictions, zero_division=0)
    }
