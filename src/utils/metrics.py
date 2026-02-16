import torch
import numpy as np

def compute_iou(heatmap, M_plant, quantile=0.5):
    """
    Computes IoU between heatmap and binary mask.
    
    Args:
        heatmap: Attention maps (Batch x H x W)
        M_plant: Binary masks (Batch x H x W) or (Batch x C x H x W)
        quantile: Used for calculating the binary version of each heatmap
    """
    assert 0.0 <= quantile <= 1.0, "Parameter 'quantile' should be a percentage between 0.0 and 1.0"

    # Convert M_plant to (Batch x H x W) if needed
    if M_plant.dim() == 4:
        # Convert RGB/multi-channel to binary by taking mean or max
        M_plant = M_plant.mean(dim=1)  # (Batch x H x W)
    
    # Binarize the plant mask (anything > threshold is plant)
    M_plant = (M_plant > 0.05).bool()

    # Calcul du seuil pour chaque heatmap du batch et conversion en carte binaire
    threshold = torch.quantile(heatmap, q=quantile, dim=(1, 2), keepdim=True)
    M_pred = (heatmap >= threshold)

    # Calcul de l'intersection et de l'union pour chaque paire
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
    """
    Compute attention rollout from multi-scale attention maps.
    """
    # Preserve device from input
    if isinstance(attention_maps, (list, tuple)) and len(attention_maps) > 0:
        if isinstance(attention_maps[0], (list, tuple)) and len(attention_maps[0]) > 0:
            device = attention_maps[0][0].device
        else:
            device = attention_maps[0].device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract attention for the specified branch
    if isinstance(attention_maps, (list, tuple)):
        if branch_idx < len(attention_maps):
            attn_list = attention_maps[branch_idx]
        else:
            attn_list = attention_maps[-1]
    else:
        attn_list = [attention_maps]
    
    # Process attention maps
    rollout = None
    for attn in attn_list:
        if attn.dim() == 4:  # (Batch x Heads x Tokens x Tokens)
            # Average over heads
            attn_avg = attn.mean(dim=1)  # (Batch x Tokens x Tokens)
            
            # Add identity and normalize
            I = torch.eye(attn_avg.size(-1), device=device).unsqueeze(0)
            attn_avg = attn_avg + I
            attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
            
            # Cumulative product
            if rollout is None:
                rollout = attn_avg
            else:
                rollout = torch.bmm(attn_avg, rollout)
    
    if rollout is None:
        raise ValueError("No valid attention maps found")
    
    # Extract CLS attention to patches
    cls_attn = rollout[:, 0, 1:]  # (Batch x num_patches)
    
    # Reshape to spatial dimensions
    batch_size = cls_attn.shape[0]
    num_patches = cls_attn.shape[1]
    patch_size = int(np.sqrt(num_patches))
    
    rollout_spatial = cls_attn.reshape(batch_size, patch_size, patch_size)
    
    return rollout_spatial


def compute_metrics(predictions, labels):
    """
    Compute classification metrics.
    
    Args:
        predictions: Model predictions (numpy array or tensor)
        labels: Ground truth labels (numpy array or tensor)
    
    Returns:
        dict: Dictionary with accuracy and F1 score
    """
    from sklearn.metrics import accuracy_score, f1_score
    
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted', zero_division=0)
    }
