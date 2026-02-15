from typing import Final, Optional, Type

from timm.layers._fx import register_notrace_function
from timm.layers.attention import Attention
from timm.models.vision_transformer import Block

import torch
from torch import nn as nn
from torch.nn import functional as F

from ...vendor.crossvit_ibm.models.crossvit import MultiScaleBlock, VisionTransformer, _compute_num_patches

@torch.fx.wrap
@register_notrace_function
def maybe_add_mask(scores: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    return scores if attn_mask is None else scores + attn_mask

class AttentionMap(Attention):
    """Standard Multi-head Self Attention module with QKV projection.

    Adapted version for rollout calcul from Attention in timm.layers.attention
    """
    def __init__(self, dim, num_heads = 8, attn_head_dim = None, dim_out = None, qkv_bias = False, qk_norm = False, scale_norm = False, proj_bias = True, attn_drop = 0, proj_drop = 0, norm_layer = None, device=None, dtype=None):
        super().__init__(dim, num_heads, attn_head_dim, dim_out, qkv_bias, qk_norm, scale_norm, proj_bias, attn_drop, proj_drop, norm_layer, device, dtype)

    def forward(
            self,
            x: torch.Tensor,
            attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = maybe_add_mask(attn, attn_mask)
        attn = attn.softmax(dim=-1)
        self.attn_map = attn.detach() # to store attention map for later use in MultiScaleBlock
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.attn_dim)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class BlockMap(Block):
    """Transformer block with pre-normalization.
    
    Adapted version for rollout calcul from Block in timm.models.vision_transformer
    """
    def __init__(self, dim, num_heads, mlp_ratio = 4, qkv_bias = False, qk_norm = False, scale_attn_norm = False, scale_mlp_norm = False, proj_bias = True, proj_drop = 0, attn_drop = 0, init_values = None, drop_path = 0, act_layer = nn.GELU, norm_layer = ..., mlp_layer = ..., attn_layer = ..., depth = 0, device=None, dtype=None):
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_norm, scale_attn_norm, scale_mlp_norm, proj_bias, proj_drop, attn_drop, init_values, drop_path, act_layer, norm_layer, mlp_layer, attn_layer, depth, device, dtype)

        dd = {'device': device, 'dtype': dtype}
        self.attn = AttentionMap(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            scale_norm=scale_attn_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
            depth=depth,
            **dd,
        )

class MultiScaleBlockMap(MultiScaleBlock):
    """
    MultiScaleBlock from IBM Cross-Vit implementation adapted for rollout calcul.
    
    Checkout vendors.models.crossvit.py for original implementation
    """
    def __init__(self, dim, patches, depth, num_heads, mlp_ratio, qkv_bias=False, qk_scale=None, drop=0, attn_drop=0, drop_path=0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__(dim, patches, depth, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer)

        num_branches = len(dim)
        self.num_branches = num_branches

        self.blocks = nn.ModuleList()
        for d in range(num_branches):
            tmp = []
            for i in range(depth[d]):
                tmp.append(
                    BlockMap(dim=dim[d], num_heads=num_heads[d], mlp_ratio=mlp_ratio[d], qkv_bias=qkv_bias, 
                          drop=drop, attn_drop=attn_drop, drop_path=drop_path[i], norm_layer=norm_layer))
            if len(tmp) != 0:
                self.blocks.append(nn.Sequential(*tmp))

        if len(self.blocks) == 0:
            self.blocks = None
    
    def forward(self, x):
        outs_b = [block(x_) for x_, block in zip(x, self.blocks)]

        # Adding attention matrices for each branch
        all_attn = []
        for branch_blocks in self.blocks:
            branch_attn = []
            for block in branch_blocks:
                branch_attn.append(block.attn.attn_map)
            all_attn.append(branch_attn)

        # only take the cls token out
        proj_cls_token = [proj(x[:, 0:1]) for x, proj in zip(outs_b, self.projs)]

        # cross attention
        outs = []
        for i in range(self.num_branches):
            tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
            tmp = self.fusion[i](tmp)
            reverted_proj_cls_token = self.revert_projs[i](tmp[:, 0:1, ...])
            tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
            outs.append(tmp)
        return outs, all_attn

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

def compute_heatmap(rollout_map, image, is_segmented:int=0, alpha:float=0.6):
    """
    Computes the heatmap from a rollout/attention map for an image (segmented or not).

    Args:
        rollout_map: Attention map (Batch x N x N). `N` = nb_patches + 1 (for CLS token)
        image: The image reference for interpolation of attention
        is_segemented: Using the segmented version of `image` or not - default is `0` for `False`
    """
    assert abs(is_segmented) <= 1, "Parameter 'is_segmented' should be 0 for False, 1 for True"
    assert abs(alpha)<= 1.0, "Parameter 'alpha' should be a percentage between 0.0 and 1.0"

    cls_influence = rollout_map[:, 0, 1:]
    N = rollout_map.shape[1] # B,N,N -> N
    H_img, W_img = image.shape[1:] # C,H,W -> H,W
    H_patches = W_patches = int((N-1)**0.5)

    heatmap = cls_influence.reshape(H_patches, W_patches) # Reshape en grid
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) # Normalisation

    # Interpolation à la taille de l'image
    heatmap = heatmap.unsqueeze(0).unsqueeze(0)  # 1 x 1 x H_patches x W_patches
    heatmap = F.interpolate(heatmap, size=(H_img, W_img), mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze()

    # Superposition entre image et attention (heatmap)
    heatmap = alpha * heatmap.unsqueeze(0) + (1-alpha) * image
    return heatmap

def compute_iou(heatmap, M_plant, quantile:float=0.8):
    """
    Computes the intersection over union for a given heatmap and mask.

    Args:
        heatmap: Attention map with continuous values in [0,1], H x W
        M_plant: The binary mask of the image (0: background, 1: plant), H x W
        quantile: Used for calculating the binary version of our `heatmap` into a prediction mask `M_pred` - default is `0.8`
    """
    assert abs(quantile)<= 1.0, "Parameter 'quantile' should be a percentage between 0.0 and 1.0"

    threshold = torch.quantile(heatmap, q=0.8)
    M_pred = (heatmap >= threshold)
    M_plant = M_plant.bool() # Conversion to bolean matrix for logical operations

    intersection = (M_pred & M_plant).sum()
    union = (M_pred | M_plant).sum()
    iou = intersection / union

    return iou

def iou_loss(heatmap, M_plant):
    """
    Computes the additional loss value based on the IoU for the ``heatmap`` and the labels (``M_plant``)

    Args:
        heatmap: Attention map with continuous values in [0,1], H x W
        M_plant: The binary mask of the image (0: background, 1: plant), H x W
    """
    intersection = (heatmap * M_plant).sum()
    union = (heatmap + M_plant - heatmap*M_plant).sum()
    loss = 1 - (intersection / (union + 1e-6)) # 1e-6 pour évitez le cas ou union = 0
    return loss


