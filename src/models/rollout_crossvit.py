from typing import Final, Optional, Type

from timm.layers._fx import register_notrace_function
from timm.layers.attention import Attention
from timm.models.vision_transformer import Block

import torch
from torch import nn as nn
from torch.nn import functional as F

from vendor.crossvit_ibm.models.crossvit import MultiScaleBlock, VisionTransformer, _compute_num_patches

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
