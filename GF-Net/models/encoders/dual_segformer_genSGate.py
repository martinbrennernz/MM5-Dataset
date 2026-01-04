import torch
import torch.nn as nn
from typing import List
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from ..net_utils import FeatureFusionModule as FFM
from ..net_utils import FeatureRectifyModule as FRM
from ..cmm5_fusion import (FineRegistrationFusion, CoarseRegistration, SpatialContributionAttention,
                           TPSRegistration, CrossAttention, RGBEnhancementFusion, StageWiseRGBIntensityFusion, SigmoidGateModule)
import sys
import math
import time
from config import config as cfg
from engine.logger import get_logger
import logging

logger = get_logger()
# logger.setLevel(logging.DEBUG)

# Dummy helper function to get tensor shape/dtype/device info safely
def get_tensor_info(tensor, name="Tensor"):
    if tensor is None:
        return f"{name}: None"
    return f"{name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}"

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class TwoInputIdentity(nn.Module):
    def forward(self, x, y):
        # Returning zeros disables the contribution; you could return y if you want passthrough
        return torch.zeros_like(x)
    
class RGBXTransformer(nn.Module):
    def __init__(
        self,
        *,
        n_modal: int = 1,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        extra_in_chans: List[int],
        num_classes: int = 1000,
        embed_dims: List[int] = [64, 128, 256, 512],
        num_heads: List[int] = [1, 2, 4, 8],
        mlp_ratios: List[float] = [4, 4, 4, 4],
        qkv_bias: bool = False,
        qk_scale = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        norm_layer = nn.LayerNorm,
        norm_fuse = nn.BatchNorm2d,
        depths: List[int] = [3, 4, 6, 3],
        sr_ratios: List[int] = [8, 4, 2, 1],
        alignment_method: str = "frm",
        fusion_method: str = "ffm", # Underlying fusion primitive
        fusion_combination: str = "sigmoid_gating", # Default and only active combination method
        use_intensity_enhancement: bool = False,
        sgate_fusion_mode: str = "add", # "add" or "avg" for sigmoid_gating
    ):
        super().__init__()
        self.n_modal = n_modal
        self.alignment_method = alignment_method.lower()
        self.fusion_method = fusion_method.lower()
        self.fusion_combination = fusion_combination.lower() # Will be "sigmoid_gating"
        self.use_intensity_enhancement = use_intensity_enhancement
        self.sgate_fusion_mode = sgate_fusion_mode
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.num_heads = num_heads # Storing num_heads if needed by sub-modules

        # Attributes for sgate_fusion_mode options "weighted_sum", "concat_conv", "attention", "hierarchical_cross_attention"
        # are removed as these modes are no longer supported.
        # "add" and "avg" sgate_fusion_modes do not require these specific attributes.
        # self.sgate_fusion_weights = None
        # self.sgate_concat_conv_layers = nn.ModuleList()
        # self.sgate_attention_thermal_combiners = nn.ModuleList()
        # self.sgate_attention_uv_combiners = nn.ModuleList()
        # self.sgate_inter_contrib_cross_att = nn.ModuleList()
        # self.sgate_final_cross_att = nn.ModuleList()
        
        logger.info(f"Initializing RGBXTransformer:")
        logger.info(f"  n_modal (extra): {n_modal}")
        logger.info(f"  extra_in_chans: {extra_in_chans}")
        logger.info(f"  use_intensity_enhancement (RGB+Mod0): {self.use_intensity_enhancement}")
        logger.info(f"  alignment_method: {self.alignment_method}")
        logger.info(f"  fusion_combination: {self.fusion_combination} (Sigmoid Gating is the only active method)")
        if self.fusion_combination == "sigmoid_gating": # This will always be true now
            logger.info(f"  sgate_fusion_mode: {self.sgate_fusion_mode} (Only 'add' or 'avg' are fully supported)")
            if self.sgate_fusion_mode not in ["add", "avg"]:
                 logger.warning(f"Sigmoid Gating mode '{self.sgate_fusion_mode}' is selected, but only 'add' and 'avg' are "
                                f"actively supported. Others will default to 'add'.")
        else:
            # This else block should ideally not be reached if fusion_combination is forced to "sigmoid_gating"
            logger.error(f"Fusion combination '{self.fusion_combination}' is not 'sigmoid_gating'. This is unexpected.")
            # Defaulting to sigmoid_gating if somehow it's different.
            self.fusion_combination = "sigmoid_gating"


        if n_modal > 0 and len(extra_in_chans) != n_modal:
             raise ValueError(f"Length of extra_in_chans ({len(extra_in_chans)}) must match n_modal ({n_modal})")

        # --- Patch Embeddings --- (Standard, no changes)
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        self.extra_patch_embed1 = nn.ModuleList()
        self.extra_patch_embed2 = nn.ModuleList()
        self.extra_patch_embed3 = nn.ModuleList()
        self.extra_patch_embed4 = nn.ModuleList()
        if n_modal > 0:
            self.extra_patch_embed1 = nn.ModuleList([OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=extra_in_chans[m], embed_dim=embed_dims[0]) for m in range(n_modal)])
            self.extra_patch_embed2 = nn.ModuleList([OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1]) for _ in range(n_modal)])
            self.extra_patch_embed3 = nn.ModuleList([OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2]) for _ in range(n_modal)])
            self.extra_patch_embed4 = nn.ModuleList([OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3]) for _ in range(n_modal)])

        # --- Transformer Encoder Blocks --- (Standard, no changes)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.block1 = nn.ModuleList([Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        self.extra_block1 = nn.ModuleList([nn.ModuleList([Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0]) for i in range(depths[0])]) for _ in range(self.n_modal)])
        self.extra_norm1 = nn.ModuleList([norm_layer(embed_dims[0]) for _ in range(self.n_modal)])
        cur += depths[0]
        self.block2 = nn.ModuleList([Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        self.extra_block2 = nn.ModuleList([nn.ModuleList([Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1]) for i in range(depths[1])]) for _ in range(self.n_modal)])
        self.extra_norm2 = nn.ModuleList([norm_layer(embed_dims[1]) for _ in range(self.n_modal)])
        cur += depths[1]
        self.block3 = nn.ModuleList([Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        self.extra_block3 = nn.ModuleList([nn.ModuleList([Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2]) for i in range(depths[2])]) for _ in range(self.n_modal)])
        self.extra_norm3 = nn.ModuleList([norm_layer(embed_dims[2]) for _ in range(self.n_modal)])
        cur += depths[2]
        self.block4 = nn.ModuleList([Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        self.extra_block4 = nn.ModuleList([nn.ModuleList([Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[3]) for i in range(depths[3])]) for _ in range(self.n_modal)])
        self.extra_norm4 = nn.ModuleList([norm_layer(embed_dims[3]) for _ in range(self.n_modal)])

        # --- Alignment Modules ---
        self.FRMs_pair = nn.ModuleList()
        self.coarse_reg = nn.ModuleList()
        self.tps_reg = nn.ModuleList()
        if self.alignment_method == "frm":
            logger.info("Using FRM Alignment Modules.")
            self.FRMs_pair = nn.ModuleList([nn.ModuleList([FRM(dim=d, reduction=1) for _ in range(n_modal)]) for d in embed_dims])
        elif self.alignment_method == "stn":
             logger.info("Using STN Alignment Modules.")
             self.coarse_reg = nn.ModuleList([nn.ModuleList([CoarseRegistration(input_channels=d) for _ in range(n_modal)]) for d in embed_dims])
        elif self.alignment_method == "tps":
             logger.info("Using TPS Alignment Modules.")
             self.tps_reg = nn.ModuleList([nn.ModuleList([TPSRegistration(input_channels=d, num_ctrl_pts=5) for _ in range(n_modal)]) for d in embed_dims])
        elif self.alignment_method != "none": raise ValueError(f"Unknown alignment_method {self.alignment_method!r}")
        else: logger.info("Feature alignment explicitly disabled.")
        # Ensure lists exist even if empty
        if not isinstance(self.FRMs_pair, nn.ModuleList): self.FRMs_pair = nn.ModuleList([nn.ModuleList() for _ in embed_dims])
        if not isinstance(self.coarse_reg, nn.ModuleList): self.coarse_reg = nn.ModuleList([nn.ModuleList() for _ in embed_dims])
        if not isinstance(self.tps_reg, nn.ModuleList): self.tps_reg = nn.ModuleList([nn.ModuleList() for _ in embed_dims])

        # --- Stage-Wise RGB + Modality 0 (DIM) Fusion Modules --- (Standard, no changes)
        self.rgb_intensity_fuse = nn.ModuleList()
        if self.use_intensity_enhancement:
             if self.n_modal < 1: raise ValueError("use_intensity_enhancement=True requires at least 1 extra modality (DIM).")
             self.rgb_intensity_fuse = nn.ModuleList([StageWiseRGBIntensityFusion(dim=d) for d in embed_dims])

        # --- Main Fusion Modules (Choose one based on fusion_method) ---
        self.fuse_block = nn.ModuleList() # For methods like FFM, STN-Fusion, Add
        self.fuse_pair = nn.ModuleList()  # For methods like CAFB
        if self.fusion_method == "ffm":
            logger.info("Using FFM as the fusion primitive.")
            self.fuse_block = nn.ModuleList([FFM(dim=d, reduction=1, num_heads=h, norm_layer=norm_fuse) for d, h in zip(embed_dims, num_heads)])
        elif self.fusion_method == "cafb":
             logger.info("Using CAFB as the fusion primitive.")
             self.fuse_pair = nn.ModuleList([nn.ModuleList([CrossAttention(dim=d) for _ in range(n_modal)]) for d in embed_dims])
        elif self.fusion_method == "stn":
             logger.info("Using FineRegistrationFusion as the fusion primitive.")
             self.fuse_block = nn.ModuleList([FineRegistrationFusion(input_channels=d) for d in embed_dims])
        elif self.fusion_method == "add":
             logger.info("Using simple addition as the fusion primitive.")
             self.fuse_block = nn.ModuleList([nn.Identity() for _ in embed_dims]) # Addition handled in forward
        else: raise ValueError(f"Unknown fusion_method {self.fusion_method!r}")
        # Ensure lists exist even if empty
        if not isinstance(self.fuse_block, nn.ModuleList): self.fuse_block = nn.ModuleList([nn.Identity() for _ in embed_dims])
        if not isinstance(self.fuse_pair, nn.ModuleList): self.fuse_pair = nn.ModuleList([nn.ModuleList() for _ in embed_dims])


        # --- Fusion Combination Specific Layers ---
        self.concat_fuse = nn.ModuleList()
        self.extra_fuse = nn.ModuleList()
        self.gates = nn.ModuleList() # Original softmax attention gates
        # Sigmoid Gating modules 
        self.sigmoid_gates = nn.ModuleList([nn.ModuleList() for _ in range(len(self.embed_dims))])
        
        if self.fusion_combination == "concat":
             logger.info("Initializing Concat fusion layers.")
             # Correct concat input channels: Base + (n_modal - 1 if intensity enhancement else n_modal) remaining modalities
             # This calculation is complex depending on dim_consumed. Easier to check in forward.
             # Let's assume it concatenates base + all *remaining* modalities.
             # If DIM consumed, need base + T + UV -> 1 + (n_modal-1) = n_modal features
             # If DIM not consumed, need base + DIM + T + UV -> 1 + n_modal features
             # The forward pass logic for concat needs careful channel checking.
             # Initializing assuming base + T + UV (i.e., n_modal-1 extra)
             if self.use_intensity_enhancement:
                 num_concat_modalities = self.n_modal # Base (already fused) + remaining (T, UV)
             else:
                 num_concat_modalities = self.n_modal + 1 # Base + DIM + T + UV
             logger.info(f"  Concat assuming {num_concat_modalities} features total per stage.")
             # This assumes concat_fuse takes the concatenated input and reduces to embed_dims[k]
             self.concat_fuse = nn.ModuleList([nn.Conv2d(embed_dims[k] * num_concat_modalities, embed_dims[k], kernel_size=1) for k in range(len(embed_dims))])

        elif self.fusion_combination == "hierarchical_extra":
             logger.info("Initializing Hierarchical Extra fusion layers.")
             # Fuses remaining modalities sequentially (T with UV if both present)
             num_hierarchical_fusions = max(0, self.n_modal - 1 - (1 if self.use_intensity_enhancement else 0))
             logger.info(f"  Hierarchical expecting {num_hierarchical_fusions} fusion steps per stage.")
             if num_hierarchical_fusions > 0:
                # Assuming FFM is used for hierarchical steps
                self.extra_fuse = nn.ModuleList([nn.ModuleList([FFM(dim=embed_dims[k], reduction=1, num_heads=num_heads[k], norm_layer=norm_fuse) for _ in range(num_hierarchical_fusions)]) for k in range(len(embed_dims))])
             else:
                 self.extra_fuse = nn.ModuleList([nn.ModuleList() for _ in embed_dims])

        elif self.fusion_combination == "attention_gating": # Original Softmax ATG
             logger.info("Initializing Softmax Attention Gating layers.")
             # Number of features to gate: Base + remaining modalities
             if self.use_intensity_enhancement:
                 num_features_to_gate = self.n_modal # Base(RGB+DIM) + T + UV
             else:
                 num_features_to_gate = self.n_modal + 1 # Base(RGB) + DIM + T + UV
             logger.info(f"  Softmax ATG gating {num_features_to_gate} features per stage.")
             self.gates = nn.ModuleList([nn.Sequential(
                    nn.Conv2d(embed_dims[k] * num_features_to_gate, num_features_to_gate, kernel_size=1),
                    nn.Softmax(dim=1)
                ) for k in range(len(embed_dims))])

        # <<< --- NEW SIGMOID GATING INITIALIZATION --- >>>
        elif self.fusion_combination == "sigmoid_gating": 
            logger.info("Initializing Sigmoid Gating modules for all extra modalities (excluding the first if intensity enhancement is enabled).")
            for k_idx, channels in enumerate(self.embed_dims):
                if self.use_intensity_enhancement:
                    gate_modalities = range(1, self.n_modal)  # Modality 0 (e.g. DIM) handled by enhancement
                else:
                    gate_modalities = range(self.n_modal)
                # For each gating modality at this stage, create a SigmoidGateModule
                for m in gate_modalities:
                    gate_module = SigmoidGateModule(base_channels=channels, mod_channels=channels, output_channels=channels)
                    # Optionally initialize bias for the UV modality (index 2)
                    if m == 2:
                        gate_mlp_block = gate_module.gate_mlp
                        final_conv_for_uv_bias = None
                        if len(gate_mlp_block) > 1 and isinstance(gate_mlp_block[-2], nn.Conv2d) and gate_mlp_block[-2].bias is not None:
                            final_conv_for_uv_bias = gate_mlp_block[-2]
                        if final_conv_for_uv_bias is not None:
                            nn.init.constant_(final_conv_for_uv_bias.bias, 0.0)
                        else:
                            logger.warning(f"Could not find final Conv2d with bias in UV gate_mlp for stage {k_idx} to initialize bias.")
                    self.sigmoid_gates[k_idx].append(gate_module)
                # If enhancement, pad list with TwoInputIdentity() at index 0 so gate indexing always matches modality index
                if self.use_intensity_enhancement:
                    self.sigmoid_gates[k_idx].insert(0, TwoInputIdentity())
                # If any modalities are missing, pad with TwoInputIdentity() for uniform length
                while len(self.sigmoid_gates[k_idx]) < self.n_modal:
                    self.sigmoid_gates[k_idx].append(TwoInputIdentity())
        else:
            logger.warning(f"Unexpected fusion_combination '{self.fusion_combination}'. No fusion combination modules initialized.")
            # Initialize with identities for compatibility if needed
            self.sigmoid_gates = nn.ModuleList([nn.ModuleList([TwoInputIdentity() for _ in range(self.n_modal)]) for _ in self.embed_dims])

        
        # Ensure other removed fusion helper lists are empty or Identity if strictly needed elsewhere (they shouldn't be)
        self.concat_fuse = nn.ModuleList() # Emptied
        self.extra_fuse = nn.ModuleList()  # Emptied
        self.gates = nn.ModuleList()       # Emptied (original softmax gates)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
             is_special_fusion_conv = False
             if hasattr(self, 'rgb_intensity_fuse'):
                 for enhancer in self.rgb_intensity_fuse:
                     if hasattr(enhancer, 'modules') and m in enhancer.modules(): is_special_fusion_conv = True; break
             if not is_special_fusion_conv:
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
             is_special_fusion_bn = False
             if hasattr(self, 'rgb_intensity_fuse'):
                 for enhancer in self.rgb_intensity_fuse:
                     if hasattr(enhancer, 'modules') and m in enhancer.modules(): is_special_fusion_bn = True; break
             if not is_special_fusion_bn:
                 nn.init.constant_(m.weight, 1.0)
                 nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        elif pretrained is None:
             logger.info("Initializing weights from scratch.")
        else:
            raise TypeError('pretrained must be a str or None')

    def _apply_single_fusion(self, stage_idx, modal_idx, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # This helper is mainly for "sequential" or "parallel" combinations which are now removed.
        # However, StageWiseRGBIntensityFusion might still use a similar concept internally
        # or if fusion_method itself uses it.
        if self.fusion_method == "add":
            return A + B
        elif self.fusion_method == "ffm":
            if stage_idx < len(self.fuse_block): return self.fuse_block[stage_idx](A, B)
            else: return A + B # Fallback
        elif self.fusion_method == "cafb":
             if stage_idx < len(self.fuse_pair) and modal_idx < len(self.fuse_pair[stage_idx]): return self.fuse_pair[stage_idx][modal_idx](A, B)
             else: return A + B # Fallback
        elif self.fusion_method == "stn":
             if stage_idx < len(self.fuse_block): return self.fuse_block[stage_idx](B, A) # Note order for STN
             else: return A + B # Fallback
        else:
            logger.warning(f"Unknown fusion_method '{self.fusion_method}' in _apply_single_fusion. Defaulting to Add.")
            return A + B


    def forward_features(self, x_rgb: torch.Tensor, xs: List[torch.Tensor]):
        """ Forward pass through the backbone with detailed debugging and Intensity Enhancement AFTER Alignment. """
        logger.debug("="*20 + " Entering forward_features " + "="*20)
        logger.debug(f"Initial RGB input: {get_tensor_info(x_rgb, 'x_rgb')}")
        logger.debug(f"Initial extra modalities (count={len(xs)}):")
        for i, x in enumerate(xs): logger.debug(f"  xs[{i}]: {get_tensor_info(x, f'xs[{i}]')}")
        logger.debug(f"Config: n_modal={self.n_modal}, use_intensity_enhancement={self.use_intensity_enhancement}, alignment_method='{self.alignment_method}', fusion_combination='{self.fusion_combination}'")

        B = x_rgb.size(0)
        outs = [] 

        patch_embed_rgb = [self.patch_embed1, self.patch_embed2, self.patch_embed3, self.patch_embed4]
        patch_embed_ext = [self.extra_patch_embed1, self.extra_patch_embed2, self.extra_patch_embed3, self.extra_patch_embed4]
        blocks_rgb = [self.block1, self.block2, self.block3, self.block4]
        blocks_ext = [self.extra_block1, self.extra_block2, self.extra_block3, self.extra_block4]
        norms_rgb = [self.norm1, self.norm2, self.norm3, self.norm4]
        norms_ext = [self.extra_norm1, self.extra_norm2, self.extra_norm3, self.extra_norm4]
        current_xs_features = xs

        for k in range(len(self.depths)): # Loop through stages 0, 1, 2, 3
            # ... (❶ Patch Embedding, ❷ Transformer Blocks, ❸ Layer Normalization, ❹ Reshape as before) ...
            x_rgb_tok, H, W = patch_embed_rgb[k](x_rgb)
            xs_tok = []
            for m in range(self.n_modal):
                if current_xs_features[m] is not None:
                    tok, _, _ = patch_embed_ext[k][m](current_xs_features[m])
                    xs_tok.append(tok)
                else: xs_tok.append(None)
            
            x_rgb_processed = x_rgb_tok
            for blk in blocks_rgb[k]: x_rgb_processed = blk(x_rgb_processed, H, W)
            
            xs_tok_processed = []
            for m in range(self.n_modal):
                xm = xs_tok[m]
                if xm is not None:
                    for blk in blocks_ext[k][m]: xm = blk(xm, H, W)
                    xs_tok_processed.append(xm)
                else: xs_tok_processed.append(None)

            x_rgb_norm = norms_rgb[k](x_rgb_processed)
            xs_tok_norm = []
            for m in range(self.n_modal):
                if xs_tok_processed[m] is not None:
                    xs_tok_norm.append(norms_ext[k][m](xs_tok_processed[m]))
                else: xs_tok_norm.append(None)

            Ck = self.embed_dims[k]
            rgb_img = x_rgb_norm.reshape(B, H, W, Ck).permute(0, 3, 1, 2).contiguous()
            xs_img = []
            for m, xt in enumerate(xs_tok_norm):
                if xt is not None:
                    xs_img.append(xt.reshape(B, H, W, Ck).permute(0, 3, 1, 2).contiguous())
                else: xs_img.append(None)
            
            # --- Fusion/Enhancement ---
            dim_consumed_this_stage = False
            base_feat_for_fusion = rgb_img

            # ❻ Alignment (Applied per extra modality relative to original RGB)
            xs_img_aligned = []
            base_for_align = rgb_img
            for m in range(self.n_modal):
                x_mod_img = xs_img[m]
                if x_mod_img is None: xs_img_aligned.append(None); continue
                alignment_needed = cfg.x_align_needed[m] if hasattr(cfg, 'x_align_needed') and m < len(cfg.x_align_needed) else False
                if alignment_needed and self.alignment_method != "none":
                    aligned_x = None
                    if self.alignment_method == "frm":
                        if k < len(self.FRMs_pair) and m < len(self.FRMs_pair[k]): _, aligned_x = self.FRMs_pair[k][m](base_for_align.detach(), x_mod_img)
                    elif self.alignment_method == "stn":
                         if k < len(self.coarse_reg) and m < len(self.coarse_reg[k]): aligned_x, _ = self.coarse_reg[k][m](x_mod_img.detach(), base_for_align.detach())
                    elif self.alignment_method == "tps":
                         if k < len(self.tps_reg) and m < len(self.tps_reg[k]): aligned_x, _ = self.tps_reg[k][m](x_mod_img, base_for_align)
                    xs_img_aligned.append(aligned_x if aligned_x is not None else x_mod_img)
                else:
                    xs_img_aligned.append(x_mod_img)

            # ❺ Optional Stage-Wise RGB+DIM Enhancement (Post-Alignment)
            if self.use_intensity_enhancement and self.n_modal > 0 and xs_img_aligned[0] is not None:
                if k < len(self.rgb_intensity_fuse):
                    base_feat_for_fusion = self.rgb_intensity_fuse[k](base_feat_for_fusion, xs_img_aligned[0])
                    dim_consumed_this_stage = True
                else: logger.warning(f"Stage {k}: StageWiseRGBIntensityFusion module missing.")
            
            # ❽ Main Fusion (Now only Sigmoid Gating)
            modalities_to_fuse = []
            modalities_indices = []
            for m in range(self.n_modal):
                if xs_img_aligned[m] is None: continue
                if dim_consumed_this_stage and m == 0: continue
                modalities_to_fuse.append(xs_img_aligned[m])
                modalities_indices.append(m)

            if not modalities_to_fuse:
                fused = base_feat_for_fusion
                logger.debug(f"    No remaining modalities for main fusion. Output is base feature.")
            else:
                logger.debug(f"    Applying fusion combination: {self.fusion_combination}")
                
                # --- Existing Fusion Methods ---
                if self.fusion_combination == "sequential":
                    fused = base_feat_for_fusion
                    logger.debug(f"      Starting sequential fusion with base: {get_tensor_info(fused, 'base')}")
                    # Fuses base(RGB+DIM) first with T (idx 1), then result with UV (idx 2)
                    for i, mod_idx in enumerate(modalities_indices):
                        logger.debug(f"      Fusing sequentially with modality index {mod_idx} ({i+1}/{len(modalities_indices)})...")
                        fused = self._apply_single_fusion(k, mod_idx, fused, modalities_to_fuse[i])
                        logger.debug(f"      Result after fusing mod {mod_idx}: {get_tensor_info(fused, 'fused')}")
    
                elif self.fusion_combination == "parallel_sum":
                    logger.debug(f"      Starting parallel_sum fusion...")
                    # Calculates Fuse(Base, T) and Fuse(Base, UV) separately
                    parts = [self._apply_single_fusion(k, mod_idx, base_feat_for_fusion, modalities_to_fuse[i])
                             for i, mod_idx in enumerate(modalities_indices)]
                    logger.debug(f"      Calculated {len(parts)} parts by applying fusion to base.")
                    fused = base_feat_for_fusion + sum(parts)
                    logger.warning("Parallel_sum logic might need review based on _apply_single_fusion's exact behavior.")
                    logger.debug(f"      Result (Base + Sum(Parts)): {get_tensor_info(fused, 'fused')}")
    
                elif self.fusion_combination == "parallel_avg":
                    logger.debug(f"      Starting parallel_avg fusion...")
                     # Calculates Fuse(Base, T) and Fuse(Base, UV) separately
                    parts = [self._apply_single_fusion(k, mod_idx, base_feat_for_fusion, modalities_to_fuse[i])
                             for i, mod_idx in enumerate(modalities_indices)]
                    logger.debug(f"      Calculated {len(parts)} parts by applying fusion to base.")
                    all_parts_for_avg = [base_feat_for_fusion] + parts
                    logger.debug(f"      Averaging {len(all_parts_for_avg)} tensors (base + parts)...")
                    fused = torch.mean(torch.stack(all_parts_for_avg, dim=0), dim=0)
                    logger.debug(f"      Result (Avg(Base + Parts)): {get_tensor_info(fused, 'fused')}")
    
                elif self.fusion_combination == "concat":
                    logger.debug(f"      Starting concat fusion...")
                    # Concatenates Base(RGB+DIM), T, UV
                    # ... (rest of concat logic) ...
                    if k < len(self.concat_fuse):
                        all_feats_to_concat = [base_feat_for_fusion] + modalities_to_fuse
                        logger.debug(f"      Concatenating {len(all_feats_to_concat)} features (base + selected modalities T, UV)...")
                        cat_feats = torch.cat(all_feats_to_concat, dim=1)
                        logger.debug(f"      Concatenated features: {get_tensor_info(cat_feats, 'cat_feats')}")
                        expected_channels = getattr(self.concat_fuse[k], 'in_channels', None)
                        logger.debug(f"      Concat module self.concat_fuse[{k}] expects channels: {expected_channels}")
                        if expected_channels is not None and cat_feats.shape[1] == expected_channels:
                             logger.debug(f"      Applying concat_fuse module...")
                             fused = self.concat_fuse[k](cat_feats)
                        elif expected_channels is None:
                             logger.debug(f"      Concat module for stage {k} does not have in_channels, attempting call anyway.")
                             try:
                                fused = self.concat_fuse[k](cat_feats)
                             except Exception as e:
                                logger.warning(f"Concat forward failed stage {k}: {e}. Falling back to sum.")
                                fused = base_feat_for_fusion + sum(modalities_to_fuse) # Sum T and UV only
                        else:
                             logger.warning(f"Concat channel mismatch stage {k} (expected {expected_channels}, got {cat_feats.shape[1]}). Falling back to sum.")
                             fused = base_feat_for_fusion + sum(modalities_to_fuse) # Sum T and UV only
                        logger.debug(f"      Concat result: {get_tensor_info(fused, 'fused')}")
                    else:
                         logger.warning(f"Concat module missing stage {k}. Falling back to sum.")
                         fused = base_feat_for_fusion + sum(modalities_to_fuse) # Sum T and UV only
                         logger.debug(f"      Concat fallback sum result: {get_tensor_info(fused, 'fused')}")
    
    
                elif self.fusion_combination == "hierarchical_extra":
                     # Fuses T and UV first, then fuses result with Base(RGB+DIM)
                     logger.debug(f"      Starting hierarchical_extra fusion...")
                     # ... (rest of hierarchical logic) ...
                     if k < len(self.extra_fuse):
                         # Check if there are at least 2 modalities left (T and UV)
                         if len(modalities_to_fuse) >= 2:
                             # Fuse remaining modalities together first (T with UV in this case)
                             extra_f = modalities_to_fuse[0] # Starts with T (index 1)
                             logger.debug(f"      Starting hierarchical fusion with mod {modalities_indices[0]}: {get_tensor_info(extra_f, 'extra_f')}")
                             num_extra_modalities_to_fuse = len(modalities_to_fuse)
                             num_extra_fusion_modules_needed = num_extra_modalities_to_fuse - 1 # Should be 1 here
                             logger.debug(f"      Need {num_extra_fusion_modules_needed} extra fusion steps.")
    
                             if num_extra_fusion_modules_needed > 0:
                                 if num_extra_fusion_modules_needed <= len(self.extra_fuse[k]):
                                     for i in range(num_extra_fusion_modules_needed):
                                         next_mod_idx = modalities_indices[i+1] # Should be UV (index 2)
                                         logger.debug(f"      Hierarchically fusing with mod {next_mod_idx} using extra_fuse[{k}][{i}]...")
                                         # Fuse T with UV using extra_fuse[k][0]
                                         extra_f = self.extra_fuse[k][i](extra_f, modalities_to_fuse[i+1])
                                         logger.debug(f"        Result after extra fuse step {i}: {get_tensor_info(extra_f, 'extra_f')}")
                                 else:
                                     logger.warning(f"Not enough extra_fuse modules stage {k} (needed {num_extra_fusion_modules_needed}, have {len(self.extra_fuse[k])}). Stopping hierarchical fusion early.")
                             else:
                                logger.debug(f"      No extra hierarchical steps needed (only one modality left to fuse).") # Should not happen if T, UV present
    
                             # Fuse Base(RGB+DIM) with the fused T+UV features
                             last_mod_idx_in_hierarchical = modalities_indices[-1] # index 2 (UV)
                             logger.debug(f"      Performing final fusion of base with hierarchical result (extra_f) using single fusion for index {last_mod_idx_in_hierarchical}...")
                             fused = self._apply_single_fusion(k, last_mod_idx_in_hierarchical, base_feat_for_fusion, extra_f)
                             logger.warning("Hierarchical fusion's final merge step using _apply_single_fusion might need review.")
                             logger.debug(f"      Hierarchical result: {get_tensor_info(fused, 'fused')}")
    
                         elif len(modalities_to_fuse) == 1: # Only T or UV remained
                             logger.debug(f"      Only one modality ({modalities_indices[0]}) remaining, applying single fusion.")
                             fused = self._apply_single_fusion(k, modalities_indices[0], base_feat_for_fusion, modalities_to_fuse[0])
                         else: # Should not happen if filtering is correct
                              logger.warning(f"Hierarchical fusion called with no modalities to fuse?")
                              fused = base_feat_for_fusion
    
                     else:
                         logger.warning(f"Extra fuse module missing stage {k}. Falling back to sum.")
                         fused = base_feat_for_fusion + sum(modalities_to_fuse) # Sum T + UV
                         logger.debug(f"      Hierarchical fallback sum result: {get_tensor_info(fused, 'fused')}")
    
    
                elif self.fusion_combination == "attention_gating":
                    # Applies softmax attention over Base(RGB+DIM), T, UV
                    logger.debug(f"      Starting attention_gating fusion...")
                    # ... (rest of attention gating logic) ...
                    if k < len(self.gates):
                         all_feats_to_gate = [base_feat_for_fusion] + modalities_to_fuse # [Base, T, UV]
                         num_feats_to_gate = len(all_feats_to_gate)
                         logger.debug(f"      Concatenating {num_feats_to_gate} features for gating...")
                         try:
                             cat_feats = torch.cat(all_feats_to_gate, dim=1)
                             logger.debug(f"      Concatenated features for gate: {get_tensor_info(cat_feats, 'cat_feats_gate')}")
                             gate_module = self.gates[k]
                             input_layer_for_check = gate_module[0] if isinstance(gate_module, nn.Sequential) else gate_module
                             expected_channels = getattr(input_layer_for_check, 'in_channels', None)
                             logger.debug(f"      Gate module self.gates[{k}] expects channels: {expected_channels}")
    
                             if expected_channels is not None and cat_feats.shape[1] != expected_channels:
                                 logger.warning(f"Gate input channel mismatch stage {k} (expected {expected_channels}, got {cat_feats.shape[1]}). Falling back to sum.")
                                 fused = base_feat_for_fusion + sum(modalities_to_fuse) # Sum T + UV
                             else:
                                 logger.debug(f"      Applying gate module self.gates[{k}]...")
                                 weights = self.gates[k](cat_feats) # Expects [B, num_feats_to_gate, H, W] output
                                 logger.debug(f"      Gate output weights: {get_tensor_info(weights, 'weights')}")
    
                                 if weights.shape[1] == num_feats_to_gate:
                                     logger.debug(f"      Applying weights and summing...")
                                     weighted_features = [weights[:, i:i+1, :, :] * all_feats_to_gate[i] for i in range(num_feats_to_gate)]
                                     fused = torch.sum(torch.stack(weighted_features, dim=0), dim=0)
                                 else:
                                     logger.warning(f"Gate weight channel output mismatch stage {k} (expected {num_feats_to_gate}, got {weights.shape[1]}). Falling back to sum.")
                                     fused = base_feat_for_fusion + sum(modalities_to_fuse) # Sum T + UV
                             logger.debug(f"      Attention gating result: {get_tensor_info(fused, 'fused')}")
    
                         except Exception as e:
                             logger.warning(f"Attention Gating forward failed stage {k}: {e}. Falling back to sum.")
                             fused = base_feat_for_fusion + sum(modalities_to_fuse) # Sum T + UV
                         logger.debug(f"      Attention gating result (incl. potential fallback): {get_tensor_info(fused, 'fused')}")
    
                    else:
                         logger.warning(f"Gate module missing stage {k}. Falling back to sum.")
                         fused = base_feat_for_fusion + sum(modalities_to_fuse) # Sum T + UV
                         logger.debug(f"      Attention gating fallback sum result: {get_tensor_info(fused, 'fused')}")


                # <<< --- NEW SIGMOID GATING BLOCK --- >>>
                elif self.fusion_combination == "sigmoid_gating":
                    # --- GENERALIZED SIGMOID GATING FUSION LOGIC (only fusion path) ---
                    logger.debug(f"      Applying generalized sigmoid_gating fusion...")
                    fused = base_feat_for_fusion
                
                    modality_offset = 1 if self.use_intensity_enhancement else 0
                    gated_contributions = []
                
                    for i, mod_idx in enumerate(modalities_indices):
                        if self.use_intensity_enhancement and mod_idx == 0:
                            # Already consumed by enhancement
                            continue
                        mod_feat = modalities_to_fuse[i]
                        gate_idx = mod_idx - modality_offset
                        if k < len(self.sigmoid_gates) and gate_idx < len(self.sigmoid_gates[k]):
                            gate_module = self.sigmoid_gates[k][gate_idx]
                            contrib = gate_module(base_feat_for_fusion, mod_feat)
                            gated_contributions.append(contrib)
                            logger.debug(f"        Added gated contribution for modality {mod_idx} at stage {k} ({get_tensor_info(contrib)})")
                        else:
                            logger.warning(f"Sigmoid gate module missing for stage {k}, modality {mod_idx}. Skipping.")
                
                    if self.sgate_fusion_mode == "add":
                        if gated_contributions:
                            fused = fused + sum(gated_contributions)
                    elif self.sgate_fusion_mode == "avg":
                        if gated_contributions:
                            avg_gated_contrib = torch.mean(torch.stack(gated_contributions, dim=0), dim=0) if len(gated_contributions) > 1 else gated_contributions[0]
                            fused = fused + avg_gated_contrib
                    else:  # Fallback
                        logger.warning(f"Sigmoid gating encountered sgate_fusion_mode '{self.sgate_fusion_mode}'. Defaulting to 'add' behavior.")
                        if gated_contributions:
                            fused = fused + sum(gated_contributions)
                
                    logger.debug(f"      Sigmoid gating final result: {get_tensor_info(fused)}")
                # --- END GENERALIZED SIGMOID GATING FUSION LOGIC ---
            
            outs.append(fused)
            x_rgb = fused 
            current_xs_features = xs_img_aligned 
            
        logger.debug("="*20 + " Exiting forward_features " + "="*20)
        return outs

    def forward(self, x_rgb, x_e):
        if isinstance(x_e, torch.Tensor):
             # Use self.extra_in_chans for splitting, which is defined during init
             # ch_counts = [1 if single else 3 for single in cfg.x_is_single_channel] # Old way based on global cfg
             if not hasattr(self, 'extra_in_chans') or not self.extra_in_chans:
                 raise ValueError("self.extra_in_chans is not defined or empty. Cannot split x_e tensor.")
             ch_counts = self.extra_in_chans
             xs = list(x_e.split(ch_counts, dim=1)) # dim=1 is channel if x_e is BCHW
        elif isinstance(x_e, (list, tuple)):
             xs = list(x_e)
        else:
             raise TypeError(f"Unsupported type for x_e: {type(x_e)}")
        out = self.forward_features(x_rgb, xs)
        return out

# --- load_dualpath_model and mit_bX model variants remain the same ---

def load_dualpath_model(model, model_file):
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict: raw_state_dict = raw_state_dict['model']
        elif 'state_dict' in raw_state_dict: raw_state_dict = raw_state_dict['state_dict']
    elif isinstance(model_file, dict):
         raw_state_dict = model_file
    else: raise TypeError('pretrained must be a str or a dict')
    new_state_dict = model.state_dict(); loaded_keys_count = 0; skipped_keys_count = 0
    model_keys = set(new_state_dict.keys())
    for k_pretrained, v_pretrained in raw_state_dict.items():
        k_model = k_pretrained.replace('module.', '')
        if k_model in new_state_dict and new_state_dict[k_model].shape == v_pretrained.shape:
            new_state_dict[k_model] = v_pretrained; loaded_keys_count += 1; model_keys.remove(k_model)
        elif k_model.startswith(('patch_embed', 'block', 'norm')):
            parts = k_model.split('.', 1)
            if len(parts) == 2:
                 base_name, rest = parts; k_extra = f"extra_{base_name}.{rest}"
                 if k_extra in new_state_dict and new_state_dict[k_extra].shape == v_pretrained.shape:
                     new_state_dict[k_extra] = v_pretrained; loaded_keys_count += 1
                     if k_extra in model_keys: model_keys.remove(k_extra)
                     if k_model in model_keys and new_state_dict[k_model].shape == v_pretrained.shape:
                         new_state_dict[k_model] = v_pretrained; model_keys.remove(k_model)
                 elif k_model in model_keys and new_state_dict[k_model].shape == v_pretrained.shape:
                     new_state_dict[k_model] = v_pretrained; loaded_keys_count += 1; model_keys.remove(k_model)
                 else: skipped_keys_count += 1
            else: skipped_keys_count += 1
        else: skipped_keys_count += 1
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    truly_missing = [k for k in missing_keys if k in model_keys]
    logger.info(f"Loaded {loaded_keys_count}, Skipped {skipped_keys_count}, Missing {len(truly_missing)}, Unexpected {len(unexpected_keys)}")
    if truly_missing: logger.warning(f"  Missing: {truly_missing}")
    if unexpected_keys: logger.warning(f"  Unexpected: {unexpected_keys}")
    del raw_state_dict, new_state_dict; logger.info(f"Load time: {time.time() - t_start:.2f}s")


class mit_b0(RGBXTransformer):
    def __init__(
        self,
        img_size: int,
        in_chans: int, # Typically 3 for RGB
        extra_in_chans: List[int],
        n_modal: int = 1,
        # fuse_cfg=None, # This was in mit_b2 signature but not used in its super call
        alignment_method: str = "frm",
        fusion_method: str = "ffm",
        fusion_combination: str = "sigmoid_gating", # Consistent with previous changes
        norm_fuse=nn.BatchNorm2d,
        use_intensity_enhancement: bool = False,
        num_classes: int = 1000, # Default from RGBXTransformer
        sgate_fusion_mode: str = "add", # Default from RGBXTransformer
        **kwargs
    ):
        # mit_b0 specific architectural parameters
        embed_dims = [32, 64, 160, 256]
        num_heads = [1, 2, 5, 8]
        mlp_ratios = [4, 4, 4, 4]
        depths = [2, 2, 2, 2]
        sr_ratios = [8, 4, 2, 1]
        b0_patch_size = 4 # Original mit_b0 super call used patch_size=4

        super(mit_b0, self).__init__(
            n_modal=n_modal,
            img_size=img_size,
            patch_size=b0_patch_size, # mit_b0 specific value
            in_chans=in_chans,
            extra_in_chans=extra_in_chans,
            num_classes=num_classes,
            embed_dims=embed_dims,      # mit_b0 specific
            num_heads=num_heads,        # mit_b0 specific (though same as b2 for these heads)
            mlp_ratios=mlp_ratios,      # mit_b0 specific (though same as b2)
            qkv_bias=kwargs.get('qkv_bias', True), # Default from other MiT variants
            norm_layer=kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6)), # Default
            norm_fuse=norm_fuse,
            depths=depths,              # mit_b0 specific
            sr_ratios=sr_ratios,        # mit_b0 specific (though same as b2)
            alignment_method=alignment_method,
            fusion_method=fusion_method,
            fusion_combination=fusion_combination,
            use_intensity_enhancement=use_intensity_enhancement,
            sgate_fusion_mode=sgate_fusion_mode,
            drop_rate=kwargs.get('drop_rate', 0.0),
            attn_drop_rate=kwargs.get('attn_drop_rate', 0.0),
            drop_path_rate=kwargs.get('drop_path_rate', 0.1),
            **kwargs # Pass through any other relevant kwargs for RGBXTransformer
        )

class mit_b1(RGBXTransformer): 
    def __init__(
        self,
        img_size: int,
        in_chans: int, # Typically 3 for RGB
        extra_in_chans: List[int],
        n_modal: int = 1,
        # fuse_cfg=None, # This was in mit_b2 signature but not used in its super call
        alignment_method: str = "frm",
        fusion_method: str = "ffm",
        fusion_combination: str = "sigmoid_gating", # Consistent with previous changes
        norm_fuse=nn.BatchNorm2d,
        use_intensity_enhancement: bool = False,
        num_classes: int = 1000, # Default from RGBXTransformer
        sgate_fusion_mode: str = "add", # Default from RGBXTransformer
        **kwargs
    ):

        # mit_b1 specific architectural parameters
        embed_dims = [64, 128, 320, 512] # B1 uses these
        num_heads = [1, 2, 5, 8]
        mlp_ratios = [4, 4, 4, 4]
        depths = [2, 2, 2, 2] # B1 depths
        sr_ratios = [8, 4, 2, 1]
        b1_patch_size = 4
        
        super(mit_b1, self).__init__(
            n_modal=n_modal,
            img_size=img_size,
            patch_size=b1_patch_size, # mit_b0 specific value
            in_chans=in_chans,
            extra_in_chans=extra_in_chans,
            num_classes=num_classes,
            embed_dims=embed_dims,      # mit_b0 specific
            num_heads=num_heads,        # mit_b0 specific (though same as b2 for these heads)
            mlp_ratios=mlp_ratios,      # mit_b0 specific (though same as b2)
            qkv_bias=kwargs.get('qkv_bias', True), # Default from other MiT variants
            norm_layer=kwargs.get('norm_layer', partial(nn.LayerNorm, eps=1e-6)), # Default
            norm_fuse=norm_fuse,
            depths=depths,              # mit_b0 specific
            sr_ratios=sr_ratios,        # mit_b0 specific (though same as b2)
            alignment_method=alignment_method,
            fusion_method=fusion_method,
            fusion_combination=fusion_combination,
            use_intensity_enhancement=use_intensity_enhancement,
            sgate_fusion_mode=sgate_fusion_mode,
            drop_rate=kwargs.get('drop_rate', 0.0),
            attn_drop_rate=kwargs.get('attn_drop_rate', 0.0),
            drop_path_rate=kwargs.get('drop_path_rate', 0.1),
            **kwargs # Pass through any other relevant kwargs for RGBXTransformer
            )

class mit_b2(RGBXTransformer):
    def __init__(
        self,
        img_size: int,
        in_chans: int,
        extra_in_chans: List[int],
        n_modal: int = 1,
        fuse_cfg=None,
        alignment_method: str = "frm",
        fusion_method: str = "ffm",
        fusion_combination: str = "sigmoid_gating", # Changed default
        norm_fuse=nn.BatchNorm2d,
        use_intensity_enhancement = False, 
        num_classes=1000,
        sgate_fusion_mode = "avg",
        **kwargs
    ):
        embed_dims=[64, 128, 320, 512]; num_heads=[1, 2, 5, 8]; depths=[3, 4, 6, 3]; sr_ratios=[8, 4, 2, 1]; mlp_ratios=[4, 4, 4, 4]
        super(mit_b2, self).__init__(
            n_modal=n_modal, img_size=img_size, patch_size=7, in_chans=in_chans, extra_in_chans=extra_in_chans,
            num_classes=num_classes, embed_dims=embed_dims, num_heads=num_heads, mlp_ratios=mlp_ratios,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_fuse=norm_fuse, depths=depths, sr_ratios=sr_ratios,
            alignment_method=alignment_method, fusion_method=fusion_method, fusion_combination=fusion_combination,
            use_intensity_enhancement=use_intensity_enhancement, sgate_fusion_mode=sgate_fusion_mode,
            drop_rate=kwargs.get('drop_rate', 0.0), attn_drop_rate=kwargs.get('attn_drop_rate', 0.0), drop_path_rate=kwargs.get('drop_path_rate', 0.1),
            **kwargs)

class mit_b3(RGBXTransformer):
    def __init__(self, fuse_cfg=None, alignment_method="frm", fusion_method="ffm", use_intensity_enhancement=False, fusion_combination="sigmoid_gating", **kwargs):
        extra_chans = kwargs.get('extra_in_chans', [1] * kwargs.get('n_modal', 1))
        super(mit_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, alignment_method=alignment_method, fusion_method=fusion_method,
            extra_in_chans=extra_chans, use_intensity_enhancement=use_intensity_enhancement, fusion_combination=fusion_combination, **kwargs)

class mit_b4(RGBXTransformer):
    def __init__(self, fuse_cfg=None, alignment_method="frm", fusion_method="ffm", use_intensity_enhancement=False, fusion_combination="sigmoid_gating", **kwargs):
        extra_chans = kwargs.get('extra_in_chans', [1] * kwargs.get('n_modal', 1))
        super(mit_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, alignment_method=alignment_method, fusion_method=fusion_method,
            extra_in_chans=extra_chans, use_intensity_enhancement=use_intensity_enhancement, fusion_combination=fusion_combination, **kwargs)

class mit_b5(RGBXTransformer):
    def __init__(self, fuse_cfg=None, alignment_method="frm", fusion_method="ffm", use_intensity_enhancement=False, fusion_combination="sigmoid_gating", **kwargs):
        extra_chans = kwargs.get('extra_in_chans', [1] * kwargs.get('n_modal', 1))
        super(mit_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, alignment_method=alignment_method, fusion_method=fusion_method,
            extra_in_chans=extra_chans, use_intensity_enhancement=use_intensity_enhancement, fusion_combination=fusion_combination, **kwargs)
