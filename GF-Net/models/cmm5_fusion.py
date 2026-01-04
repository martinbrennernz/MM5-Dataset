#!/usr/bin/env python3
"""
Created on Fri Apr 18 19:06:14 2025

@author: Martin Brenner
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math # Added for sqrt in init

class StageWiseRGBIntensityFusion(nn.Module):
    """
    Fuses RGB and Intensity features at a specific stage using learned convolutions.
    Outputs an enhanced RGB feature map.
    """
    def __init__(self, dim, hidden_dim_ratio=0.5):
        """
        Args:
            dim (int): Number of channels for the input/output features (embedding dim for the stage).
            hidden_dim_ratio (float): Ratio to determine the hidden dimension.
        """
        super().__init__()
        self.dim = dim
        hidden_dim = max(16, int(dim * hidden_dim_ratio)) # Ensure a minimum hidden dim

        # Simple fusion network: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Conv
        # Takes concatenated [RGB, Intensity] features
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim * 2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
             nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True) # Output enhanced RGB features
        )

        # Optional: Add a residual connection from the original RGB
        self.use_residual = True

        self.apply(self._init_weights) # Apply standard weight init

    def _init_weights(self, m):
        """Initialize weights for Conv2d and BatchNorm2d."""
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
             trunc_normal_(m.weight, std=.02)
             if isinstance(m, nn.Linear) and m.bias is not None:
                 nn.init.constant_(m.bias, 0)

    def forward(self, rgb_feat, intensity_feat):
        """
        Args:
            rgb_feat (torch.Tensor): RGB features [B, C, H, W] for the current stage.
            intensity_feat (torch.Tensor): Intensity features [B, C, H, W] for the current stage.

        Returns:
            torch.Tensor: Enhanced RGB features [B, C, H, W].
        """
        # Concatenate along the channel dimension
        x = torch.cat([rgb_feat, intensity_feat], dim=1) # [B, 2*C, H, W]

        # Pass through fusion network
        fused_enhancement = self.fusion_conv(x) # [B, C, H, W]

        # Add residual connection
        if self.use_residual:
            enhanced_rgb = rgb_feat + fused_enhancement
        else:
            enhanced_rgb = fused_enhancement

        return enhanced_rgb



class RGBEnhancementFusion(nn.Module):
    """
    Enhances RGB features using Intensity and Depth features via learned modulations.
    Assumes RGB, Intensity (I), and Depth (D) features are input.
    Outputs enhanced RGB features.
    """
    def __init__(self, dim, hidden_dim_ratio=0.5):
        """
        Args:
            dim (int): Number of channels for the input/output features.
            hidden_dim_ratio (float): Ratio to determine the hidden dimension
                                      of the modulation networks.
        """
        super().__init__()
        self.dim = dim
        hidden_dim = max(16, int(dim * hidden_dim_ratio)) # Ensure a minimum hidden dim

        # Small Conv networks to generate modulation signals from Intensity and Depth
        # Output channels = 2 * dim (dim for additive, dim for multiplicative)
        self.mod_net_I = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim), # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2 * dim, kernel_size=1, bias=True) # Use bias here
        )
        self.mod_net_D = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim), # Added BatchNorm
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 2 * dim, kernel_size=1, bias=True) # Use bias here
        )

        # Initialize the final conv layers to output near-zero modulations initially
        # This makes the module start closer to an identity mapping for RGB
        nn.init.zeros_(self.mod_net_I[-1].weight)
        nn.init.zeros_(self.mod_net_I[-1].bias)
        nn.init.zeros_(self.mod_net_D[-1].weight)
        nn.init.zeros_(self.mod_net_D[-1].bias)

        # Optional: A final conv layer for the enhanced RGB before output
        # self.final_conv = nn.Conv2d(dim, dim, kernel_size=1)

        self.apply(self._init_weights) # Apply standard weight init to other layers

    def _init_weights(self, m):
        """Initialize weights for Conv2d and BatchNorm2d."""
        if isinstance(m, nn.Conv2d):
            # Kaiming He initialization for Conv2d
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            # Initialize BatchNorm params
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear): # Added for completeness if Linear layers are used elsewhere
             trunc_normal_(m.weight, std=.02)
             if isinstance(m, nn.Linear) and m.bias is not None:
                 nn.init.constant_(m.bias, 0)


    def forward(self, rgb_feat, intensity_feat, depth_feat):
        """
        Args:
            rgb_feat (torch.Tensor): RGB features [B, C, H, W]
            intensity_feat (torch.Tensor): Intensity features [B, C, H, W]
            depth_feat (torch.Tensor): Depth features [B, C, H, W]

        Returns:
            torch.Tensor: Enhanced RGB features [B, C, H, W]
        """
        # Generate modulations from Intensity and Depth
        mod_I = self.mod_net_I(intensity_feat) # [B, 2C, H, W]
        mod_D = self.mod_net_D(depth_feat)     # [B, 2C, H, W]

        # Combine modulations (simple addition, could be concatenation + conv)
        combined_mod = mod_I + mod_D

        # Split into additive and multiplicative parts
        add_mod, mul_mod = combined_mod.chunk(2, dim=1) # Each is [B, C, H, W]

        # Apply modulation to RGB features
        # Sigmoid ensures multiplicative factor is between 0 and 1
        # Tanh allows additive factor to be positive or negative [-1, 1] - adjust scaling if needed
        enhanced_rgb = rgb_feat * torch.sigmoid(mul_mod) + torch.tanh(add_mod)

        # Add residual connection from original RGB
        final_enhanced_rgb = rgb_feat + enhanced_rgb

        # Optional: Apply final conv
        # final_enhanced_rgb = self.final_conv(final_enhanced_rgb)

        return final_enhanced_rgb

class SigmoidGateModule(nn.Module):
    """
    Applies an individual sigmoid gate to a modality based on concatenated features.

    Takes a base feature map and a modality feature map, calculates a gate
    value for the modality, transforms the modality feature, and returns
    the gated contribution (gate * transformed_modality).
    """
    def __init__(self, base_channels, mod_channels, output_channels=None, reduction=4):
        """
        Args:
            base_channels (int): Number of channels in the base feature map.
            mod_channels (int): Number of channels in the modality feature map.
            output_channels (int, optional): Number of channels for the transformed
                                             modality feature. If None, defaults to
                                             base_channels.
            reduction (int): Reduction factor for the intermediate layer in the gate MLP.
                             Set to None or 1 to disable reduction MLP (use 1x1 conv only).
        """
        super().__init__()
        if output_channels is None:
            output_channels = base_channels

        self.output_channels = output_channels
        gate_input_channels = base_channels + mod_channels
        
        if reduction is not None and reduction > 1:
            # Simple MLP for gate prediction with reduction
            self.gate_mlp = nn.Sequential(
                nn.Conv2d(gate_input_channels, gate_input_channels // reduction, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(gate_input_channels // reduction, 1, 1, bias=True),
                nn.Sigmoid()
            )
        else:
             # Single 1x1 Conv for gate prediction
             self.gate_mlp = nn.Sequential(
                 nn.Conv2d(gate_input_channels, 1, 1, bias=True),
                 nn.Sigmoid()
             )

        # Optional 1x1 convolution to transform the modality features
        # before applying the gate (e.g., match channels or refine)
        if mod_channels != output_channels:
            self.transform_conv = nn.Conv2d(mod_channels, output_channels, 1, bias=False)
        else:
            self.transform_conv = nn.Identity() # No transform needed if channels match

    def forward(self, base_feat, mod_feat):
        """
        Args:
            base_feat (torch.Tensor): Base feature map (e.g., fused RGB+I+D).
            mod_feat (torch.Tensor): Feature map of the modality to be gated (e.g., Aligned T).

        Returns:
            torch.Tensor: The gated contribution (gate * transformed_modality).
        """
        # Concatenate base and modality features
        concat_feat = torch.cat((base_feat, mod_feat), dim=1)

        # Calculate gate values (spatial attention map)
        gate = self.gate_mlp(concat_feat) # Shape: [B, 1, H, W]

        # Transform modality features (optional)
        transformed_mod_feat = self.transform_conv(mod_feat) # Shape: [B, output_channels, H, W]

        # Apply gate and return contribution
        gated_contribution = gate * transformed_mod_feat
        return gated_contribution

class SpatialContributionAttention(nn.Module):
    def __init__(self, base_channels, mod_channels, reduction=4):
        """
        Generates a spatial attention map for a modality contribution,
        guided by both the base feature and the modality itself.
        The output is attention_map * mod_feat.
        """
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Conv2d(base_channels + mod_channels, (base_channels + mod_channels) // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d((base_channels + mod_channels) // reduction, 1, 1, bias=True), # Output a single channel attention map
            nn.Sigmoid() # Attention map values between 0 and 1
        )
        # This module assumes mod_channels will be the same as the output channels of the contribution
        # No explicit transform_conv here, as SigmoidGateModule already produces the contrib

    def forward(self, base_feat, mod_contrib_feat):
        """
        Args:
            base_feat (torch.Tensor): Base feature map.
            mod_contrib_feat (torch.Tensor): Modality contribution feature map (e.g., thermal_contrib).
                                           This is *after* it has been processed by its own SigmoidGate.
        Returns:
            torch.Tensor: Spatially weighted modality contribution.
        """
        concat_feat = torch.cat((base_feat, mod_contrib_feat), dim=1)
        attention_map = self.gate_mlp(concat_feat) # [B, 1, H, W]
        return mod_contrib_feat * attention_map

class CrossAttention(nn.Module):
    """Bidirectional cross‑attention used in TransFuse."""
    def __init__(self, dim, n_heads=8, qkv_bias=True, attn_drop=0.):
        super().__init__()
        assert dim % n_heads == 0
        self.nh = n_heads
        self.dk = dim // n_heads
        self.scale = self.dk ** -0.5

        self.q   = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)  # for stream‑1
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)  # for stream‑2
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.q.weight, std=.02)
        trunc_normal_(self.kv1.weight, std=.02)
        trunc_normal_(self.kv2.weight, std=.02)
        nn.init.constant_(self.q.bias, 0.)
        nn.init.constant_(self.kv1.bias, 0.)
        nn.init.constant_(self.kv2.bias, 0.)
        trunc_normal_(self.proj.weight, std=.02)
        nn.init.constant_(self.proj.bias, 0.)

    def forward(self, x1, x2):
        """x1,x2: (B,C,H,W)   – assume same spatial size."""
        B,C,H,W = x1.shape
        N       = H*W
        x1 = x1.flatten(2).transpose(1,2)   # B N C
        x2 = x2.flatten(2).transpose(1,2)

        q1 = self.q(x1).view(B,N,self.nh,self.dk).permute(0,2,1,3)          # B h N d
        q2 = self.q(x2).view(B,N,self.nh,self.dk).permute(0,2,1,3)

        k1,v1 = self.kv1(x1).view(B,N,2,self.nh,self.dk).permute(2,0,3,1,4) # (2,B,h,N,d)
        k2,v2 = self.kv2(x2).view(B,N,2,self.nh,self.dk).permute(2,0,3,1,4)

        # attention logits
        attn12 = (q1 @ k2.transpose(-2,-1)) * self.scale   # stream‑1 attends to 2
        attn21 = (q2 @ k1.transpose(-2,-1)) * self.scale   # vice versa
        # (optionally multiply by –1 like TransFuse for dissimilarity)
        a12 = self.attn_drop(attn12.softmax(-1))
        a21 = self.attn_drop(attn21.softmax(-1))

        y1 = (a12 @ v2).transpose(1,2).reshape(B,N,C)      # B N C
        y2 = (a21 @ v1).transpose(1,2).reshape(B,N,C)

        y1 = self.proj(y1).transpose(1,2).reshape(B,C,H,W) # B C H W
        y2 = self.proj(y2).transpose(1,2).reshape(B,C,H,W)
        return y1 + y2        # final fused & aligned feature


class CoarseRegistration(nn.Module):
    """
    A 4D version of coarse registration.
    Expects src_feat_4d, tgt_feat_4d of shape [B, C, H, W].
    Outputs aligned_src_feat_4d of shape [B, C, H, W].
    """
    def __init__(self, input_channels):
        super(CoarseRegistration, self).__init__()

        # input_channels * 2 because we concat the two modalities
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels * 2, 64, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6)
        )
        # Initialize the affine transform to the identity
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, src_feat_4d, tgt_feat_4d):
        """
        src_feat_4d, tgt_feat_4d: [B, C, H, W]
        """
        # 1) Concat along channel dimension => [B, 2*C, H, W]
        x = torch.cat([src_feat_4d, tgt_feat_4d], dim=1)

        # 2) Pass through localization conv + fc
        x = self.localization(x)                 # -> [B, 128, 1, 1]
        x = x.view(x.size(0), -1)               # -> [B, 128]
        theta = self.fc_loc(x).view(-1, 2, 3)   # -> [B, 2, 3]

        # 3) Build the sampling grid for src_feat_4d size
        grid = F.affine_grid(
            theta,
            size=src_feat_4d.size(),  # same shape as src_feat_4d
            align_corners=False
        )

        # 4) Warp the source feature
        aligned_src_feat_4d = F.grid_sample(
            src_feat_4d, grid, align_corners=False
        )

        return aligned_src_feat_4d, theta


class TPSRegistration(nn.Module):
    """
    A Thin‐Plate‐Spline (TPS) STN module.
    Given source and target feature‐maps [B,C,H,W],
    predicts control‐point offsets and warps src → tgt via TPS.
    """
    def __init__(self,
                 input_channels: int,
                 num_ctrl_pts: int = 5,   # e.g. 5×5 control grid
                 hidden_dim: int = 128):
        super().__init__()
        self.num_ctrl_pts = num_ctrl_pts
        self.grid_size = num_ctrl_pts

        # Localization network: from concatenated feats → 2*(grid_size**2) offsets
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels * 2, hidden_dim, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * (num_ctrl_pts**2))
        )
        # initialize to identity: zero offsets
        nn.init.zeros_(self.localization[-1].weight)
        nn.init.zeros_(self.localization[-1].bias)

        # Build a fixed grid of control‐point centers in [−1,1]×[−1,1]
        # shape: (K*K, 2)
        self.register_buffer(
            "base_ctrl_pts",
            self._make_regular_grid(num_ctrl_pts).view(-1,2)
        )

    def forward(self,
                src_feat: torch.Tensor,
                tgt_feat: torch.Tensor):
        """
        src_feat, tgt_feat: [B, C, H, W]
        returns: (warped_src, ctrl_pts)
          - warped_src: TPS‐warped version of src_feat
          - ctrl_pts: [B, K*K, 2] absolute control points (in normalized coords)
        """
        B, C, H, W = src_feat.shape

        # 1) predict control‐point offsets
        x = torch.cat([src_feat, tgt_feat], dim=1)            # [B,2C,H,W]
        delta = self.localization(x)                          # [B, 2*K*K]
        delta = delta.view(B, -1, 2)                          # [B, K*K, 2]

        # 2) absolute control points
        ctrl_pts = self.base_ctrl_pts.unsqueeze(0) + delta    # [B, K*K, 2]

        # 3) compute TPS sampling grid with our own helper
        sampling_grid = self._compute_tps_grid(ctrl_pts, H, W)  # [B, H, W, 2]

        # 4) warp the source feature
        warped = F.grid_sample(src_feat, sampling_grid, align_corners=False)
        return warped, ctrl_pts

    @staticmethod
    def _make_regular_grid(K: int, device=None):
        """Create K×K grid in normalized coords [−1,1]."""
        lin = torch.linspace(-1, 1, steps=K, device=device)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        return torch.stack([xx, yy], dim=-1)  # [K,K,2]

    def _compute_tps_grid(self,
                          ctrl_pts: torch.Tensor,
                          H: int, W: int) -> torch.Tensor:
        """
        Solve the TPS system for each batch element and produce
        a sampling grid of shape [B,H,W,2] in normalized coords.
        """
        B, K2, _ = ctrl_pts.shape
        K = int(K2**0.5)
        device = ctrl_pts.device
        eps = 1e-6

        # 1) pairwise distances between ctrl_pts → [B, K2, K2]
        diff = ctrl_pts.unsqueeze(2) - ctrl_pts.unsqueeze(1)  # [B,K2,K2,2]
        r = torch.norm(diff, dim=-1)                         # [B,K2,K2]
        U = r**2 * torch.log(r**2 + eps)                     # [B,K2,K2]

        # 2) build the LHS matrix L of shape [B, K2+3, K2+3]
        #    [ U   1   X ]
        #    [ 1^T 0   0 ]
        #    [ X^T 0   0 ]
        ones_K = torch.ones(B, K2, 1, device=device)         # [B,K2,1]
        pts = ctrl_pts                                      # [B,K2,2]

        # Top block: [ U | 1 | X ] -> [B, K2, K2+1+2]
        top = torch.cat([U, ones_K, pts], dim=2)

        # Middle block: [1^T | 0 | 0] -> [B,1,K2+1+2]
        ones_T = ones_K.transpose(1,2)                      # [B,1,K2]
        zero_mid1 = torch.zeros(B, 1, 1, device=device)     # [B,1,1]
        zero_mid2 = torch.zeros(B, 1, 2, device=device)     # [B,1,2]
        mid = torch.cat([ones_T, zero_mid1, zero_mid2], dim=2)

        # Bottom block: [X^T | 0 | 0] -> [B,2,K2+1+2]
        xt = pts.transpose(1,2)                             # [B,2,K2]
        zero_bot1 = torch.zeros(B, 2, 1, device=device)     # [B,2,1]
        zero_bot2 = torch.zeros(B, 2, 2, device=device)     # [B,2,2]
        bot = torch.cat([xt, zero_bot1, zero_bot2], dim=2)

        # Assemble L
        L = torch.cat([top, mid, bot], dim=1)               # [B, K2+3, K2+3]

        # 3) right-hand side: [pts; zeros(1×2); zeros(2×2)] -> [B, K2+3, 2]
        zero_rhs1 = torch.zeros(B, 1, 2, device=device)     # [B,1,2]
        zero_rhs2 = torch.zeros(B, 2, 2, device=device)     # [B,2,2]
        Y = torch.cat([pts, zero_rhs1, zero_rhs2], dim=1)

        # 4) solve for TPS coefficients
        coeffs = torch.linalg.solve(L, Y)                   # [B, K2+3, 2]

        # 5) build dense grid [H*W,2]
        lin_x = torch.linspace(-1, 1, W, device=device)
        lin_y = torch.linspace(-1, 1, H, device=device)
        yy, xx = torch.meshgrid(lin_y, lin_x, indexing='ij')
        grid = torch.stack((xx, yy), dim=-1)                # [H, W, 2]
        grid_flat = grid.view(-1, 2)                        # [H*W, 2]

        # 6) U for every pixel to control points: [B, H*W, K2]
        # diff2 = grid_flat.unsqueeze(0).unsqueeze(2) - ctrl_pts.unsqueeze(2)  # [B,K2,H*W,2]
        diff2 = ctrl_pts.unsqueeze(2) - grid_flat.unsqueeze(0).unsqueeze(1)  # -> [B, K2, P, 2]
        r2 = torch.norm(diff2, dim=-1)                   # [B,K2,H*W]
        U2 = r2**2 * torch.log(r2**2 + eps)              # [B,K2,H*W]
        U2 = U2.permute(0, 2, 1)                         # [B,H*W,K2]

        # concatenate per-pixel features: [U2 | 1 | xy] -> [B, H*W, K2+1+2]
        ones_pix = torch.ones(B, grid_flat.shape[0], 1, device=device)
        pix_feat = torch.cat([U2, ones_pix, grid_flat.unsqueeze(0).expand(B, -1, 2)], dim=2)

        # 7) warp coordinates and reshape to [B,H,W,2]
        warped_flat = pix_feat @ coeffs                  # [B, H*W, 2]
        sampling_grid = warped_flat.view(B, H, W, 2)

        return sampling_grid

class FineRegistrationFusion(nn.Module):
    """
    Expects aligned_modality_feat and rgb_feat in shape [B, C, H, W].
    Returns fused features in shape [B, C, H, W].
    """
    def __init__(self, input_channels):
        super(FineRegistrationFusion, self).__init__()
        # We concat => 2 * input_channels for conv1
        self.conv1 = nn.Conv2d(input_channels * 2, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, input_channels, kernel_size=3, padding=1)

    def forward(self, aligned_modality_feat, rgb_feat):
        # Concat along channels: [B, 2*input_channels, H, W]
        x = torch.cat([aligned_modality_feat, rgb_feat], dim=1)
        x = self.relu(self.conv1(x))
        fused_feat = self.conv2(x)  # [B, input_channels, H, W]
        return fused_feat


class DIPUpsampler(nn.Module):
    """
    Only used when I res < rgb resolution
    """
    def __init__(self, in_ch=1, out_ch=1, num_layers=5, hidden=64):
        super().__init__()
        layers = [nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers-2):
            layers += [nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(hidden, out_ch, 3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, target_size=None):
        # x: (B,1,h,w)
        if target_size is not None and (x.shape[-2], x.shape[-1]) != target_size:
            x = F.interpolate(x, size=target_size,
                              mode='bilinear', align_corners=False)
        return self.net(x)

class CrossModalTransformer(nn.Module):
    def __init__(self, dim, depth=4, heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads)
            for _ in range(depth)
        ])
    def forward(self, feats):
        # feats: (B, N_modal*H*W, dim)
        for l in self.layers:
            feats = l(feats)
        return feats


class HyperKiteHead(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, hidden, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden, dim, 3, padding=1)  

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

