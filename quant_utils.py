"""
量化工具模块 (Quantization Utilities)
====================================

核心功能:
  - quantize_tensor(): 模拟量化函数（量化→反量化）
  - MixedPrecisionLinear: 混合精度线性层（W4 + A4/A8）

量化策略:
  - 权重: 固定 W4 (4-bit 对称量化, group_size=128)
  - 激活: A4/A8 (4/8-bit 非对称量化, 按敏感度选择)

⚠️ 注意: 这是【模拟量化】，用于评估精度损失，不会加速推理。
   真实加速请使用 GGUF 格式 + llama.cpp。
"""

import torch
import torch.nn as nn


def quantize_tensor(
    x: torch.Tensor, 
    n_bits: int, 
    group_size: int = 128, 
    sym: bool = True
) -> torch.Tensor:
    """
    模拟量化函数 (Fake Quantization)
    
    将张量量化到 n_bits 位后反量化回浮点数，模拟精度损失。
    
    Args:
        x: 输入张量
        n_bits: 量化位数 (2/4/8)
        group_size: 分组大小，-1或0表示不分组
        sym: True=对称量化(权重), False=非对称量化(激活)
    
    Returns:
        模拟量化后的张量（与输入同shape同dtype）
    
    量化公式:
        对称: q = round(x / scale), scale = max|x| / (2^(n-1) - 1)
        非对称: q = round(x/scale + zp), scale = (max-min) / (2^n - 1)
    """
    # 分组量化
    if group_size > 0:
        original_shape = x.shape
        x_flat = x.reshape(-1)
        
        # Padding 使其能被 group_size 整除
        remainder = x_flat.shape[0] % group_size
        if remainder != 0:
            x_flat = torch.nn.functional.pad(x_flat, (0, group_size - remainder))
        
        x_groups = x_flat.reshape(-1, group_size)
        
        if sym:  # 对称量化（权重）
            xmax = x_groups.abs().amax(dim=1, keepdim=True).clamp(min=1e-5)
            q_max = 2**(n_bits - 1) - 1
            scale = xmax / q_max
            x_q = torch.clamp(torch.round(x_groups / scale), -q_max, q_max)
            x_deq = x_q * scale
        else:  # 非对称量化（激活）
            xmin = x_groups.amin(dim=1, keepdim=True)
            xmax = x_groups.amax(dim=1, keepdim=True)
            scale = ((xmax - xmin) / (2**n_bits - 1)).clamp(min=1e-5)
            zero_point = torch.round(-xmin / scale)
            x_q = torch.clamp(torch.round(x_groups / scale + zero_point), 0, 2**n_bits - 1)
            x_deq = (x_q - zero_point) * scale
        
        return x_deq.flatten()[:original_shape.numel()].reshape(original_shape)
    
    else:  # Per-tensor 量化
        if sym:
            xmax = x.abs().max().clamp(min=1e-5)
            q_max = 2**(n_bits - 1) - 1
            scale = xmax / q_max
            x_q = torch.clamp(torch.round(x / scale), -q_max, q_max)
            return x_q * scale
        else:
            xmin, xmax = x.min(), x.max()
            scale = ((xmax - xmin) / (2**n_bits - 1)).clamp(min=1e-5)
            zero_point = torch.round(-xmin / scale)
            x_q = torch.clamp(torch.round(x / scale + zero_point), 0, 2**n_bits - 1)
            return (x_q - zero_point) * scale


class MixedPrecisionLinear(nn.Module):
    """
    混合精度线性层 (W4 + A4/A8)
    
    将 nn.Linear 替换为支持混合精度量化的版本:
      - 权重: 固定 W4 (4-bit 对称量化)
      - 激活: A4 或 A8 (根据层敏感度)
    
    技术:
      - SmoothQuant: 通过缩放平衡激活值和权重的量化难度
      - 权重裁剪: 减少 outlier 影响
    
    Args:
        original_linear: 原始 nn.Linear 层
        w_bits: 权重位数（固定为4）
        a_bits: 激活位数（4或8）
        clip_ratio: 权重裁剪比例 (0.0~1.0)
        smooth_alpha: SmoothQuant α参数 (0.0~1.0)
    
    Example:
        >>> layer = MixedPrecisionLinear(original, w_bits=4, a_bits=8)
        >>> output = layer(input_tensor)
    """
    
    def __init__(
        self, 
        original_linear: nn.Linear, 
        w_bits: int = 4, 
        a_bits: int = 8,
        clip_ratio: float = 0.9, 
        smooth_alpha: float = 0.5
    ):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.w_bits = w_bits
        self.a_bits = a_bits
        self.clip_ratio = clip_ratio
        self.smooth_alpha = smooth_alpha
        
        # 复制权重和偏置
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：应用 W4 + A4/A8 混合精度量化"""
        # === SmoothQuant 激活平滑 ===
        act_max = x.abs().amax(dim=0, keepdim=True).amax(dim=1, keepdim=True).clamp(min=1e-5)
        weight_max = self.weight.abs().amax(dim=0).clamp(min=1e-5)
        
        # 缩放因子: s = (act_max^α) / (weight_max^(1-α))
        scales = (act_max.pow(self.smooth_alpha) / weight_max.pow(1 - self.smooth_alpha)).clamp(min=1e-5)
        
        x_smooth = x / scales
        w_smooth = self.weight * scales.squeeze()
        
        # === 权重量化 (固定 W4) ===
        limit = w_smooth.abs().amax() * self.clip_ratio
        w_clipped = torch.clamp(w_smooth, -limit, limit)
        w_q = quantize_tensor(w_clipped, n_bits=4, group_size=128, sym=True)
        
        # === 激活量化 (A4 或 A8) ===
        x_q = quantize_tensor(x_smooth, n_bits=self.a_bits, group_size=-1, sym=False)
        
        return torch.nn.functional.linear(x_q, w_q, self.bias)
    
    def extra_repr(self) -> str:
        return (f'in={self.in_features}, out={self.out_features}, '
                f'W{self.w_bits}A{self.a_bits}, clip={self.clip_ratio}')
