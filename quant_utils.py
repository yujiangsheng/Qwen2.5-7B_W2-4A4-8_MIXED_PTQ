"""
量化工具模块 (Quantization Utilities)
====================================

本模块提供混合精度量化的核心函数和类。

核心组件:
---------
- quantize_tensor(): 张量模拟量化函数
- MixedPrecisionLinear: 混合精度线性层

重要说明:
---------
本模块实现的是【模拟量化】(Fake Quantization)，用于：
- 评估量化对模型精度的影响
- 搜索最优的逐层位宽配置

如需真正的推理加速，请使用 llama.cpp 等真实量化框架。

使用示例:
---------
>>> from quant_utils import quantize_tensor, MixedPrecisionLinear
>>> 
>>> # 模拟 2-bit 权重量化
>>> weight = torch.randn(1024, 1024)
>>> weight_q = quantize_tensor(weight, n_bits=2, group_size=128)
>>> 
>>> # 创建混合精度线性层
>>> original = nn.Linear(512, 1024)
>>> quant_layer = MixedPrecisionLinear(original, w_bits=2, a_bits=8)
"""

import torch
import torch.nn as nn


# ============================================================================
# 核心量化函数
# ============================================================================

def quantize_tensor(
    x: torch.Tensor, 
    n_bits: int, 
    group_size: int = 128, 
    sym: bool = True
) -> torch.Tensor:
    """
    模拟量化函数 (Fake Quantization)
    
    将输入张量量化到 n_bits 位后再反量化回浮点数，
    模拟真实量化带来的精度损失。
    
    量化公式:
    ---------
    对称量化: x_q = clamp(round(x / scale), -qmax, qmax) * scale
    非对称量化: x_q = (clamp(round(x / scale + zp), 0, qmax) - zp) * scale
    
    参数:
    -----
    x : torch.Tensor
        待量化张量
    n_bits : int
        量化位数 (2, 4, 8 等)
    group_size : int
        分组大小，每组独立计算 scale（默认 128）
        设为 -1 或 0 表示 per-tensor 量化
    sym : bool
        是否对称量化（默认 True）
        - True: 适用于权重（分布通常对称）
        - False: 适用于激活值（分布可能不对称）
    
    返回:
    ------
    torch.Tensor
        模拟量化后的张量（dtype 不变）
    
    示例:
    ------
    >>> weight = torch.randn(1024, 1024)
    >>> weight_q = quantize_tensor(weight, n_bits=2, group_size=128, sym=True)
    """
    
    # 分组量化模式
    if group_size > 0:
        original_shape = x.shape
        x_flat = x.reshape(-1)
        
        # 如果张量大小不能被group_size整除，需要padding
        remainder = x_flat.shape[0] % group_size
        if remainder != 0:
            pad_len = group_size - remainder
            x_flat = torch.nn.functional.pad(x_flat, (0, pad_len))
        
        # 重塑为 (num_groups, group_size) 以便分组计算
        x_groups = x_flat.reshape(-1, group_size)
        
        if sym:
            # === 对称量化 ===
            # 计算每组的最大绝对值作为scale基准
            xmax = x_groups.abs().amax(dim=1, keepdim=True)
            xmax = torch.clamp(xmax, min=1e-5)  # 防止除零
            
            # 对称量化范围: [-qmax, qmax]
            q_max = 2**(n_bits - 1) - 1
            scale = xmax / q_max
            
            # 量化：round(x / scale)，然后clamp到有效范围
            x_q = torch.clamp(torch.round(x_groups / scale), -q_max, q_max)
            # 反量化：x_q * scale
            x_deq = x_q * scale
        else:
            # === 非对称量化 ===
            # 分别计算每组的最小值和最大值
            xmin = x_groups.amin(dim=1, keepdim=True)
            xmax = x_groups.amax(dim=1, keepdim=True)
            
            # 计算scale和zero_point
            scale = (xmax - xmin) / (2**n_bits - 1)
            scale = torch.clamp(scale, min=1e-5)  # 防止除零
            zero_point = torch.round(-xmin / scale)
            
            # 量化：round(x / scale + zero_point)，然后clamp到 [0, 2^n-1]
            x_q = torch.clamp(torch.round(x_groups / scale + zero_point), 0, 2**n_bits - 1)
            # 反量化：(x_q - zero_point) * scale
            x_deq = (x_q - zero_point) * scale
        
        # 去除padding并恢复原始形状
        x_deq = x_deq.flatten()[:original_shape.numel()].reshape(original_shape)
        return x_deq
    
    else:
        # === Per-tensor量化模式（不分组）===
        if sym:
            # 对称量化
            xmax = x.abs().max()
            xmax = torch.clamp(xmax, min=1e-5)
            q_max = 2**(n_bits - 1) - 1
            scale = xmax / q_max
            x_q = torch.clamp(torch.round(x / scale), -q_max, q_max)
            return x_q * scale
        else:
            # 非对称量化
            xmin = x.min()
            xmax = x.max()
            scale = (xmax - xmin) / (2**n_bits - 1)
            scale = torch.clamp(scale, min=1e-5)
            zero_point = torch.round(-xmin / scale)
            x_q = torch.clamp(torch.round(x / scale + zero_point), 0, 2**n_bits - 1)
            return (x_q - zero_point) * scale


class MixedPrecisionLinear(nn.Module):
    """
    混合精度线性层 (W2/W4 + A4/A8)
    
    替换原始 nn.Linear，支持可配置的权重/激活值量化。
    实现 SmoothQuant 风格的激活平滑技术。
    
    量化策略:
    ---------
    - W2 + A8: 低敏感层，低精度权重 + 高精度激活补偿
    - W4 + A4: 高敏感层，高精度权重 + 低精度激活
    
    技术原理:
    ---------
    1. SmoothQuant 平滑: 将量化难度从激活值转移到权重
       X' = X / s,  W' = W * s
       其中 s = (max|X|^α) / (max|W|^(1-α))
    
    2. 权重裁剪: 减少 outlier 影响
       W_clipped = clamp(W, -limit, limit)
       limit = max|W| * clip_ratio
    
    参数:
    -----
    original_linear : nn.Linear
        原始 PyTorch 线性层
    w_bits : int
        权重位数 (2 或 4)
    a_bits : int
        激活值位数 (4 或 8)
    clip_ratio : float
        权重裁剪比例 (0.0~1.0)
    smooth_alpha : float
        SmoothQuant α 参数 (0.0~1.0)
    
    示例:
    ------
    >>> original = nn.Linear(512, 1024)
    >>> quant_layer = MixedPrecisionLinear(
    ...     original, w_bits=2, a_bits=8, 
    ...     clip_ratio=0.9, smooth_alpha=0.5
    ... )
    >>> output = quant_layer(input_tensor)
    """
    
    def __init__(self, original_linear: nn.Linear, w_bits: int, a_bits: int, 
                 clip_ratio: float = 0.9, smooth_alpha: float = 0.5):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # 量化参数
        self.w_bits = w_bits          # 权重位数
        self.a_bits = a_bits          # 激活值位数
        self.clip_ratio = clip_ratio  # 裁剪比例
        self.smooth_alpha = smooth_alpha  # 平滑参数
        
        # 复制原始权重和偏置
        self.weight = nn.Parameter(original_linear.weight.data.clone())
        if original_linear.bias is not None:
            self.bias = nn.Parameter(original_linear.bias.data.clone())
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，应用混合精度量化
        
        W2: 低敏感度层，激进压缩
        W4: 高敏感度层，保持精度
        """
        # === SmoothQuant风格的激活值平滑 ===
        # 计算激活值的最大绝对值（per-token-per-channel）
        act_max = x.abs().amax(dim=0, keepdim=True).amax(dim=1, keepdim=True)
        # 计算权重的最大绝对值（per-input-channel）
        weight_max = self.weight.abs().amax(dim=0)
        
        # 防止除零
        act_max = torch.clamp(act_max, min=1e-5)
        weight_max = torch.clamp(weight_max, min=1e-5)
        
        # 计算缩放因子: s = (act_max^α) / (weight_max^(1-α))
        alpha = self.smooth_alpha
        scales = (act_max.pow(alpha) / weight_max.pow(1 - alpha)).clamp(min=1e-5)
        
        # 应用平滑缩放
        x_smoothed = x / scales
        w_smoothed = self.weight * scales.squeeze()
        
        # === 权重量化 ===
        # 裁剪权重以减少outlier影响
        limit = w_smoothed.abs().amax() * self.clip_ratio
        w_clipped = torch.clamp(w_smoothed, -limit, limit)
        # 分组对称量化
        w_q = quantize_tensor(w_clipped, n_bits=self.w_bits, group_size=128, sym=True)
        
        # === 激活值量化 ===
        # Per-tensor非对称量化（激活值分布通常不对称）
        x_q = quantize_tensor(x_smoothed, n_bits=self.a_bits, group_size=-1, sym=False)
        
        # 线性变换
        return torch.nn.functional.linear(x_q, w_q, self.bias)
    
    def extra_repr(self) -> str:
        """返回层的字符串表示"""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'w_bits={self.w_bits}, a_bits={self.a_bits}, '
                f'clip_ratio={self.clip_ratio}, smooth_alpha={self.smooth_alpha}')
