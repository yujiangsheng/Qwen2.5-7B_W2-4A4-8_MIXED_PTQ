"""
量化工具模块 (Quantization Utilities)
====================================

⚠️ 重要说明：模拟量化 vs 真实量化
---------------------------------

本模块实现的是【模拟量化】（Fake Quantization），而非【真实量化】。

┌─────────────────────────────────────────────────────────────────────┐
│  模拟量化 (Simulated Quantization) - 本模块实现                     │
├─────────────────────────────────────────────────────────────────────┤
│  FP32 权重 → 量化(round) → 反量化 → FP32 权重                       │
│                                                                     │
│  数据类型始终是 FP32，只是模拟了量化带来的精度损失                   │
│  ❌ 不会加速推理（反而因为额外操作而变慢）                           │
│  ✅ 用于评估量化对模型精度的影响                                     │
│  ✅ 用于搜索最优的逐层位宽配置                                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  真实量化 (Real Quantization) - 需要使用专门的推理框架               │
├─────────────────────────────────────────────────────────────────────┤
│  FP32 权重 → 转换为 INT4/INT8 → 存储为 GGUF/GPTQ 格式               │
│                                                                     │
│  计算直接使用低精度整数运算（硬件加速）                              │
│  ✅ 真正加速推理（5-10 倍）                                          │
│  ✅ 大幅减少内存占用（70-85%）                                       │
│  推荐框架: llama.cpp, bitsandbytes, TensorRT-LLM                    │
└─────────────────────────────────────────────────────────────────────┘

本模块功能：
-----------
- 模拟量化函数：将张量量化后再反量化，模拟精度损失
- 分组量化：按 group_size 分组，每组独立计算 scale
- 对称/非对称量化模式

核心概念：
---------
1. 对称量化 (Symmetric): 量化范围 [-qmax, qmax]，无零点偏移
   - 适用于权重量化，分布通常对称
   - scale = max(|x|) / (2^(n-1) - 1)

2. 非对称量化 (Asymmetric): 量化范围 [0, 2^n - 1]，有零点偏移
   - 适用于激活值量化，分布可能不对称
   - scale = (max - min) / (2^n - 1)
   - zero_point = round(-min / scale)

3. 分组量化 (Group Quantization):
   - 将权重按 group_size 分组，每组独立计算 scale
   - 提高量化精度，尤其对低 bit 量化（W2/W4）效果显著
"""

import torch
import torch.nn as nn


def quantize_tensor(x: torch.Tensor, n_bits: int, group_size: int = 128, sym: bool = True) -> torch.Tensor:
    """
    模拟量化函数（Fake Quantization）
    
    将输入张量量化到n_bits位后再反量化回浮点数，
    模拟真实量化带来的精度损失。
    
    参数：
    -----
    x : torch.Tensor
        待量化的输入张量
    n_bits : int
        量化位数 (2, 4, 8等)
    group_size : int, optional
        分组大小，-1或0表示不分组（per-tensor量化）
        默认128，适合大多数场景
    sym : bool, optional
        是否使用对称量化，默认True
        - True: 对称量化，适用于权重
        - False: 非对称量化，适用于激活值
    
    返回：
    ------
    torch.Tensor
        模拟量化后的张量（与输入同shape同dtype）
    
    示例：
    ------
    >>> weight = torch.randn(1024, 1024)
    >>> weight_q = quantize_tensor(weight, n_bits=2, group_size=128, sym=True)
    >>> # weight_q 形状与 weight 相同，但精度损失模拟了2-bit量化效果
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
    混合精度线性层
    
    替换原始nn.Linear，支持可配置的权重/激活值量化位数。
    实现SmoothQuant风格的激活平滑技术，将量化难度从激活值转移到权重。
    
    核心技术：
    ----------
    1. SmoothQuant平滑：通过可学习的缩放因子，平衡激活值和权重的量化误差
       x' = x / s,  W' = W * s
       其中 s = (max|x|^α) / (max|W|^(1-α))
    
    2. 权重裁剪（Clipping）：将权重裁剪到 [-limit, limit] 范围
       limit = max|W| * clip_ratio
       较小的clip_ratio可减少outlier影响，但可能损失信息
    
    参数：
    -----
    original_linear : nn.Linear
        原始的PyTorch线性层
    w_bits : int
        权重量化位数 (2, 4, 8)
    a_bits : int
        激活值量化位数（通常为8）
    clip_ratio : float
        权重裁剪比例 (0.0~1.0)
    smooth_alpha : float
        SmoothQuant的alpha参数 (0.0~1.0)
        - alpha=0: 所有平滑转移到激活值
        - alpha=1: 所有平滑转移到权重
        - alpha=0.5: 平衡
    
    示例：
    ------
    >>> original = nn.Linear(512, 1024)
    >>> quant_layer = MixedPrecisionLinear(original, w_bits=2, a_bits=8, 
    ...                                      clip_ratio=0.9, smooth_alpha=0.5)
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
