"""
混合精度 PTQ 主程序 (Mixed-Precision Post-Training Quantization)
================================================================

本程序使用遗传算法搜索最优的逐层量化位宽配置。

量化策略 (W2/W4 + A4/A8):
-----------------------
- 低敏感层: W2 + A8 (低精度权重 + 高精度激活补偿)
- 高敏感层: W4 + A4 (高精度权重 + 低精度激活)

工作流程:
---------
1. 加载预训练模型
2. 敏感度分析: 评估每层对 W2/W4 × A4/A8 四种组合的敏感程度
3. 遗传算法优化: 搜索最优位宽配置
4. 保存配置到文件

重要说明:
---------
本程序使用【模拟量化】评估精度，不会加速推理。
如需真正的加速，请使用 compare_real_quant.py。

使用方法:
---------
# 基本用法（自动检测设备）
python mixed_precision_ptq.py

# 指定参数
python mixed_precision_ptq.py --device mps --ga_gen 20 --target_compression 0.25

输出:
-----
mixed_precision_config.pt: 每层量化配置字典，包含:
  - w_bits: 权重位宽 (2 或 4)
  - a_bits: 激活位宽 (4 或 8)
  - clip_ratio: 权重裁剪比例
  - smooth_alpha: SmoothQuant 参数

下一步:
-------
1. 导出 GGUF: python export_gguf_official.py
2. 性能对比: python compare_real_quant.py
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm
import numpy as np

from data_utils import get_calib_dataset, create_mock_input
from quant_utils import quantize_tensor
from genetic_optim import MixedPrecisionGA, LayerSensitivityAnalyzer


# ============================================================================
# 工具函数
# ============================================================================

def get_best_device() -> str:
    """
    自动检测最佳可用设备
    
    优先级: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str, device: str):
    """
    加载预训练模型
    
    参数:
    -----
    model_id : str
        HuggingFace 模型 ID 或本地路径
    device : str
        目标设备 ('cuda', 'mps', 'cpu')
    
    返回:
    ------
    tuple
        (model, tokenizer)
    """
    print(f"正在加载模型: {model_id}")
    print(f"目标设备: {device}")
    
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    elif device == "mps":
        # MPS目前对float16支持有限，使用float32
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float32
        )
        model = model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float32
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    return model, tokenizer


def get_linear_layers(model) -> list:
    """
    获取模型中所有需要量化的线性层
    
    只选择decoder layers中的线性层，跳过embedding和lm_head
    
    参数：
    -----
    model : AutoModelForCausalLM
        目标模型
    
    返回：
    ------
    list
        [(层名称, 层模块), ...] 列表
    """
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "layers" in name:
            layers.append((name, module))
    return layers


def evaluate_layer_sensitivity(layer, calib_input, device) -> dict:
    """
    评估单层对不同量化位宽的敏感度（权重W2/W4 + 激活A4/A8）
    
    参数：
    -----
    layer : nn.Linear
        待评估的线性层
    calib_input : torch.Tensor
        校准输入，shape: [batch, seq_len, in_features]
    device : str
        计算设备
    
    返回：
    ------
    dict
        位宽组合到MSE的映射: {(w_bits, a_bits): mse, ...}
        例如: {(2, 4): mse_w2a4, (2, 8): mse_w2a8, (4, 4): mse_w4a4, (4, 8): mse_w4a8}
    """
    # 获取原始FP输出
    with torch.no_grad():
        original_output = layer(calib_input)
    
    sensitivity = {}
    # 测试所有权重和激活位宽组合: W2/W4 x A4/A8
    for w_bits in [2, 4]:
        for a_bits in [4, 8]:
            w = layer.weight
            
            # 裁剪权重（减少outlier影响）
            limit = w.abs().amax() * 0.9
            w_clipped = torch.clamp(w, -limit, limit)
            
            # 量化权重
            w_q = quantize_tensor(w_clipped, n_bits=w_bits, group_size=128, sym=True)
            
            # 量化激活（A4或A8）
            x_q = quantize_tensor(calib_input, n_bits=a_bits, group_size=-1, sym=False)
            
            # 计算量化后输出
            with torch.no_grad():
                out_q = torch.nn.functional.linear(x_q, w_q, layer.bias)
            
            # 计算MSE
            mse = torch.mean((out_q - original_output) ** 2).item()
            sensitivity[(w_bits, a_bits)] = mse
    
    # 为了兼容性，也保留原来的格式（取A8的值）
    sensitivity[2] = sensitivity.get((2, 8), sensitivity.get((2, 4), 0.1))
    sensitivity[4] = sensitivity.get((4, 8), sensitivity.get((4, 4), 0.01))
    
    return sensitivity


def create_fitness_function(layers_to_quantize: list, sensitivities: dict):
    """
    创建遗传算法的适应度函数（增强版）
    
    适应度 = -加权总MSE（MSE越小，适应度越高）
    权重基于敏感度比例：比例高的层（更敏感）给予更高权重
    
    参数：
    -----
    layers_to_quantize : list
        待量化层列表
    sensitivities : dict
        预计算的敏感度字典
    
    返回：
    ------
    Callable
        适应度函数
    """
    # 预计算层权重（基于敏感度比例）
    layer_weights = []
    n_layers = len(layers_to_quantize)
    
    for i, (name, _) in enumerate(layers_to_quantize):
        sens = sensitivities.get(name, {2: 0.1, 4: 0.01})
        w2_mse = sens.get(2, 0.1)
        w4_mse = sens.get(4, 0.01)
        
        # 敏感度比例
        ratio = w2_mse / max(w4_mse, 1e-8)
        
        # 首尾层特殊处理（提高权重）
        if i < 7 or i >= n_layers - 7:  # 第一层和最后一层的所有子层
            ratio *= 1.5
        
        layer_weights.append(np.log1p(ratio))  # 对数变换平滑权重
    
    # 归一化权重到 [0.5, 2.0]
    weights = np.array(layer_weights)
    if weights.max() > weights.min():
        weights = 0.5 + 1.5 * (weights - weights.min()) / (weights.max() - weights.min())
    else:
        weights = np.ones_like(weights)
    
    def fitness_function(bit_config):
        total_weighted_mse = 0
        for i, (name, _) in enumerate(layers_to_quantize):
            bits = int(bit_config[i])
            mse = sensitivities[name].get(bits, sensitivities[name][4])
            # 加权MSE
            total_weighted_mse += mse * weights[i]
        return -total_weighted_mse  # 负MSE作为适应度
    
    return fitness_function


def save_config(layers_to_quantize: list, best_config: np.ndarray, 
                output_path: str) -> dict:
    """
    保存混合精度配置
    
    参数：
    -----
    layers_to_quantize : list
        层列表
    best_config : np.ndarray
        最优位宽配置
    output_path : str
        输出文件路径
    
    返回：
    ------
    dict
        配置字典
    """
    mixed_config = {}
    w2_layers, w4_layers = [], []
    
    # 统计激活位宽
    a4_layers, a8_layers = [], []
    
    for i, (name, _) in enumerate(layers_to_quantize):
        w_bits = int(best_config[i])
        
        # 根据权重位宽决定激活位宽：
        # - W2层使用A8（低精度权重需要高精度激活来补偿）
        # - W4层可以使用A4（高精度权重可以容忍低精度激活）
        a_bits = 8 if w_bits == 2 else 4
        
        mixed_config[name] = {
            'w_bits': w_bits,
            'a_bits': a_bits,
            'clip_ratio': 0.7 if w_bits == 2 else 0.9,  # W2使用更激进的裁剪
            'smooth_alpha': 0.5
        }
        
        if w_bits == 2:
            w2_layers.append(name)
        else:  # w_bits == 4
            w4_layers.append(name)
        
        if a_bits == 4:
            a4_layers.append(name)
        else:
            a8_layers.append(name)
    
    # 打印配置摘要
    print(f"\n{'='*60}")
    print("混合精度配置摘要")
    print('='*60)
    
    print(f"\nW2层 - 低敏感度 ({len(w2_layers)}个):")
    for name in w2_layers[:5]:
        print(f"  - {name}")
    if len(w2_layers) > 5:
        print(f"  ... 及其他 {len(w2_layers) - 5} 层")
    
    print(f"\nW4层 - 高敏感度 ({len(w4_layers)}个):")
    for name in w4_layers[:5]:
        print(f"  - {name}")
    if len(w4_layers) > 5:
        print(f"  ... 及其他 {len(w4_layers) - 5} 层")
    
    print(f"\nA4层 - 低精度激活 ({len(a4_layers)}个):")
    print(f"  (与W4层对应，使用4-bit激活)")
    
    print(f"\nA8层 - 高精度激活 ({len(a8_layers)}个):")
    print(f"  (与W2层对应，使用8-bit激活补偿低精度权重)")
    
    # 计算压缩统计
    n_layers = len(layers_to_quantize)
    total_bits_orig = n_layers * 16  # 假设原始FP16
    total_bits_quant = sum(best_config)
    compression = total_bits_quant / total_bits_orig
    
    print(f"\n{'='*60}")
    print("压缩统计")
    print('='*60)
    print(f"  总层数: {n_layers}")
    print(f"  W2 (低敏感度): {len(w2_layers)} 层")
    print(f"  W4 (高敏感度): {len(w4_layers)} 层")
    print(f"  压缩比: {compression:.1%} (原始大小的 {compression*100:.1f}%)")
    print(f"  内存节省: {(1-compression)*100:.1f}%")
    print('='*60)
    
    # 保存配置
    torch.save(mixed_config, output_path)
    print(f"\n✓ 配置已保存至: {output_path}")
    
    return mixed_config


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(
        description="混合精度PTQ量化 - 基于遗传算法优化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法 (自动检测设备)
  python mixed_precision_ptq.py
  
  # 指定模型和设备
  python mixed_precision_ptq.py --model_id Qwen/Qwen2.5-7B-Instruct --device cuda
  
  # 完整参数
  python mixed_precision_ptq.py \\
      --model_id Qwen/Qwen2.5-7B-Instruct \\
      --device mps \\
      --n_layers 196 \\
      --ga_pop 20 \\
      --ga_gen 15 \\
      --target_compression 0.25 \\
      --output my_config.pt
        """
    )
    
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace模型ID或本地路径")
    parser.add_argument('--device', type=str, default=get_best_device(),
                        help="计算设备: cuda, mps, cpu")
    parser.add_argument('--n_samples', type=int, default=64,
                        help="校准样本数量")
    parser.add_argument('--n_layers', type=int, default=196,
                        help="要量化的层数 (Qwen2.5-7B共196层)")
    parser.add_argument('--ga_pop', type=int, default=30,
                        help="遗传算法种群大小（增大可提高搜索范围）")
    parser.add_argument('--ga_gen', type=int, default=25,
                        help="遗传算法迭代代数（增加可获得更好收敛）")
    parser.add_argument('--target_compression', type=float, default=0.25,
                        help="目标压缩比 (0.25表示原大小的25%%)")
    parser.add_argument('--output', type=str, default="mixed_precision_config.pt",
                        help="输出配置文件路径")
    
    args = parser.parse_args()
    
    # 打印配置
    print("="*60)
    print("混合精度PTQ量化")
    print("="*60)
    print(f"模型: {args.model_id}")
    print(f"设备: {args.device}")
    print(f"目标层数: {args.n_layers}")
    print(f"目标压缩比: {args.target_compression:.0%}")
    print("="*60 + "\n")
    
    # Step 1: 加载模型
    model, tokenizer = load_model(args.model_id, args.device)
    
    # 获取所有线性层
    all_layers = get_linear_layers(model)
    layers_to_quantize = all_layers[:args.n_layers]
    n_layers = len(layers_to_quantize)
    print(f"\n将对 {n_layers} 个线性层进行量化分析\n")
    
    # Step 2: 敏感度分析
    print("="*60)
    print("步骤1: 层敏感度分析")
    print("="*60)
    
    sensitivities = {}
    for name, layer in tqdm(layers_to_quantize, desc="分析敏感度"):
        # 创建模拟输入
        mock_input = create_mock_input(
            layer, 
            batch_size=1, 
            seq_len=128,
            device=layer.weight.device,
            dtype=layer.weight.dtype
        )
        
        sens = evaluate_layer_sensitivity(layer, mock_input, args.device)
        sensitivities[name] = sens
        
        # 使用敏感度比例进行分类（W2_MSE / W4_MSE）
        w2_mse = sens[2]
        w4_mse = sens[4]
        ratio = w2_mse / max(w4_mse, 1e-8)
        
        # 首尾层特殊处理
        layer_idx = len(sensitivities) - 1
        is_edge = layer_idx < 7 or layer_idx >= args.n_layers - 7
        threshold = 15 if is_edge else 30  # 边缘层用更严格的阈值
        
        if ratio > threshold:
            category = f"高敏感度(W4) [比例:{ratio:.1f}]"
        else:
            category = f"低敏感度(W2) [比例:{ratio:.1f}]"
        
        # 只显示部分层的详细信息
        if len(sensitivities) <= 10 or len(sensitivities) % 20 == 0:
            print(f"  {name}: W2={w2_mse:.4f}, W4={sens[4]:.4f} -> {category}")
    
    # Step 3: 遗传算法优化
    print("\n" + "="*60)
    print("步骤2: 遗传算法优化")
    print("="*60)
    
    fitness_func = create_fitness_function(layers_to_quantize, sensitivities)
    
    ga = MixedPrecisionGA(
        n_layers=n_layers,
        population_size=args.ga_pop,
        n_generations=args.ga_gen,
        mutation_rate=0.12,
        elite_ratio=0.15,
        adaptive_mutation=True
    )
    
    # 传递敏感度信息给GA，用于智能初始化和加权变异
    layer_names = [name for name, _ in layers_to_quantize]
    ga.set_layer_sensitivities(sensitivities, layer_names)
    
    best_config = ga.optimize(fitness_func, target_compression=args.target_compression)
    
    # Step 4: 保存配置
    print("\n" + "="*60)
    print("步骤3: 生成最终配置")
    print("="*60)
    
    save_config(layers_to_quantize, best_config, args.output)
    
    print("\n✓ 混合精度PTQ完成!")
    print(f"  使用 'python test_mixed_precision.py' 测试推理效果")


if __name__ == "__main__":
    main()
