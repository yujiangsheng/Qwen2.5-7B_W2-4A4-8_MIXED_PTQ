"""
混合精度 PTQ 主程序 (Mixed-Precision Post-Training Quantization)
================================================================

工作流程:
  1. 加载预训练模型
  2. 敏感度分析: 评估每层对 A4/A8 的敏感程度
  3. 遗传算法优化: 搜索最优激活位宽配置
  4. 保存配置文件

量化策略 (W4 + A4/A8):
  - 权重: 固定 W4 (4-bit)
  - 激活: A4/A8 混合（按敏感度选择）

⚠️ 这是模拟量化，用于搜索最优配置。真实加速请使用 GGUF + llama.cpp。

用法:
  python mixed_precision_ptq.py
  python mixed_precision_ptq.py --device mps --ga_gen 15
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from tqdm import tqdm
import numpy as np

from data_utils import create_mock_input
from quant_utils import quantize_tensor
from genetic_optim import MixedPrecisionGA


def get_device() -> str:
    """自动检测最佳设备: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id: str, device: str):
    """加载预训练模型和分词器"""
    print(f"📦 加载模型: {model_id}")
    print(f"📍 设备: {device}")
    
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        if device == "mps":
            model = model.to("mps")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer


def get_linear_layers(model) -> list:
    """获取所有需要量化的线性层（跳过 embedding 和 lm_head）"""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "layers" in name:
            layers.append((name, module))
    return layers


def evaluate_layer_sensitivity(layer, calib_input, device) -> dict:
    """
    评估单层敏感度（权重固定 W4，测试 A4/A8）
    
    Returns:
        {4: mse_a4, 8: mse_a8}
    """
    with torch.no_grad():
        original_output = layer(calib_input)
    
    sensitivity = {}
    w = layer.weight
    
    # W4 权重量化
    limit = w.abs().amax() * 0.9
    w_clipped = torch.clamp(w, -limit, limit)
    w_q = quantize_tensor(w_clipped, n_bits=4, group_size=128, sym=True)
    
    # 测试 A4/A8
    for a_bits in [4, 8]:
        x_q = quantize_tensor(calib_input, n_bits=a_bits, group_size=-1, sym=False)
        with torch.no_grad():
            out_q = torch.nn.functional.linear(x_q, w_q, layer.bias)
        mse = torch.mean((out_q - original_output) ** 2).item()
        sensitivity[a_bits] = mse
    
    return sensitivity


def create_fitness_function(layers_to_quantize: list, sensitivities: dict):
    """创建遗传算法适应度函数"""
    # 计算层权重（基于敏感度比例）
    layer_weights = []
    n_layers = len(layers_to_quantize)
    
    for i, (name, _) in enumerate(layers_to_quantize):
        sens = sensitivities.get(name, {4: 0.1, 8: 0.01})
        ratio = sens.get(4, 0.1) / max(sens.get(8, 0.01), 1e-8)
        
        # 首尾层加权
        if i < 7 or i >= n_layers - 7:
            ratio *= 1.5
        
        layer_weights.append(np.log1p(ratio))
    
    # 归一化到 [0.5, 2.0]
    weights = np.array(layer_weights)
    if weights.max() > weights.min():
        weights = 0.5 + 1.5 * (weights - weights.min()) / (weights.max() - weights.min())
    else:
        weights = np.ones_like(weights)
    
    def fitness_function(bit_config):
        total_mse = 0
        for i, (name, _) in enumerate(layers_to_quantize):
            a_bits = int(bit_config[i])
            mse = sensitivities[name].get(a_bits, sensitivities[name][8])
            total_mse += mse * weights[i]
        return -total_mse  # 负 MSE 作为适应度
    
    return fitness_function


def save_config(layers_to_quantize: list, best_config: np.ndarray, output_path: str):
    """保存混合精度配置"""
    mixed_config = {}
    a4_layers, a8_layers = [], []
    
    for i, (name, _) in enumerate(layers_to_quantize):
        a_bits = int(best_config[i])
        mixed_config[name] = {
            'w_bits': 4,
            'a_bits': a_bits,
            'clip_ratio': 0.9,
            'smooth_alpha': 0.5
        }
        (a4_layers if a_bits == 4 else a8_layers).append(name)
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("📊 混合精度配置摘要 (W4 + A4/A8)")
    print('='*60)
    print(f"  权重: 所有层 W4 (4-bit)")
    print(f"  A4层 (低敏感度): {len(a4_layers)} 个")
    print(f"  A8层 (高敏感度): {len(a8_layers)} 个")
    
    avg_a_bits = np.mean(best_config)
    compression = (4 + avg_a_bits) / (4 + 8)
    print(f"  平均激活位宽: {avg_a_bits:.2f} bit")
    print(f"  压缩比: {compression:.1%} (相对于W4A8)")
    print('='*60)
    
    torch.save(mixed_config, output_path)
    print(f"\n✅ 配置已保存: {output_path}")
    
    return mixed_config


def main():
    parser = argparse.ArgumentParser(
        description="混合精度PTQ (W4 + A4/A8) - 基于遗传算法优化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python mixed_precision_ptq.py
  python mixed_precision_ptq.py --device mps --ga_gen 15 --target_compression 0.75
        """
    )
    
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--device', type=str, default=get_device())
    parser.add_argument('--n_layers', type=int, default=196, help="量化层数")
    parser.add_argument('--ga_pop', type=int, default=30, help="GA种群大小")
    parser.add_argument('--ga_gen', type=int, default=25, help="GA迭代代数")
    parser.add_argument('--target_compression', type=float, default=0.75, help="目标压缩比")
    parser.add_argument('--output', type=str, default="mixed_precision_config.pt")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🚀 混合精度PTQ (W4 + A4/A8)")
    print("="*60)
    print(f"  模型: {args.model_id}")
    print(f"  设备: {args.device}")
    print(f"  目标压缩比: {args.target_compression:.0%}")
    print("="*60 + "\n")
    
    # 1. 加载模型
    model, tokenizer = load_model(args.model_id, args.device)
    
    all_layers = get_linear_layers(model)
    layers_to_quantize = all_layers[:args.n_layers]
    n_layers = len(layers_to_quantize)
    print(f"\n📊 待量化层数: {n_layers}")
    
    # 2. 敏感度分析
    print("\n" + "="*60)
    print("📈 Step 1: 敏感度分析 (A4 vs A8)")
    print("="*60)
    
    sensitivities = {}
    for name, layer in tqdm(layers_to_quantize, desc="分析敏感度"):
        mock_input = create_mock_input(
            layer, batch_size=1, seq_len=128,
            device=layer.weight.device, dtype=layer.weight.dtype
        )
        
        sens = evaluate_layer_sensitivity(layer, mock_input, args.device)
        sensitivities[name] = sens
        
        # 显示部分结果
        idx = len(sensitivities) - 1
        if idx < 5 or idx % 30 == 0:
            ratio = sens[4] / max(sens[8], 1e-8)
            cat = "A8" if ratio > 2.5 else "A4"
            print(f"  {name}: A4={sens[4]:.4f}, A8={sens[8]:.4f} -> {cat}")
    
    # 3. 遗传算法优化
    print("\n" + "="*60)
    print("🧬 Step 2: 遗传算法优化")
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
    
    layer_names = [name for name, _ in layers_to_quantize]
    ga.set_layer_sensitivities(sensitivities, layer_names)
    
    best_config = ga.optimize(fitness_func, target_compression=args.target_compression)
    
    # 4. 保存配置
    print("\n" + "="*60)
    print("💾 Step 3: 保存配置")
    print("="*60)
    
    save_config(layers_to_quantize, best_config, args.output)
    
    print("\n✅ 混合精度PTQ完成!")
    print("  下一步: python export_gguf_official.py")


if __name__ == "__main__":
    main()
