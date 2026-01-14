"""
模拟量化对比测试 (Simulated Quantization Comparison)
====================================================

⚠️ 注意：这是【模拟量化】测试，用于验证量化精度，不能获得加速效果！
        如需真正的加速，请使用 compare_real_quant.py

本脚本对比【原始模型】与【模拟量化模型】的推理效果。

模拟量化说明：
-------------
- 模拟量化：FP32 → 量化 → 反量化 → FP32（有精度损失）
- 计算仍是 FP32，只是模拟了低精度带来的精度损失
- 用途：评估量化配置对模型精度的影响
- 速度：因额外操作，比原始模型更慢

使用方法：
---------
>>> python compare_models.py
>>> python compare_models.py --prompt "解释什么是机器学习"

推荐：如需测试真实加速效果，请运行：
>>> python compare_real_quant.py
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import time
import copy
from quant_utils import MixedPrecisionLinear


def get_device() -> str:
    """
    自动检测最佳可用设备
    
    优先级: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apply_mixed_precision(model, config: dict) -> tuple:
    """
    将混合精度量化配置应用到模型
    
    遍历配置中的每个层，将原始 nn.Linear 替换为 MixedPrecisionLinear
    
    参数:
        model: HuggingFace 模型
        config: 量化配置字典
    
    返回:
        (模型, 统计信息字典)
    """
    stats = {'W2': 0, 'W4': 0, 'W8': 0}
    
    for name, params in config.items():
        parts = name.split('.')
        parent = model
        
        try:
            for part in parts[:-1]:
                parent = getattr(parent, part)
            layer_name = parts[-1]
            original = getattr(parent, layer_name)
            
            if isinstance(original, nn.Linear):
                new_layer = MixedPrecisionLinear(
                    original,
                    w_bits=params['w_bits'],
                    a_bits=params['a_bits'],
                    clip_ratio=params['clip_ratio'],
                    smooth_alpha=params['smooth_alpha']
                )
                setattr(parent, layer_name, new_layer)
                
                if params['w_bits'] == 2:
                    stats['W2'] += 1
                elif params['w_bits'] == 4:
                    stats['W4'] += 1
                else:
                    stats['W8'] += 1
                    
        except Exception as e:
            pass
    
    return model, stats


def generate_response(model, tokenizer, prompt: str, device: str, 
                      max_new_tokens: int = 150) -> tuple:
    """
    生成模型回复并返回耗时
    
    返回：(回复内容, 生成时间秒, 生成的token数)
    """
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 计时
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # 计算生成的token数
    new_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    return response, elapsed, new_tokens


def print_comparison(prompt: str, original_result: tuple, quant_result: tuple, idx: int):
    """打印对比结果"""
    orig_response, orig_time, orig_tokens = original_result
    quant_response, quant_time, quant_tokens = quant_result
    
    print(f"\n{'='*80}")
    print(f"📝 测试用例 {idx}")
    print(f"{'='*80}")
    print(f"\n🔹 问题: {prompt}")
    
    print(f"\n{'─'*80}")
    print(f"🔵 【原始模型】 Qwen2.5-7B-Instruct")
    print(f"{'─'*80}")
    print(f"{orig_response}")
    print(f"\n   ⏱️  耗时: {orig_time:.2f}s | Tokens: {orig_tokens} | 速度: {orig_tokens/orig_time:.1f} tokens/s")
    
    print(f"\n{'─'*80}")
    print(f"🟢 【量化模型】 Mixed-Precision (W2/W4/W8)")
    print(f"{'─'*80}")
    print(f"{quant_response}")
    print(f"\n   ⏱️  耗时: {quant_time:.2f}s | Tokens: {quant_tokens} | 速度: {quant_tokens/quant_time:.1f} tokens/s")
    
    # 计算加速比
    speedup = orig_time / quant_time if quant_time > 0 else 0
    print(f"\n   📊 加速比: {speedup:.2f}x")


def main():
    """主程序"""
    parser = argparse.ArgumentParser(
        description="对比原始模型与量化模型的推理效果",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace模型ID")
    parser.add_argument('--config', type=str, default="mixed_precision_config.pt",
                        help="混合精度配置文件路径")
    parser.add_argument('--prompt', type=str, default=None,
                        help="自定义测试提示（可选）")
    parser.add_argument('--max_tokens', type=int, default=150,
                        help="最大生成token数")
    
    args = parser.parse_args()
    
    device = get_device()
    
    print("\n" + "="*80)
    print("🔬 Qwen2.5-7B 模型对比测试")
    print("   原始模型 vs 混合精度量化模型")
    print("="*80)
    print(f"\n📍 设备: {device}")
    print(f"📦 模型: {args.model_id}")
    print(f"📄 配置: {args.config}")
    
    # ========== 加载原始模型 ==========
    print("\n" + "─"*80)
    print("⏳ 正在加载原始模型...")
    
    if device == "mps":
        original_model = AutoModelForCausalLM.from_pretrained(
            args.model_id, 
            torch_dtype=torch.float32
        )
        original_model = original_model.to("mps")
    else:
        original_model = AutoModelForCausalLM.from_pretrained(
            args.model_id, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    original_model.eval()
    print("✅ 原始模型加载完成")
    
    # ========== 加载量化模型 ==========
    print("\n⏳ 正在加载量化模型...")
    
    if device == "mps":
        quant_model = AutoModelForCausalLM.from_pretrained(
            args.model_id, 
            torch_dtype=torch.float32
        )
        quant_model = quant_model.to("mps")
    else:
        quant_model = AutoModelForCausalLM.from_pretrained(
            args.model_id, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    
    # 应用混合精度配置
    try:
        config = torch.load(args.config, map_location='cpu')
        quant_model, stats = apply_mixed_precision(quant_model, config)
        
        print("✅ 量化模型加载完成")
        print(f"   📊 量化层统计: W2={stats['W2']}, W4={stats['W4']}, W8={stats['W8']}")
        
        total = stats['W2'] + stats['W4'] + stats['W8']
        bits_total = stats['W2'] * 2 + stats['W4'] * 4 + stats['W8'] * 8
        bits_orig = total * 16
        compression = bits_total / bits_orig if bits_orig > 0 else 1
        print(f"   💾 压缩比: {compression:.1%} | 内存节省: {(1-compression)*100:.1f}%")
        
    except FileNotFoundError:
        print(f"\n❌ 配置文件未找到: {args.config}")
        print("   请先运行: python mixed_precision_ptq.py")
        return
    
    quant_model.eval()
    
    # ========== 测试用例 ==========
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            # 基础数学
            "计算 123 × 456 = ?",
            
            # 知识问答
            "请简要介绍一下太阳系的八大行星。",
            
            # 逻辑推理
            "小明比小红大3岁，小红今年10岁，请问小明5年后多少岁？",
            
            # 代码生成
            "用Python实现一个二分查找算法，要求有详细注释。",
            
            # 创意写作
            "请用一句话描述人工智能的未来。",
            
            # 英文理解
            "Translate the following to Chinese: 'The quick brown fox jumps over the lazy dog.'",
            
            # 专业知识
            "什么是Transformer架构？请简要说明其核心机制。",
            
            # 常识推理
            "为什么天空是蓝色的？用简单的语言解释。",
        ]
    
    print("\n" + "="*80)
    print("🚀 开始对比测试")
    print("="*80)
    
    # 收集统计数据
    total_orig_time = 0
    total_quant_time = 0
    total_orig_tokens = 0
    total_quant_tokens = 0
    
    for idx, prompt in enumerate(prompts, 1):
        # 原始模型推理
        orig_result = generate_response(
            original_model, tokenizer, prompt, device, 
            max_new_tokens=args.max_tokens
        )
        
        # 量化模型推理
        quant_result = generate_response(
            quant_model, tokenizer, prompt, device, 
            max_new_tokens=args.max_tokens
        )
        
        # 打印对比结果
        print_comparison(prompt, orig_result, quant_result, idx)
        
        # 累计统计
        total_orig_time += orig_result[1]
        total_quant_time += quant_result[1]
        total_orig_tokens += orig_result[2]
        total_quant_tokens += quant_result[2]
    
    # ========== 总结统计 ==========
    print("\n" + "="*80)
    print("📊 测试总结")
    print("="*80)
    
    print(f"\n🔵 原始模型:")
    print(f"   总耗时: {total_orig_time:.2f}s")
    print(f"   总Tokens: {total_orig_tokens}")
    print(f"   平均速度: {total_orig_tokens/total_orig_time:.1f} tokens/s")
    
    print(f"\n🟢 量化模型:")
    print(f"   总耗时: {total_quant_time:.2f}s")
    print(f"   总Tokens: {total_quant_tokens}")
    print(f"   平均速度: {total_quant_tokens/total_quant_time:.1f} tokens/s")
    
    avg_speedup = total_orig_time / total_quant_time if total_quant_time > 0 else 0
    print(f"\n📈 平均加速比: {avg_speedup:.2f}x")
    print(f"💾 内存节省: {(1-compression)*100:.1f}%")
    
    print("\n" + "="*80)
    print("✅ 对比测试完成!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
