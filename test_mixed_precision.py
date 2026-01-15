"""
模拟量化推理测试 (Simulated Quantization Test)
==============================================

⚠️ 这是模拟量化测试，不会加速！真正加速请用 compare_real_quant.py。

功能:
  - 加载模型并应用混合精度配置
  - 执行推理测试验证量化精度

用法:
  python test_mixed_precision.py
  python test_mixed_precision.py --prompt "你好"
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from quant_utils import MixedPrecisionLinear


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apply_mixed_precision(model, config: dict) -> tuple:
    """应用混合精度配置到模型"""
    stats = {'A4': 0, 'A8': 0}
    
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
                stats['A4' if params['a_bits'] == 4 else 'A8'] += 1
        except Exception as e:
            print(f"⚠️ 跳过 {name}: {e}")
    
    return model, stats


def generate_response(model, tokenizer, prompt: str, device: str, max_tokens: int = 100) -> str:
    """生成回复"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="模拟量化推理测试 (W4 + A4/A8)")
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--config', type=str, default="mixed_precision_config.pt")
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--max_tokens', type=int, default=200)
    
    args = parser.parse_args()
    device = get_device()
    
    print("\n" + "="*60)
    print("🧪 模拟量化推理测试 (W4 + A4/A8)")
    print("="*60)
    print(f"  设备: {device}")
    print(f"  模型: {args.model_id}")
    print("="*60 + "\n")
    
    # 加载模型
    print("📦 加载模型...")
    if device == "mps":
        model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
        model = model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, torch_dtype=torch.float16, device_map="auto"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    # 应用配置
    try:
        config = torch.load(args.config, map_location='cpu')
        model, stats = apply_mixed_precision(model, config)
        
        total = stats['A4'] + stats['A8']
        avg_a_bits = (stats['A4'] * 4 + stats['A8'] * 8) / total if total > 0 else 8
        
        print(f"\n✅ 应用混合精度配置:")
        print(f"   A4层: {stats['A4']}个, A8层: {stats['A8']}个")
        print(f"   平均激活位宽: {avg_a_bits:.2f} bit")
    except FileNotFoundError:
        print(f"❌ 配置文件未找到: {args.config}")
        print("   请先运行: python mixed_precision_ptq.py")
        return
    
    model.eval()
    
    # 测试
    prompts = [args.prompt] if args.prompt else [
        "1+1等于多少？",
        "用一句话解释量子计算。",
        "用Python写一个冒泡排序。"
    ]
    
    print("\n" + "="*60)
    print("📝 推理测试")
    print("="*60)
    
    for prompt in prompts:
        response = generate_response(model, tokenizer, prompt, device, args.max_tokens)
        print(f"\n>>> {prompt}")
        print(f"<<< {response}")
        print("-" * 40)
    
    print("\n✅ 测试完成!")


if __name__ == "__main__":
    main()
