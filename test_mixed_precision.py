"""
模拟量化推理测试 (Simulated Quantization Inference Test)
=======================================================

本脚本用于测试模拟量化后的模型推理效果。

重要说明:
---------
这是【模拟量化】测试，用于验证量化精度，不能获得加速效果！
如需真正的加速，请使用 compare_real_quant.py

功能:
-----
1. 加载预训练模型
2. 应用混合精度量化配置
3. 执行推理测试

使用方法:
---------
python test_mixed_precision.py
python test_mixed_precision.py --prompt "你好，请介绍一下自己"

推荐:
-----
如需测试真实加速效果，请运行:
python compare_real_quant.py
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from quant_utils import MixedPrecisionLinear


# ============================================================================
# 工具函数
# ============================================================================

def get_device() -> str:
    """自动检测最佳可用设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def apply_mixed_precision(model, config: dict) -> tuple:
    """
    将混合精度配置应用到模型
    
    参数:
    -----
    model : AutoModelForCausalLM
        原始模型
    config : dict
        混合精度配置字典
    
    返回:
    ------
    tuple
        (模型, 统计信息字典)
    """
    stats = {'W2': 0, 'W4': 0}
    
    for name, params in config.items():
        parts = name.split('.')
        parent = model
        
        try:
            # 导航到父模块
            for part in parts[:-1]:
                parent = getattr(parent, part)
            layer_name = parts[-1]
            original = getattr(parent, layer_name)
            
            if isinstance(original, nn.Linear):
                # 创建量化层替换原始层
                new_layer = MixedPrecisionLinear(
                    original,
                    w_bits=params['w_bits'],
                    a_bits=params['a_bits'],
                    clip_ratio=params['clip_ratio'],
                    smooth_alpha=params['smooth_alpha']
                )
                setattr(parent, layer_name, new_layer)
                
                # 统计
                if params['w_bits'] == 2:
                    stats['W2'] += 1
                else:  # w_bits == 4
                    stats['W4'] += 1
                    
        except Exception as e:
            print(f"警告: 无法替换层 {name}: {e}")
    
    return model, stats


def generate_response(model, tokenizer, prompt: str, device: str, 
                      max_new_tokens: int = 100) -> str:
    """
    生成模型回复
    
    参数：
    -----
    model : AutoModelForCausalLM
        模型
    tokenizer : AutoTokenizer
        分词器
    prompt : str
        用户输入
    device : str
        计算设备
    max_new_tokens : int
        最大生成token数
    
    返回：
    ------
    str
        模型回复
    """
    # 构建对话格式
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 编码
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 使用贪婪解码确保结果可重复
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码（只取新生成的部分）
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    return response


def main():
    """主程序"""
    parser = argparse.ArgumentParser(
        description="混合精度量化推理测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本测试
  python test_mixed_precision.py
  
  # 自定义提示
  python test_mixed_precision.py --prompt "用Python写一个快速排序"
  
  # 使用自定义配置文件
  python test_mixed_precision.py --config my_config.pt
        """
    )
    
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace模型ID")
    parser.add_argument('--config', type=str, default="mixed_precision_config.pt",
                        help="混合精度配置文件路径")
    parser.add_argument('--prompt', type=str, default=None,
                        help="自定义测试提示（可选）")
    parser.add_argument('--max_tokens', type=int, default=200,
                        help="最大生成token数（默认 200）")
    
    args = parser.parse_args()
    
    device = get_device()
    print("="*60)
    print("混合精度量化推理测试")
    print("="*60)
    print(f"设备: {device}")
    print(f"模型: {args.model_id}")
    print(f"配置: {args.config}")
    print("="*60 + "\n")
    
    # 加载模型
    print("正在加载模型...")
    if device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, 
            torch_dtype=torch.float32
        )
        model = model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    # 加载并应用混合精度配置
    try:
        config = torch.load(args.config, map_location='cpu')
        model, stats = apply_mixed_precision(model, config)
        
        print("\n✓ 成功应用混合精度量化:")
        print(f"  W2层 (低敏感度): {stats['W2']}个")
        print(f"  W4层 (高敏感度): {stats['W4']}个")
        total = stats['W2'] + stats['W4']
        print(f"  总计: {total}个量化层")
        
        # 计算压缩率
        bits_total = stats['W2'] * 2 + stats['W4'] * 4
        bits_orig = total * 16
        compression = bits_total / bits_orig if bits_orig > 0 else 1
        print(f"  压缩比: {compression:.1%}")
        print(f"  内存节省: {(1-compression)*100:.1f}%")
        
    except FileNotFoundError:
        print(f"\n✗ 配置文件未找到: {args.config}")
        print("  请先运行: python mixed_precision_ptq.py")
        return
    
    model.eval()
    
    # 测试用例
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "1+1等于多少？",
            "What is artificial intelligence?",
            "用一句话解释量子计算。",
            "请写一个简单的Python冒泡排序函数。"
        ]
    
    print("\n" + "="*60)
    print("推理测试结果")
    print("="*60)
    
    for prompt in prompts:
        response = generate_response(
            model, tokenizer, prompt, device, 
            max_new_tokens=args.max_tokens
        )
        
        print(f"\n>>> 问题: {prompt}")
        print(f"<<< 回答: {response}")
        print("-" * 40)
    
    print("\n" + "="*60)
    print("✓ 推理测试完成!")
    print("="*60)


if __name__ == "__main__":
    main()
