"""
真实量化三模型对比测试 (Real Quantization Comparison)
=====================================================

对比三种模型的推理性能:
  1. 原始模型 (FP32/FP16) - Transformers
  2. 混合精度量化 (W4 + A4/A8) - llama.cpp
  3. Q4_K_M 统一量化 (4-bit) - llama.cpp

这是真实量化测试，可获得实际加速效果！

用法:
  python compare_real_quant.py
  python compare_real_quant.py --skip_original --max_tokens 200
"""

import torch
import time
import argparse
import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_with_transformers(model, tokenizer, prompt: str, device: str, max_tokens: int = 100):
    """使用 Transformers 生成回复"""
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 预热
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=3, pad_token_id=tokenizer.eos_token_id)
    
    # 推理
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    elapsed = time.time() - start
    tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response, elapsed, tokens


def generate_with_llamacpp(llm, prompt: str, max_tokens: int = 100):
    """使用 llama.cpp 生成回复"""
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # 预热
    _ = llm(formatted, max_tokens=3, echo=False)
    
    # 推理
    start = time.time()
    output = llm(formatted, max_tokens=max_tokens, echo=False, stop=["<|im_end|>", "<|endoftext|>"])
    elapsed = time.time() - start
    
    response = output['choices'][0]['text'].strip()
    tokens = output['usage']['completion_tokens']
    return response, elapsed, tokens


def find_gguf(path: str, alt_paths: list = None) -> str:
    """查找 GGUF 模型文件"""
    if os.path.exists(path):
        return path
    if alt_paths:
        for p in alt_paths:
            matches = glob.glob(p)
            if matches:
                return matches[0]
    return None


def print_result(name: str, response: str, elapsed: float, tokens: int, icon: str = ""):
    """打印结果"""
    print(f"\n{'─'*70}")
    print(f"{icon}【{name}】")
    print(f"{'─'*70}")
    print(response[:350] + "..." if len(response) > 350 else response)
    speed = tokens / elapsed if elapsed > 0 else 0
    print(f"\n   ⏱️ {elapsed:.2f}s | {tokens} tokens | {speed:.1f} tok/s")


def main():
    parser = argparse.ArgumentParser(description="真实量化三模型对比")
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--q4km_path', type=str, default="models/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    parser.add_argument('--mixed_path', type=str, default="models/qwen2.5-7b-mixed.gguf")
    parser.add_argument('--max_tokens', type=int, default=200)
    parser.add_argument('--skip_original', action='store_true', help="跳过原始模型")
    
    args = parser.parse_args()
    device = get_device()
    
    print("\n" + "="*70)
    print("🚀 真实量化三模型对比测试")
    print("="*70)
    print(f"📍 设备: {device}")
    
    models = {}
    stats = {k: {'time': 0, 'tokens': 0, 'memory': 0} for k in ['original', 'mixed', 'q4km']}
    tokenizer = None
    
    # 加载原始模型
    if not args.skip_original:
        print("\n⏳ 加载原始模型...")
        if device == "mps":
            original = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float32)
            original = original.to("mps")
        else:
            original = AutoModelForCausalLM.from_pretrained(
                args.model_id, torch_dtype=torch.float16, device_map="auto"
            )
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        original.eval()
        
        params = sum(p.numel() for p in original.parameters())
        mem = params * 4 / 1e9 if device == "mps" else params * 2 / 1e9
        stats['original']['memory'] = mem
        models['original'] = original
        print(f"✅ 原始模型 | {params/1e9:.2f}B 参数 | ~{mem:.1f} GB")
    else:
        print("\n⏭️ 跳过原始模型")
    
    # 加载 llama.cpp
    try:
        from llama_cpp import Llama
    except ImportError:
        print("\n❌ 请安装: CMAKE_ARGS=\"-DLLAMA_METAL=on\" pip install llama-cpp-python")
        return
    
    # 加载 Q4_K_M
    print("\n⏳ 加载 Q4_K_M...")
    q4km_path = find_gguf(args.q4km_path, ["models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"])
    if q4km_path:
        try:
            models['q4km'] = Llama(model_path=q4km_path, n_ctx=4096, n_gpu_layers=-1, verbose=False)
            stats['q4km']['memory'] = os.path.getsize(q4km_path) / 1e9
            print(f"✅ Q4_K_M | ~{stats['q4km']['memory']:.1f} GB")
        except Exception as e:
            print(f"⚠️ Q4_K_M 加载失败: {e}")
    else:
        print(f"⚠️ Q4_K_M 未找到")
    
    # 加载混合精度
    print("\n⏳ 加载混合精度模型...")
    mixed_path = find_gguf(args.mixed_path, ["models/qwen2.5-7b-mixed.gguf"])
    if mixed_path:
        try:
            models['mixed'] = Llama(model_path=mixed_path, n_ctx=4096, n_gpu_layers=-1, verbose=False)
            stats['mixed']['memory'] = os.path.getsize(mixed_path) / 1e9
            print(f"✅ 混合精度 | ~{stats['mixed']['memory']:.1f} GB")
        except Exception as e:
            print(f"⚠️ 混合精度加载失败: {e}")
    else:
        print(f"⚠️ 混合精度未找到，请先运行 export_gguf_official.py")
    
    if not models:
        print("\n❌ 没有可测试的模型")
        return
    
    # 测试
    prompts = [
        "1+1等于多少？",
        "什么是Transformer架构？用一句话解释。",
        "用Python写一个快速排序。",
    ]
    
    print("\n" + "="*70)
    print("🚀 开始测试")
    print("="*70)
    
    for idx, prompt in enumerate(prompts, 1):
        print(f"\n{'='*70}")
        print(f"📝 测试 {idx}: {prompt}")
        print("="*70)
        
        results = {}
        
        if 'original' in models:
            try:
                r, t, n = generate_with_transformers(models['original'], tokenizer, prompt, device, args.max_tokens)
                results['original'] = (r, t, n)
                stats['original']['time'] += t
                stats['original']['tokens'] += n
                print_result("原始模型 (FP32/FP16)", r, t, n, "🔵 ")
            except Exception as e:
                print(f"⚠️ 原始模型失败: {e}")
        
        if 'q4km' in models:
            try:
                r, t, n = generate_with_llamacpp(models['q4km'], prompt, args.max_tokens)
                results['q4km'] = (r, t, n)
                stats['q4km']['time'] += t
                stats['q4km']['tokens'] += n
                print_result("Q4_K_M (4-bit)", r, t, n, "🟢 ")
            except Exception as e:
                print(f"⚠️ Q4_K_M 失败: {e}")
        
        if 'mixed' in models:
            try:
                r, t, n = generate_with_llamacpp(models['mixed'], prompt, args.max_tokens)
                results['mixed'] = (r, t, n)
                stats['mixed']['time'] += t
                stats['mixed']['tokens'] += n
                print_result("混合精度 (W4 + A4/A8)", r, t, n, "🟡 ")
            except Exception as e:
                print(f"⚠️ 混合精度失败: {e}")
    
    # 总结
    print("\n" + "="*70)
    print("📊 测试总结")
    print("="*70)
    
    print("\n┌────────────────────┬────────────┬──────────┬────────────┐")
    print("│ 模型               │ 内存       │ 总耗时   │ 平均速度   │")
    print("├────────────────────┼────────────┼──────────┼────────────┤")
    
    for key, name in [('original', '原始 (FP32/FP16)'), ('q4km', 'Q4_K_M (4-bit)'), ('mixed', '混合精度 (W4A4/8)')]:
        if key in models and stats[key]['time'] > 0:
            speed = stats[key]['tokens'] / stats[key]['time']
            print(f"│ {name:<18} │ ~{stats[key]['memory']:5.1f} GB  │ {stats[key]['time']:6.2f}s  │ {speed:6.1f} tok/s│")
    
    print("└────────────────────┴────────────┴──────────┴────────────┘")
    
    print("\n✅ 对比测试完成!")


if __name__ == "__main__":
    main()
