"""
真实量化三模型对比测试 (Real Quantization Comparison)
=====================================================

本脚本对比三种模型的推理性能和输出质量：
1. 原始模型 (FP32/FP16) - 使用 Transformers 库
2. 混合精度量化模型 (W2/W4) - 使用 llama.cpp (自定义 GGUF)
3. Q4_K_M 统一量化 (4-bit) - 使用 llama.cpp (标准 GGUF)

⚠️ 重要说明：
-----------
这是真实量化测试，使用 llama.cpp 进行真正的低精度推理。
与模拟量化不同，真实量化可以获得实际的加速效果！

典型结果：
---------
- 推理速度：提升 5-10 倍
- 内存占用：减少 70-85%
- 回答质量：接近原始模型

使用方法：
---------
# 完整三模型对比
>>> python compare_real_quant.py

# 跳过原始模型（节省内存）
>>> python compare_real_quant.py --skip_original

# 自定义测试
>>> python compare_real_quant.py --max_tokens 200

# 下载 Q4_K_M 模型
>>> huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \\
...     Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir models
"""

import torch
import time
import argparse
import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def generate_with_transformers(model, tokenizer, prompt: str, device: str, 
                                max_new_tokens: int = 100) -> tuple:
    """
    使用 Transformers 生成回复（原始模型）
    
    参数:
        model: HuggingFace 模型
        tokenizer: 分词器
        prompt: 用户输入
        device: 计算设备
        max_new_tokens: 最大生成 token 数
    
    返回:
        (回复内容, 耗时秒数, 生成的token数)
    """
    # 构建对话格式
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 预热（让 GPU 进入工作状态）
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=3, pad_token_id=tokenizer.eos_token_id)
    
    # 正式推理并计时
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 贪婪解码，结果可复现
            pad_token_id=tokenizer.eos_token_id
        )
    
    elapsed = time.time() - start_time
    new_tokens = outputs.shape[1] - inputs['input_ids'].shape[1]
    
    # 解码输出
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    return response, elapsed, new_tokens


def generate_with_llamacpp(llm, prompt: str, max_new_tokens: int = 100) -> tuple:
    """
    使用 llama.cpp 生成回复（真实量化模型）
    
    llama.cpp 使用真正的低精度整数运算，可以获得实际加速。
    
    参数:
        llm: llama_cpp.Llama 模型实例
        prompt: 用户输入
        max_new_tokens: 最大生成 token 数
    
    返回:
        (回复内容, 耗时秒数, 生成的token数)
    """
    # Qwen2.5 的聊天模板格式
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # 预热
    _ = llm(formatted_prompt, max_tokens=3, echo=False)
    
    # 正式推理并计时
    start_time = time.time()
    
    output = llm(
        formatted_prompt,
        max_tokens=max_new_tokens,
        echo=False,
        stop=["<|im_end|>", "<|endoftext|>"]  # 停止词
    )
    
    elapsed = time.time() - start_time
    
    response = output['choices'][0]['text'].strip()
    tokens = output['usage']['completion_tokens']
    
    return response, elapsed, tokens


def find_gguf_model(path: str, alt_paths: list = None) -> str:
    """
    查找 GGUF 模型文件
    
    参数:
        path: 主路径
        alt_paths: 备选路径列表
    
    返回:
        找到的模型路径，如果未找到返回 None
    """
    if os.path.exists(path):
        return path
    
    if alt_paths:
        for alt_path in alt_paths:
            matches = glob.glob(alt_path)
            if matches:
                return matches[0]
    
    return None


def print_result(name: str, response: str, elapsed: float, tokens: int, icon: str = ""):
    """打印单个模型的结果"""
    print(f"\n{'─'*80}")
    print(f"{icon}【{name}】")
    print(f"{'─'*80}")
    print(f"{response[:400]}..." if len(response) > 400 else response)
    speed = tokens / elapsed if elapsed > 0 else 0
    print(f"\n   ⏱️  耗时: {elapsed:.2f}s | Tokens: {tokens} | 速度: {speed:.1f} tok/s")


def main():
    parser = argparse.ArgumentParser(
        description="真实量化三模型对比测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整三模型对比
  python compare_real_quant.py
  
  # 跳过原始模型（节省内存）
  python compare_real_quant.py --skip_original
  
  # 只测试量化模型
  python compare_real_quant.py --skip_original --max_tokens 300
        """
    )
    
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Transformers 模型 ID")
    parser.add_argument('--q4km_path', type=str, 
                        default="models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                        help="Q4_K_M GGUF 模型路径")
    parser.add_argument('--mixed_path', type=str, 
                        default="models/qwen2.5-7b-mixed.gguf",
                        help="混合精度 GGUF 模型路径")
    parser.add_argument('--max_tokens', type=int, default=200,
                        help="最大生成 token 数（默认 200）")
    parser.add_argument('--skip_original', action='store_true',
                        help="跳过原始模型测试（节省内存）")
    
    args = parser.parse_args()
    device = get_device()
    
    print("\n" + "="*80)
    print("🚀 真实量化三模型对比测试")
    print("="*80)
    print(f"\n对比以下模型:")
    print(f"  1. 原始模型 (Transformers, FP32/FP16)")
    print(f"  2. 混合精度量化 (W2/W4, llama.cpp)")
    print(f"  3. Q4_K_M 统一量化 (4-bit, llama.cpp)")
    print(f"\n📍 设备: {device}")
    
    # 模型和统计数据
    models = {}
    stats = {
        'original': {'time': 0, 'tokens': 0, 'memory': 0},
        'mixed': {'time': 0, 'tokens': 0, 'memory': 0},
        'q4km': {'time': 0, 'tokens': 0, 'memory': 0},
    }
    tokenizer = None
    
    # ========== 加载原始模型 ==========
    if not args.skip_original:
        print("\n" + "─"*80)
        print("⏳ 正在加载原始模型 (Transformers)...")
        
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
        
        # 估算内存
        total_params = sum(p.numel() for p in original_model.parameters())
        orig_memory = total_params * 4 / 1e9 if device == "mps" else total_params * 2 / 1e9
        stats['original']['memory'] = orig_memory
        
        models['original'] = original_model
        print(f"✅ 原始模型加载完成 | 参数: {total_params/1e9:.2f}B | 内存: ~{orig_memory:.1f} GB")
    else:
        print("\n⏭️  跳过原始模型加载")
    
    # ========== 加载 llama.cpp ==========
    try:
        from llama_cpp import Llama
    except ImportError:
        print("\n❌ llama-cpp-python 未安装")
        print("请运行: CMAKE_ARGS=\"-DLLAMA_METAL=on\" pip install llama-cpp-python")
        return
    
    # ========== 加载 Q4_K_M 模型 ==========
    print("\n⏳ 正在加载 Q4_K_M 量化模型...")
    
    q4km_path = find_gguf_model(args.q4km_path, [
        "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "./Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        os.path.expanduser("~/.cache/huggingface/hub/models--bartowski--Qwen2.5-7B-Instruct-GGUF/snapshots/*/Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    ])
    
    if q4km_path:
        try:
            models['q4km'] = Llama(
                model_path=q4km_path,
                n_ctx=4096,
                n_gpu_layers=-1,
                n_threads=8,
                verbose=False
            )
            q4km_memory = os.path.getsize(q4km_path) / 1e9
            stats['q4km']['memory'] = q4km_memory
            print(f"✅ Q4_K_M 模型加载完成 | 内存: ~{q4km_memory:.1f} GB")
        except Exception as e:
            print(f"⚠️  Q4_K_M 模型加载失败: {e}")
    else:
        print(f"⚠️  Q4_K_M 模型未找到: {args.q4km_path}")
        print("   下载命令: huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir models")
    
    # ========== 加载混合精度模型 ==========
    print("\n⏳ 正在加载混合精度量化模型...")
    
    mixed_path = find_gguf_model(args.mixed_path, [
        "models/qwen2.5-7b-mixed.gguf",
        "./qwen2.5-7b-mixed.gguf",
    ])
    
    if mixed_path:
        try:
            models['mixed'] = Llama(
                model_path=mixed_path,
                n_ctx=4096,
                n_gpu_layers=-1,
                n_threads=8,
                verbose=False
            )
            mixed_memory = os.path.getsize(mixed_path) / 1e9
            stats['mixed']['memory'] = mixed_memory
            print(f"✅ 混合精度模型加载完成 | 内存: ~{mixed_memory:.1f} GB")
        except Exception as e:
            print(f"⚠️  混合精度模型加载失败: {e}")
    else:
        print(f"⚠️  混合精度模型未找到: {args.mixed_path}")
        print("   请先运行: python export_gguf_official.py")
    
    # 检查是否有模型可测试
    if not models:
        print("\n❌ 没有可测试的模型，请先加载至少一个模型")
        return
    
    # ========== 测试用例 ==========
    prompts = [
        "1+1等于多少？",
        "什么是Transformer架构？用一句话解释。",
        "用Python写一个快速排序算法。",
        "请简要介绍太阳系的八大行星。",
        "为什么天空是蓝色的？用简单语言解释。",
    ]
    
    print("\n" + "="*80)
    print("🚀 开始对比测试")
    print("="*80)
    
    for idx, prompt in enumerate(prompts, 1):
        print(f"\n{'='*80}")
        print(f"📝 测试用例 {idx}")
        print(f"{'='*80}")
        print(f"\n🔹 问题: {prompt}")
        
        results = {}
        
        # 原始模型推理
        if 'original' in models:
            try:
                resp, elapsed, tokens = generate_with_transformers(
                    models['original'], tokenizer, prompt, device, 
                    max_new_tokens=args.max_tokens
                )
                results['original'] = (resp, elapsed, tokens)
                stats['original']['time'] += elapsed
                stats['original']['tokens'] += tokens
                print_result("原始模型 (FP32/FP16)", resp, elapsed, tokens, "🔵 ")
            except Exception as e:
                print(f"\n⚠️  原始模型推理失败: {e}")
        
        # Q4_K_M 模型推理
        if 'q4km' in models:
            try:
                resp, elapsed, tokens = generate_with_llamacpp(
                    models['q4km'], prompt, 
                    max_new_tokens=args.max_tokens
                )
                results['q4km'] = (resp, elapsed, tokens)
                stats['q4km']['time'] += elapsed
                stats['q4km']['tokens'] += tokens
                print_result("Q4_K_M 统一量化 (4-bit)", resp, elapsed, tokens, "🟢 ")
            except Exception as e:
                print(f"\n⚠️  Q4_K_M 模型推理失败: {e}")
        
        # 混合精度模型推理
        if 'mixed' in models:
            try:
                resp, elapsed, tokens = generate_with_llamacpp(
                    models['mixed'], prompt, 
                    max_new_tokens=args.max_tokens
                )
                results['mixed'] = (resp, elapsed, tokens)
                stats['mixed']['time'] += elapsed
                stats['mixed']['tokens'] += tokens
                print_result("混合精度量化 (W2/W4)", resp, elapsed, tokens, "🟡 ")
            except Exception as e:
                print(f"\n⚠️  混合精度模型推理失败: {e}")
        
        # 速度对比
        if len(results) >= 2:
            print(f"\n{'─'*80}")
            print("📊 速度对比:")
            
            if 'original' in results and 'q4km' in results:
                speedup = results['original'][1] / results['q4km'][1]
                print(f"   Q4_K_M vs 原始: {speedup:.2f}x 加速")
            
            if 'original' in results and 'mixed' in results:
                speedup = results['original'][1] / results['mixed'][1]
                print(f"   混合精度 vs 原始: {speedup:.2f}x 加速")
            
            if 'q4km' in results and 'mixed' in results:
                ratio = results['q4km'][1] / results['mixed'][1]
                if ratio > 1:
                    print(f"   混合精度 vs Q4_K_M: {ratio:.2f}x 更快")
                else:
                    print(f"   混合精度 vs Q4_K_M: {1/ratio:.2f}x 更慢")
    
    # ========== 总结统计 ==========
    print("\n" + "="*80)
    print("📊 测试总结")
    print("="*80)
    
    print("\n┌─────────────────────┬──────────────┬──────────┬──────────────┐")
    print("│ 模型                │ 内存占用     │ 总耗时   │ 平均速度     │")
    print("├─────────────────────┼──────────────┼──────────┼──────────────┤")
    
    if 'original' in models and stats['original']['time'] > 0:
        orig_speed = stats['original']['tokens'] / stats['original']['time']
        print(f"│ 原始 (FP32/FP16)    │ ~{stats['original']['memory']:5.1f} GB    │ {stats['original']['time']:6.2f}s  │ {orig_speed:6.1f} tok/s  │")
    
    if 'q4km' in models and stats['q4km']['time'] > 0:
        q4km_speed = stats['q4km']['tokens'] / stats['q4km']['time']
        print(f"│ Q4_K_M (4-bit)      │ ~{stats['q4km']['memory']:5.1f} GB    │ {stats['q4km']['time']:6.2f}s  │ {q4km_speed:6.1f} tok/s  │")
    
    if 'mixed' in models and stats['mixed']['time'] > 0:
        mixed_speed = stats['mixed']['tokens'] / stats['mixed']['time']
        print(f"│ 混合精度 (W2/W4)    │ ~{stats['mixed']['memory']:5.1f} GB    │ {stats['mixed']['time']:6.2f}s  │ {mixed_speed:6.1f} tok/s  │")
    
    print("└─────────────────────┴──────────────┴──────────┴──────────────┘")
    
    # 对比分析
    print("\n📈 对比分析:")
    
    if 'original' in models and stats['original']['time'] > 0:
        if stats['q4km']['time'] > 0:
            speedup = stats['original']['time'] / stats['q4km']['time']
            saving = (1 - stats['q4km']['memory'] / stats['original']['memory']) * 100
            print(f"   • Q4_K_M 比原始模型快 {speedup:.1f}x，内存减少 {saving:.0f}%")
        
        if stats['mixed']['time'] > 0:
            speedup = stats['original']['time'] / stats['mixed']['time']
            saving = (1 - stats['mixed']['memory'] / stats['original']['memory']) * 100
            print(f"   • 混合精度比原始模型快 {speedup:.1f}x，内存减少 {saving:.0f}%")
    
    if stats['q4km']['time'] > 0 and stats['mixed']['time'] > 0:
        ratio = stats['q4km']['time'] / stats['mixed']['time']
        size_ratio = stats['mixed']['memory'] / stats['q4km']['memory']
        if ratio > 1:
            print(f"   • 混合精度比 Q4_K_M 快 {ratio:.1f}x，大小为其 {size_ratio:.1%}")
        else:
            print(f"   • 混合精度比 Q4_K_M 慢 {1/ratio:.1f}x，大小为其 {size_ratio:.1%}")
    
    print("\n" + "="*80)
    print("✅ 对比测试完成!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
