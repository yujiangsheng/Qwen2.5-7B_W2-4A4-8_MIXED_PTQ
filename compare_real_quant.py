"""
真实量化对比测试 (Real Quantization Comparison)
===============================================

本脚本对比【原始模型】与【真实量化模型】的推理性能。

⚠️ 重要说明：
-----------
这是真实量化测试，使用 llama.cpp 进行真正的低精度推理（INT4）。
与模拟量化不同，真实量化可以获得实际的加速效果！

典型结果：
---------
- 推理速度：提升 5-10 倍
- 内存占用：减少 70-85%
- 回答质量：接近原始模型

支持的加速后端：
--------------
- macOS: Metal (Apple Silicon GPU)
- Linux/Windows: CUDA (NVIDIA GPU)
- CPU: 所有平台

使用方法：
---------
# 默认测试（需要先下载 GGUF 模型）
>>> python compare_real_quant.py

# 自定义测试
>>> python compare_real_quant.py --max_tokens 200

# 下载 GGUF 模型
>>> huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \\
...     Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir models
"""

import torch
import time
import argparse
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


def print_comparison_result(prompt: str, orig_result: tuple, quant_result: tuple, idx: int):
    """打印单个测试用例的对比结果"""
    orig_response, orig_time, orig_tokens = orig_result
    quant_response, quant_time, quant_tokens = quant_result
    
    print(f"\n{'='*80}")
    print(f"📝 测试用例 {idx}")
    print(f"{'='*80}")
    print(f"\n🔹 问题: {prompt}")
    
    print(f"\n{'─'*80}")
    print(f"🔵 【原始模型】 Qwen2.5-7B-Instruct (FP32/FP16)")
    print(f"{'─'*80}")
    print(f"{orig_response[:300]}..." if len(orig_response) > 300 else orig_response)
    print(f"\n   ⏱️  耗时: {orig_time:.2f}s | Tokens: {orig_tokens} | 速度: {orig_tokens/orig_time:.1f} tok/s")
    
    print(f"\n{'─'*80}")
    print(f"🟢 【量化模型】 Q4_K_M GGUF (4-bit, llama.cpp + Metal)")
    print(f"{'─'*80}")
    print(f"{quant_response[:300]}..." if len(quant_response) > 300 else quant_response)
    print(f"\n   ⏱️  耗时: {quant_time:.2f}s | Tokens: {quant_tokens} | 速度: {quant_tokens/quant_time:.1f} tok/s")
    
    speedup = orig_time / quant_time if quant_time > 0 else 0
    print(f"\n   📊 加速比: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="真实量化对比测试 (llama.cpp)")
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Transformers 模型 ID")
    parser.add_argument('--gguf_path', type=str, 
                        default="models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                        help="GGUF 模型路径")
    parser.add_argument('--max_tokens', type=int, default=100,
                        help="最大生成 token 数")
    
    args = parser.parse_args()
    device = get_device()
    
    print("\n" + "="*80)
    print("🚀 真实量化对比测试")
    print("   原始模型 (Transformers) vs Q4_K_M 量化 (llama.cpp + Metal)")
    print("="*80)
    print(f"\n📍 设备: {device}")
    print(f"📦 原始模型: {args.model_id}")
    print(f"📦 量化模型: {args.gguf_path}")
    
    # ========== 加载原始模型 ==========
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
    print(f"✅ 原始模型加载完成 | 参数: {total_params/1e9:.2f}B | 内存: ~{orig_memory:.1f} GB")
    
    # ========== 加载量化模型 ==========
    print("\n⏳ 正在加载量化模型 (llama.cpp)...")
    
    try:
        from llama_cpp import Llama
        
        import os
        gguf_path = args.gguf_path
        if not os.path.exists(gguf_path):
            # 尝试其他路径
            alt_paths = [
                "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                "./Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                os.path.expanduser("~/.cache/huggingface/hub/models--bartowski--Qwen2.5-7B-Instruct-GGUF/snapshots/*/Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
            ]
            for path in alt_paths:
                import glob
                matches = glob.glob(path)
                if matches:
                    gguf_path = matches[0]
                    break
        
        if not os.path.exists(gguf_path):
            print(f"❌ GGUF 模型未找到: {gguf_path}")
            print("请先下载模型:")
            print("huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir models")
            return
        
        # 加载 llama.cpp 模型
        quant_model = Llama(
            model_path=gguf_path,
            n_ctx=4096,        # 上下文长度
            n_gpu_layers=-1,   # 使用所有 GPU 层 (Metal)
            n_threads=8,       # CPU 线程数
            verbose=False
        )
        
        # GGUF 文件大小即内存占用
        quant_memory = os.path.getsize(gguf_path) / 1e9
        print(f"✅ 量化模型加载完成 | 格式: Q4_K_M | 内存: ~{quant_memory:.1f} GB")
        print(f"   💾 内存节省: {(1 - quant_memory/orig_memory)*100:.1f}%")
        
    except ImportError:
        print("❌ llama-cpp-python 未安装")
        print("请运行: CMAKE_ARGS=\"-DLLAMA_METAL=on\" pip install llama-cpp-python")
        return
    except Exception as e:
        print(f"❌ 加载量化模型失败: {e}")
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
    
    # 收集统计数据
    total_orig_time = 0
    total_quant_time = 0
    total_orig_tokens = 0
    total_quant_tokens = 0
    
    for idx, prompt in enumerate(prompts, 1):
        # 原始模型推理
        orig_result = generate_with_transformers(
            original_model, tokenizer, prompt, device, 
            max_new_tokens=args.max_tokens
        )
        
        # 量化模型推理
        quant_result = generate_with_llamacpp(
            quant_model, prompt, 
            max_new_tokens=args.max_tokens
        )
        
        # 打印对比结果
        print_comparison_result(prompt, orig_result, quant_result, idx)
        
        # 累计统计
        total_orig_time += orig_result[1]
        total_quant_time += quant_result[1]
        total_orig_tokens += orig_result[2]
        total_quant_tokens += quant_result[2]
    
    # ========== 总结统计 ==========
    print("\n" + "="*80)
    print("📊 测试总结")
    print("="*80)
    
    print(f"\n🔵 原始模型 (Transformers, {'FP32' if device == 'mps' else 'FP16'}):")
    print(f"   内存占用: ~{orig_memory:.1f} GB")
    print(f"   总耗时: {total_orig_time:.2f}s")
    print(f"   总Tokens: {total_orig_tokens}")
    print(f"   平均速度: {total_orig_tokens/total_orig_time:.1f} tok/s")
    
    print(f"\n🟢 量化模型 (llama.cpp, Q4_K_M):")
    print(f"   内存占用: ~{quant_memory:.1f} GB")
    print(f"   总耗时: {total_quant_time:.2f}s")
    print(f"   总Tokens: {total_quant_tokens}")
    print(f"   平均速度: {total_quant_tokens/total_quant_time:.1f} tok/s")
    
    avg_speedup = total_orig_time / total_quant_time if total_quant_time > 0 else 0
    memory_saving = (1 - quant_memory/orig_memory) * 100
    
    print(f"\n{'─'*80}")
    print(f"📈 平均加速比: {avg_speedup:.2f}x")
    print(f"💾 内存节省: {memory_saving:.1f}%")
    print(f"{'─'*80}")
    
    if avg_speedup > 1:
        print(f"\n✅ 量化模型比原始模型快 {avg_speedup:.1f} 倍!")
    else:
        print(f"\n⚠️  量化模型较慢，可能是因为 Metal 优化问题")
    
    print("\n" + "="*80)
    print("✅ 对比测试完成!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
