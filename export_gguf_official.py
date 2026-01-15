"""
GGUF 格式导出工具 (GGUF Export Tool)
====================================

使用 HuggingFace 官方 gguf 库将混合精度配置导出为 GGUF 格式，
确保与 llama.cpp 完全兼容。

工作流程:
---------
1. 加载混合精度配置 (mixed_precision_config.pt)
2. 加载原始 HuggingFace 模型
3. 根据配置对每层进行量化
4. 生成包含 tokenizer 的 GGUF 文件

量化类型映射:
-----------
- W2 (2-bit) → Q4_0（Q2_K 复杂，简化处理）
- W4 (4-bit) → Q4_0
- 其他层 → F32

使用方法:
---------
# 基础用法
python export_gguf_official.py

# 自定义输出
python export_gguf_official.py --output models/my-model.gguf

# 指定配置
python export_gguf_official.py --config my_config.pt
"""

import torch
import numpy as np
import os
import argparse
import gc
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import gguf


# ============================================================================
# 工具函数
# ============================================================================

def get_device() -> str:
    """自动检测最佳计算设备"""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# 使用 gguf 库自带的量化函数
from gguf import quants as gguf_quants


def quantize_tensor(weight: np.ndarray, qtype: gguf.GGMLQuantizationType) -> np.ndarray:
    """
    使用 gguf 库的量化函数对张量进行量化
    
    参数:
        weight: 原始权重张量
        qtype: GGML 量化类型
    
    返回:
        量化后的数据
    """
    return gguf_quants.quantize(weight.astype(np.float32), qtype)


def convert_name_hf_to_gguf(name: str) -> str:
    """
    将 HuggingFace 权重名称转换为 GGUF 格式
    
    映射规则:
        model.layers.N → blk.N
        embed_tokens → token_embd
        self_attn.q_proj → attn_q
        mlp.gate_proj → ffn_gate
        等等
    """
    # 先处理 model. 前缀
    name = name.replace("model.", "")
    
    # 处理层编号
    name = name.replace("layers.", "blk.")
    
    # 处理 embed_tokens
    name = name.replace("embed_tokens.weight", "token_embd.weight")
    
    # 处理 layernorm（注意：使用完整模式避免误替换）
    name = name.replace(".input_layernorm.weight", ".attn_norm.weight")
    name = name.replace(".post_attention_layernorm.weight", ".ffn_norm.weight")
    
    # 处理 attention 相关
    name = name.replace(".self_attn.q_proj.weight", ".attn_q.weight")
    name = name.replace(".self_attn.k_proj.weight", ".attn_k.weight")
    name = name.replace(".self_attn.v_proj.weight", ".attn_v.weight")
    name = name.replace(".self_attn.o_proj.weight", ".attn_output.weight")
    name = name.replace(".self_attn.q_proj.bias", ".attn_q.bias")
    name = name.replace(".self_attn.k_proj.bias", ".attn_k.bias")
    name = name.replace(".self_attn.v_proj.bias", ".attn_v.bias")
    
    # 处理 MLP
    name = name.replace(".mlp.gate_proj.weight", ".ffn_gate.weight")
    name = name.replace(".mlp.up_proj.weight", ".ffn_up.weight")
    name = name.replace(".mlp.down_proj.weight", ".ffn_down.weight")
    
    # 处理最后的 norm（只处理 "norm.weight" 开头的情况，不是 ".attn_norm.weight"）
    if name == "norm.weight":
        name = "output_norm.weight"
    
    # lm_head 特殊处理
    if name == "lm_head.weight":
        name = "output.weight"
    
    return name


def export_mixed_precision_gguf_official(
    model_id: str,
    config_path: str,
    output_path: str
):
    """
    使用官方 gguf 库导出混合精度模型
    """
    print("\n" + "="*80)
    print("🔧 混合精度 GGUF 导出 (官方库)")
    print("="*80)
    
    # 加载量化配置
    print(f"\n📄 加载量化配置: {config_path}")
    quant_config = torch.load(config_path, weights_only=False)
    
    # 统计
    w2_count = sum(1 for v in quant_config.values() if v['w_bits'] == 2)
    w4_count = sum(1 for v in quant_config.values() if v['w_bits'] == 4)
    
    print(f"\n📊 量化配置:")
    print(f"   W2层 (低敏感度): {w2_count}")
    print(f"   W4层 (高敏感度): {w4_count}")
    
    # 加载模型配置
    print(f"\n📦 加载模型: {model_id}")
    hf_config = AutoConfig.from_pretrained(model_id)
    
    # 加载模型
    print("⏳ 加载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cpu"
    )
    
    # 加载 tokenizer 获取词表信息
    print("⏳ 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 创建 GGUF writer
    print(f"\n📝 创建 GGUF 文件: {output_path}")
    writer = gguf.GGUFWriter(output_path, "qwen2")
    
    # 添加模型元数据
    writer.add_architecture()
    writer.add_name(model_id.split("/")[-1] + "-mixed")
    writer.add_context_length(hf_config.max_position_embeddings)
    writer.add_embedding_length(hf_config.hidden_size)
    writer.add_block_count(hf_config.num_hidden_layers)
    writer.add_feed_forward_length(hf_config.intermediate_size)
    writer.add_head_count(hf_config.num_attention_heads)
    writer.add_head_count_kv(hf_config.num_key_value_heads)
    writer.add_rope_freq_base(hf_config.rope_theta)
    writer.add_layer_norm_rms_eps(hf_config.rms_norm_eps)
    
    # 添加完整的 tokenizer 信息
    print("📝 添加 tokenizer 信息...")
    
    # 使用模型的 vocab_size（比 tokenizer 大，有 padding）
    model_vocab_size = hf_config.vocab_size  # 152064
    tokenizer_vocab = tokenizer.get_vocab()
    tokenizer_vocab_size = len(tokenizer_vocab)  # 151665
    
    print(f"   模型 vocab_size: {model_vocab_size}")
    print(f"   Tokenizer vocab_size: {tokenizer_vocab_size}")
    
    # 创建完整的词表（填充到模型大小）
    tokens = [""] * model_vocab_size
    scores = [0.0] * model_vocab_size
    token_types = [gguf.TokenType.NORMAL] * model_vocab_size
    
    for token, idx in tokenizer_vocab.items():
        if idx < model_vocab_size:
            tokens[idx] = token
            scores[idx] = -float(idx)
    
    # 填充未使用的 token 位置
    for i in range(tokenizer_vocab_size, model_vocab_size):
        tokens[i] = f"[PAD_{i}]"
        token_types[i] = gguf.TokenType.UNUSED
    
    # 设置特殊 token
    if tokenizer.bos_token_id is not None and tokenizer.bos_token_id < len(token_types):
        token_types[tokenizer.bos_token_id] = gguf.TokenType.CONTROL
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id < len(token_types):
        token_types[tokenizer.eos_token_id] = gguf.TokenType.CONTROL
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id < len(token_types):
        token_types[tokenizer.pad_token_id] = gguf.TokenType.CONTROL
    if tokenizer.unk_token_id is not None and tokenizer.unk_token_id < len(token_types):
        token_types[tokenizer.unk_token_id] = gguf.TokenType.UNKNOWN
    
    # 添加词表
    writer.add_tokenizer_model("gpt2")
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
    
    # 添加 pre-tokenizer 类型（Qwen2 需要）
    try:
        writer.add_tokenizer_pre("qwen2")
    except:
        # 旧版本 gguf 可能没有这个方法
        pass
    
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    
    # 添加特殊 token ID
    if tokenizer.bos_token_id is not None:
        writer.add_bos_token_id(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        writer.add_eos_token_id(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        writer.add_pad_token_id(tokenizer.pad_token_id)
    
    # 添加 chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        writer.add_chat_template(tokenizer.chat_template)
    
    # 添加 merges（BPE tokenizer 需要）
    try:
        from huggingface_hub import hf_hub_download
        import json
        
        # 从 HuggingFace 下载 tokenizer.json
        tokenizer_json_path = hf_hub_download(model_id, 'tokenizer.json')
        print(f"   Tokenizer JSON: {tokenizer_json_path}")
        
        with open(tokenizer_json_path, 'r') as f:
            tokenizer_json = json.load(f)
        
        if 'model' in tokenizer_json and 'merges' in tokenizer_json['model']:
            merges = tokenizer_json['model']['merges']
            writer.add_token_merges(merges)
            print(f"   ✅ 添加了 {len(merges)} 个 BPE merges")
        else:
            print("   ⚠️  tokenizer.json 中没有 merges")
    except Exception as e:
        print(f"   ⚠️  无法添加 merges: {e}")
    
    # 处理权重
    print("\n🔄 量化并添加权重...")
    
    total_original_size = 0
    total_quantized_size = 0
    
    for name, param in tqdm(model.named_parameters(), desc="处理权重"):
        weight = param.data.cpu().numpy()
        original_size = weight.nbytes
        total_original_size += original_size
        
        # 转换名称
        gguf_name = convert_name_hf_to_gguf(name)
        original_shape = weight.shape
        
        # 查找量化配置
        layer_name = name.replace(".weight", "").replace(".bias", "")
        
        # 决定量化类型
        if layer_name in quant_config and ".weight" in name:
            w_bits = quant_config[layer_name]['w_bits']
            
            if w_bits == 2:
                # Q2_K 太复杂，用 Q4_0 替代
                qtype = gguf.GGMLQuantizationType.Q4_0
                quantized = quantize_tensor(weight, qtype)
                q_str = "Q4_0(W2)"
            else:  # w_bits == 4 (高敏感度层)
                qtype = gguf.GGMLQuantizationType.Q4_0
                quantized = quantize_tensor(weight, qtype)
                q_str = "Q4_0"
            
            total_quantized_size += quantized.nbytes
            
            # 添加量化张量 - 使用 gguf 库的量化数据，不需要手动指定 raw_shape
            writer.add_tensor(gguf_name, quantized, raw_dtype=qtype)
        else:
            # 非量化层，使用 F32 以确保兼容性
            weight_f32 = weight.astype(np.float32)
            total_quantized_size += weight_f32.nbytes
            writer.add_tensor(gguf_name, weight_f32, raw_dtype=gguf.GGMLQuantizationType.F32)
        
        # 释放内存
        del weight
    
    # 清理模型
    del model
    gc.collect()
    
    # 写入文件
    print("\n💾 写入文件...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    # 计算统计
    file_size = os.path.getsize(output_path)
    compression = total_original_size / file_size if file_size > 0 else 1
    
    print(f"\n{'='*80}")
    print(f"✅ 导出完成!")
    print(f"{'='*80}")
    print(f"\n📁 输出文件: {output_path}")
    print(f"📊 原始大小: {total_original_size/1024/1024/1024:.2f} GB")
    print(f"📊 文件大小: {file_size/1024/1024/1024:.2f} GB")
    print(f"📊 压缩比: {compression:.2f}x")


def main():
    """
    主函数：解析参数并执行 GGUF 导出
    """
    parser = argparse.ArgumentParser(
        description="混合精度 GGUF 导出工具（使用官方 gguf 库）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python export_gguf_official.py
  python export_gguf_official.py --output models/custom.gguf
  python export_gguf_official.py --config my_config.pt --output models/my_model.gguf
        """
    )
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace 模型 ID")
    parser.add_argument('--config', type=str, default="mixed_precision_config.pt",
                        help="混合精度量化配置文件路径")
    parser.add_argument('--output', type=str, default="models/qwen2.5-7b-mixed.gguf",
                        help="输出 GGUF 文件路径")
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    export_mixed_precision_gguf_official(
        args.model_id,
        args.config,
        args.output
    )


if __name__ == "__main__":
    main()
