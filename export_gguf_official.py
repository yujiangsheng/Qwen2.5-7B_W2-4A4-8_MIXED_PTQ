"""
GGUF 格式导出工具 (GGUF Export Tool)
===================================

使用官方 gguf 库将混合精度配置导出为 GGUF 格式，兼容 llama.cpp。

量化策略 (W4 + A4/A8):
  - 权重: 固定 W4 → Q4_0 量化
  - 激活: A4/A8 (导出时不影响权重格式)

用法:
  python export_gguf_official.py
  python export_gguf_official.py --output models/custom.gguf
"""

import torch
import numpy as np
import os
import argparse
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
import gguf
from gguf import quants as gguf_quants


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def quantize_tensor(weight: np.ndarray, qtype: gguf.GGMLQuantizationType) -> np.ndarray:
    """使用 gguf 库量化张量"""
    return gguf_quants.quantize(weight.astype(np.float32), qtype)


def convert_name_hf_to_gguf(name: str) -> str:
    """将 HuggingFace 权重名转换为 GGUF 格式"""
    name = name.replace("model.", "")
    name = name.replace("layers.", "blk.")
    name = name.replace("embed_tokens.weight", "token_embd.weight")
    name = name.replace(".input_layernorm.weight", ".attn_norm.weight")
    name = name.replace(".post_attention_layernorm.weight", ".ffn_norm.weight")
    name = name.replace(".self_attn.q_proj.weight", ".attn_q.weight")
    name = name.replace(".self_attn.k_proj.weight", ".attn_k.weight")
    name = name.replace(".self_attn.v_proj.weight", ".attn_v.weight")
    name = name.replace(".self_attn.o_proj.weight", ".attn_output.weight")
    name = name.replace(".self_attn.q_proj.bias", ".attn_q.bias")
    name = name.replace(".self_attn.k_proj.bias", ".attn_k.bias")
    name = name.replace(".self_attn.v_proj.bias", ".attn_v.bias")
    name = name.replace(".mlp.gate_proj.weight", ".ffn_gate.weight")
    name = name.replace(".mlp.up_proj.weight", ".ffn_up.weight")
    name = name.replace(".mlp.down_proj.weight", ".ffn_down.weight")
    if name == "norm.weight":
        name = "output_norm.weight"
    if name == "lm_head.weight":
        name = "output.weight"
    return name


def export_mixed_precision_gguf(model_id: str, config_path: str, output_path: str):
    """导出混合精度 GGUF 模型"""
    print("\n" + "="*70)
    print("🔧 混合精度 GGUF 导出")
    print("="*70)
    
    # 加载量化配置
    print(f"\n📄 加载配置: {config_path}")
    quant_config = torch.load(config_path, weights_only=False)
    
    a4_count = sum(1 for v in quant_config.values() if v['a_bits'] == 4)
    a8_count = sum(1 for v in quant_config.values() if v['a_bits'] == 8)
    print(f"   A4层: {a4_count}, A8层: {a8_count}")
    
    # 加载模型
    print(f"\n📦 加载模型: {model_id}")
    hf_config = AutoConfig.from_pretrained(model_id)
    
    print("⏳ 加载权重...")
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu")
    
    print("⏳ 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 创建 GGUF writer
    print(f"\n📝 创建 GGUF: {output_path}")
    writer = gguf.GGUFWriter(output_path, "qwen2")
    
    # 添加元数据
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
    
    # 添加 tokenizer
    print("📝 添加 tokenizer...")
    model_vocab_size = hf_config.vocab_size
    tokenizer_vocab = tokenizer.get_vocab()
    tokenizer_vocab_size = len(tokenizer_vocab)
    
    tokens = [""] * model_vocab_size
    scores = [0.0] * model_vocab_size
    token_types = [gguf.TokenType.NORMAL] * model_vocab_size
    
    for token, idx in tokenizer_vocab.items():
        if idx < model_vocab_size:
            tokens[idx] = token
            scores[idx] = -float(idx)
    
    for i in range(tokenizer_vocab_size, model_vocab_size):
        tokens[i] = f"[PAD_{i}]"
        token_types[i] = gguf.TokenType.UNUSED
    
    if tokenizer.bos_token_id is not None:
        token_types[tokenizer.bos_token_id] = gguf.TokenType.CONTROL
    if tokenizer.eos_token_id is not None:
        token_types[tokenizer.eos_token_id] = gguf.TokenType.CONTROL
    if tokenizer.pad_token_id is not None:
        token_types[tokenizer.pad_token_id] = gguf.TokenType.CONTROL
    
    writer.add_tokenizer_model("gpt2")
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
    
    try:
        writer.add_tokenizer_pre("qwen2")
    except:
        pass
    
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    
    if tokenizer.bos_token_id is not None:
        writer.add_bos_token_id(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        writer.add_eos_token_id(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        writer.add_pad_token_id(tokenizer.pad_token_id)
    
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        writer.add_chat_template(tokenizer.chat_template)
    
    # 添加 merges
    try:
        from huggingface_hub import hf_hub_download
        import json
        tokenizer_json = hf_hub_download(model_id, 'tokenizer.json')
        with open(tokenizer_json, 'r') as f:
            tj = json.load(f)
        if 'model' in tj and 'merges' in tj['model']:
            writer.add_token_merges(tj['model']['merges'])
            print(f"   添加了 {len(tj['model']['merges'])} 个 merges")
    except Exception as e:
        print(f"   ⚠️ 无法添加 merges: {e}")
    
    # 量化权重
    print("\n🔄 量化权重...")
    total_orig, total_quant = 0, 0
    
    for name, param in tqdm(model.named_parameters(), desc="处理权重"):
        weight = param.data.cpu().numpy()
        total_orig += weight.nbytes
        
        gguf_name = convert_name_hf_to_gguf(name)
        layer_name = name.replace(".weight", "").replace(".bias", "")
        
        if layer_name in quant_config and ".weight" in name:
            # 权重使用 Q4_0 量化
            qtype = gguf.GGMLQuantizationType.Q4_0
            quantized = quantize_tensor(weight, qtype)
            total_quant += quantized.nbytes
            writer.add_tensor(gguf_name, quantized, raw_dtype=qtype)
        else:
            # 非量化层使用 F32
            weight_f32 = weight.astype(np.float32)
            total_quant += weight_f32.nbytes
            writer.add_tensor(gguf_name, weight_f32, raw_dtype=gguf.GGMLQuantizationType.F32)
        
        del weight
    
    del model
    gc.collect()
    
    # 写入文件
    print("\n💾 写入文件...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    file_size = os.path.getsize(output_path)
    compression = total_orig / file_size if file_size > 0 else 1
    
    print(f"\n{'='*70}")
    print(f"✅ 导出完成!")
    print(f"{'='*70}")
    print(f"   输出: {output_path}")
    print(f"   原始大小: {total_orig/1e9:.2f} GB")
    print(f"   文件大小: {file_size/1e9:.2f} GB")
    print(f"   压缩比: {compression:.2f}x")


def main():
    parser = argparse.ArgumentParser(description="混合精度 GGUF 导出工具")
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument('--config', type=str, default="mixed_precision_config.pt")
    parser.add_argument('--output', type=str, default="models/qwen2.5-7b-mixed.gguf")
    
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    export_mixed_precision_gguf(args.model_id, args.config, args.output)


if __name__ == "__main__":
    main()
