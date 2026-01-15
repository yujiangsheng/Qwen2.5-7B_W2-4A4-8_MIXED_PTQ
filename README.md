# Qwen2.5-7B 混合精度量化项目

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Platform-macOS%20|%20Linux%20|%20Windows-lightgrey.svg" alt="Platform">
</p>

基于**遗传算法优化**的混合精度后训练量化（Mixed-Precision PTQ）框架，专为 **Qwen2.5-7B-Instruct** 大语言模型设计。

## 📖 目录

| 章节 | 内容 |
|------|------|
| [项目概述](#-项目概述) | 项目功能与核心特性 |
| [快速开始](#-快速开始) | 5分钟上手指南 |
| [项目结构](#-项目结构) | 代码模块说明 |
| [使用指南](#-使用指南) | 完整工作流程 |
| [技术原理](#-技术原理) | 算法与实现细节 |
| [常见问题](#-常见问题) | FAQ 与问题排查 |

---

## 🎯 项目概述

### 核心功能

| 步骤 | 说明 | 脚本 |
|------|------|------|
| 1️⃣ **敏感度分析** | 评估每层对量化的敏感程度 | `mixed_precision_ptq.py` |
| 2️⃣ **遗传算法搜索** | 全局优化逐层位宽配置 (W2/W4 + A4/A8) | `mixed_precision_ptq.py` |
| 3️⃣ **GGUF 导出** | 生成 llama.cpp 兼容格式 | `export_gguf_official.py` |
| 4️⃣ **性能对比** | 三模型推理速度/质量对比 | `compare_real_quant.py` |

### 核心特性

| 特性 | 说明 |
|------|------|
| 🧬 **遗传算法** | 避免局部最优，全局搜索最佳配置 |
| 🎯 **混合精度** | 权重 W2/W4 + 激活 A4/A8 联合优化 |
| 🔧 **SmoothQuant** | 激活值平滑，减少量化误差 |
| 📦 **GGUF 兼容** | 完全兼容 llama.cpp 真实量化推理 |
| ⚡ **多设备** | 支持 CUDA / MPS (Apple Silicon) / CPU |

### 混合精度策略 (W2/W4 + A4/A8)

本项目采用**权重-激活联合量化**策略：

| 层敏感度 | 权重位宽 | 激活位宽 | 说明 |
|----------|----------|----------|------|
| **低敏感** | W2 (2-bit) | A8 (8-bit) | 低精度权重 + 高精度激活补偿 |
| **高敏感** | W4 (4-bit) | A4 (4-bit) | 高精度权重 + 低精度激活 |

> 💡 **设计原理**: W2 层使用 A8 来补偿低精度权重的信息损失；W4 层已有足够精度，可使用 A4 进一步压缩。

---

## ⚠️ 核心概念：模拟量化 vs 真实量化

> **理解这两个概念是使用本项目的前提！**

| 对比项 | 模拟量化 | 真实量化 |
|--------|----------|----------|
| **数据类型** | FP32（模拟精度损失） | INT4/INT8（真正低精度） |
| **推理速度** | ❌ 更慢（额外计算） | ✅ 快 5-10 倍 |
| **内存占用** | ❌ 无变化 | ✅ 减少 70-85% |
| **用途** | 配置搜索、精度评估 | 生产部署 |
| **本项目脚本** | `mixed_precision_ptq.py` | `compare_real_quant.py` |

### 📊 Apple M4 Max 实测性能

| 模型 | 推理速度 | 内存占用 | 压缩比 |
|------|----------|----------|--------|
| 原始 FP16 | 14.7 tok/s | ~14 GB | 1.0x |
| Q4_K_M (4-bit) | **68.5 tok/s** | 4.4 GB | 3.3x |
| 混合精度 W2/W4+A4/A8 | 53.5 tok/s | 8.5 GB | 1.7x |

---

## 🚀 快速开始

### 环境安装

```bash
# 1. 克隆并进入项目
git clone https://github.com/your-username/Qwen2.5-7B_Mixed_PTQ.git
cd Qwen2.5-7B_Mixed_PTQ

# 2. 创建虚拟环境
python3 -m venv .venv && source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装 llama-cpp-python（真实量化推理必需）
# macOS (Metal)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
# Linux/Windows (CUDA)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
```

### 快速体验（5分钟）

```bash
# 下载预量化模型
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
    Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir models

# 运行对比测试（跳过原始模型，节省内存）
python compare_real_quant.py --skip_original
```

---

## 📁 项目结构

```
├── 📄 核心模块
│   ├── quant_utils.py          # 量化函数库 (quantize_tensor, MixedPrecisionLinear)
│   ├── genetic_optim.py        # 遗传算法优化器 (MixedPrecisionGA)
│   └── data_utils.py           # 数据工具 (校准数据加载)
│
├── ⚙️ 主程序
│   ├── mixed_precision_ptq.py  # 混合精度配置搜索（模拟量化）
│   ├── export_gguf_official.py # GGUF 格式导出（真实量化）
│   └── compare_real_quant.py   # 三模型性能对比
│
├── 🧪 测试脚本
│   └── test_mixed_precision.py # 模拟量化推理测试
│
└── 📦 输出文件
    ├── mixed_precision_config.pt          # 量化配置
    └── models/qwen2.5-7b-mixed.gguf       # GGUF 模型
```

---

## 📖 使用指南

### 完整工作流

```bash
# Step 1: 运行混合精度搜索（约30-60分钟）
python mixed_precision_ptq.py --device mps --ga_gen 20 --target_compression 0.25

# Step 2: 导出 GGUF 格式（约5分钟）
python export_gguf_official.py --output models/qwen2.5-7b-mixed.gguf

# Step 3: 三模型对比测试
python compare_real_quant.py --max_tokens 200
```

### 命令行参数

#### `mixed_precision_ptq.py` - 配置搜索

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_id` | `Qwen/Qwen2.5-7B-Instruct` | 模型 ID |
| `--device` | 自动检测 | `cuda` / `mps` / `cpu` |
| `--ga_pop` | 30 | 种群大小（越大搜索越全面） |
| `--ga_gen` | 25 | 迭代代数（越多收敛越好） |
| `--target_compression` | 0.25 | 目标压缩比 |

#### `compare_real_quant.py` - 性能对比

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--skip_original` | False | 跳过原始模型（节省内存） |
| `--max_tokens` | 200 | 最大生成长度 |

#### `export_gguf_official.py` - GGUF 导出

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `mixed_precision_config.pt` | 量化配置文件 |
| `--output` | `models/qwen2.5-7b-mixed.gguf` | 输出路径 |

---

## 🔬 技术原理

### 混合精度策略 (W2/W4 + A4/A8)

不同层对量化的敏感度不同，采用**权重-激活联合量化**策略：

| 层类型 | 敏感度 | 权重位宽 | 激活位宽 | 说明 |
|--------|--------|----------|----------|------|
| Attention Q/K/V | 高 | W4 | A4 | 注意力层需保精度 |
| FFN Gate/Up | 中 | W2-W4 | A4-A8 | 根据敏感度分析决定 |
| FFN Down | 低 | W2 | A8 | 低精度权重用高精度激活补偿 |
| LayerNorm | 高 | FP32 | FP32 | 归一化保持高精度 |

**配对原则**：
- **W2 + A8**：低精度权重需要高精度激活来补偿信息损失
- **W4 + A4**：权重精度足够时可使用低精度激活进一步压缩

### 遗传算法优化

```
初始化种群 → 适应度评估 → 选择 → 交叉 → 变异 → 迭代
     ↑                                           │
     └───────────── 直到收敛 ←────────────────────┘

染色体 = [layer0_w_bits, layer0_a_bits, layer1_w_bits, ...]
适应度 = -MSE（量化误差越小，适应度越高）
约束 = 压缩比 ≤ 目标值
```

### SmoothQuant 平滑

将激活值量化难度转移到权重：

```python
s = (max|X|^α) / (max|W|^(1-α))  # α=0.5
X' = X / s   # 激活值缩小
W' = W * s   # 权重放大
Y = X' @ W'  # 结果不变，但都更易量化
```

---

## ❓ 常见问题

### Q: 模拟量化后推理更慢了？

**A:** 正常！模拟量化只模拟精度损失，不会加速。想要加速请使用：
```bash
python compare_real_quant.py
```

### Q: MPS 设备报错？

**A:** 确保使用 `torch.float32`，并重新编译 llama-cpp-python：
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall
```

### Q: 量化后输出质量下降？

**A:** 调高目标压缩比（减少 W2 层）：
```bash
python mixed_precision_ptq.py --target_compression 0.35
```

### Q: 输出句子不完整？

**A:** 增加生成长度：
```bash
python compare_real_quant.py --max_tokens 500
```

---

## 💻 硬件要求

| 设备 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CUDA GPU | 16GB VRAM | 24GB+ (A100/4090) |
| Apple Silicon | M1 16GB | M2 Pro 32GB+ |
| CPU | 32GB RAM | 64GB+ |

---

## 📚 参考文献

- [SmoothQuant](https://arxiv.org/abs/2211.10438) - 激活值平滑量化
- [GPTQ](https://arxiv.org/abs/2210.17323) - 后训练量化
- [AWQ](https://arxiv.org/abs/2306.00978) - 激活感知量化
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - GGUF 推理框架

---

## 📄 License

MIT License © Jiangsheng Yu
