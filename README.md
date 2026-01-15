# 🚀 Qwen2.5-7B 混合精度量化 (W4 + A4/A8)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Platform-macOS%20|%20Linux%20|%20Windows-lightgrey.svg" alt="Platform">
</p>

<p align="center">
  <b>基于遗传算法的混合精度后训练量化框架</b><br>
  专为 Qwen2.5-7B-Instruct 大语言模型设计
</p>

---

## 📖 目录

- [项目概述](#-项目概述)
- [核心特性](#-核心特性)
- [量化策略详解](#-量化策略详解)
- [快速开始](#-快速开始)
- [完整工作流程](#-完整工作流程)
- [项目结构](#-项目结构)
- [命令行参数](#-命令行参数)
- [技术原理](#-技术原理)
- [性能对比](#-性能对比)
- [常见问题](#-常见问题)
- [参考文献](#-参考文献)

---

## 🎯 项目概述

本项目实现了一个完整的混合精度量化工作流，通过遗传算法为 LLM 的每一层自动搜索最优量化配置。

### 工作流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    混合精度量化完整工作流                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ 1.加载   │───▶│ 2.敏感度 │───▶│ 3.遗传   │───▶│ 4.导出   │      │
│  │ 预训练   │    │ 分析     │    │ 算法优化 │    │ GGUF     │      │
│  │ 模型     │    │ (A4/A8)  │    │ 搜索配置 │    │ 格式     │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│       ↓               ↓               ↓               ↓            │
│  Qwen2.5-7B     每层MSE对比      最优位宽配置     真实量化模型      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## ✨ 核心特性

| 特性 | 说明 |
|------|------|
| 🧬 **遗传算法优化** | 全局搜索最优配置，避免局部最优陷阱 |
| 🎯 **W4 + A4/A8 混合精度** | 权重固定4位，激活值按敏感度选择4/8位 |
| 🔧 **SmoothQuant 技术** | 激活值平滑，减少量化误差 |
| 📦 **GGUF 格式导出** | 完全兼容 llama.cpp 推理 |
| ⚡ **多设备支持** | CUDA / MPS (Apple Silicon) / CPU |
| 📊 **完整评估工具** | 速度、质量、内存全面对比 |

---

## 🔬 量化策略详解

### W4 + A4/A8 混合精度策略

```
┌─────────────────────────────────────────────────────────────────┐
│  量化策略: W4 + A4/A8                                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【权重量化 - 固定 W4】                                          │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  所有线性层权重 → 4-bit 对称量化 (group_size=128)    │       │
│  │  • 压缩比: 4x (FP16 → INT4)                          │       │
│  │  • 使用分组量化保持精度                              │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
│  【激活量化 - 混合 A4/A8】                                       │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  低敏感度层 → A4 (4-bit 非对称量化)                  │       │
│  │  • 更激进的压缩                                      │       │
│  │  • 适用于 FFN 中间层等                               │       │
│  ├─────────────────────────────────────────────────────┤       │
│  │  高敏感度层 → A8 (8-bit 非对称量化)                  │       │
│  │  • 保持计算精度                                      │       │
│  │  • 适用于 Attention、首尾层等                        │       │
│  └─────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 敏感度判定标准

敏感度 = MSE(A4量化输出, 原始输出) / MSE(A8量化输出, 原始输出)

- **比例 > 2.5** → 高敏感度，使用 A8
- **比例 ≤ 2.5** → 低敏感度，使用 A4
- **首尾层特殊处理**：阈值降至 1.5

---

## ⚠️ 重要概念：模拟量化 vs 真实量化

```
┌─────────────────────────────────────────────────────────────────┐
│  模拟量化 (Simulated) - mixed_precision_ptq.py                   │
├─────────────────────────────────────────────────────────────────┤
│  FP32 权重 → 量化(round) → 反量化 → FP32 权重(有损失)            │
│                                                                 │
│  ❌ 不会加速（计算仍是FP32）    ✅ 用于评估量化影响               │
│  ❌ 不会省内存                  ✅ 用于搜索最优配置               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  真实量化 (Real) - llama.cpp / GGUF                              │
├─────────────────────────────────────────────────────────────────┤
│  FP32 权重 → 转换 INT4/INT8 → 直接低精度计算                    │
│                                                                 │
│  ✅ 推理加速 5-10x             ✅ 内存减少 70-85%                │
│  ✅ 适合生产部署               ✅ 硬件加速支持                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/your-username/Qwen2.5-7B_W4A4-8_MIXED_PTQ.git
cd Qwen2.5-7B_W4A4-8_MIXED_PTQ

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 安装 llama-cpp-python（真实量化推理）
# macOS (Metal)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# Linux/Windows (CUDA)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
```

### 2. 快速测试（使用预量化模型）

```bash
# 下载 Q4_K_M 模型
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
    Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir models

# 运行对比测试
python compare_real_quant.py --skip_original --max_tokens 200
```

---

## 📋 完整工作流程

### 流程图

```
┌────────────────────────────────────────────────────────────────────┐
│  Step 1: 敏感度分析 + 遗传算法优化 (约 30-60 分钟)                  │
├────────────────────────────────────────────────────────────────────┤
│  python mixed_precision_ptq.py --device mps --ga_gen 15            │
│                                                                    │
│  输出: mixed_precision_config.pt                                   │
│  内容: 每层的 w_bits=4, a_bits=4或8                                │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  Step 2: 导出 GGUF 格式 (约 5-10 分钟)                              │
├────────────────────────────────────────────────────────────────────┤
│  python export_gguf_official.py --output models/qwen2.5-7b-mixed.gguf│
│                                                                    │
│  输出: models/qwen2.5-7b-mixed.gguf                                │
│  大小: 约 7-8 GB                                                   │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  Step 3: 性能对比测试                                               │
├────────────────────────────────────────────────────────────────────┤
│  python compare_real_quant.py --max_tokens 200                     │
│                                                                    │
│  对比: 原始模型 vs Q4_K_M vs 混合精度                              │
└────────────────────────────────────────────────────────────────────┘
```

### 命令示例

```bash
# 完整流程（使用默认参数）
python mixed_precision_ptq.py
python export_gguf_official.py
python compare_real_quant.py

# 自定义参数
python mixed_precision_ptq.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --device mps \
    --ga_pop 25 \
    --ga_gen 20 \
    --target_compression 0.75 \
    --output my_config.pt
```

---

## 📁 项目结构

```
Qwen2.5-7B_W4A4-8_MIXED_PTQ/
│
├── 📄 README.md                 # 项目文档
├── 📄 requirements.txt          # 依赖包列表
│
├── 🔧 核心模块
│   ├── quant_utils.py           # 量化函数 + MixedPrecisionLinear
│   ├── genetic_optim.py         # 遗传算法优化器
│   └── data_utils.py            # 校准数据加载
│
├── ⚙️ 主程序
│   ├── mixed_precision_ptq.py   # 混合精度搜索主程序
│   └── export_gguf_official.py  # GGUF 格式导出
│
├── 🧪 测试脚本
│   ├── test_mixed_precision.py  # 模拟量化测试
│   └── compare_real_quant.py    # 真实量化对比
│
├── 📦 输出文件
│   ├── mixed_precision_config.pt   # 量化配置
│   └── models/
│       ├── Qwen2.5-7B-Instruct-Q4_K_M.gguf  # 基准模型
│       └── qwen2.5-7b-mixed.gguf            # 混合精度模型
```

---

## ⌨️ 命令行参数

### mixed_precision_ptq.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_id` | `Qwen/Qwen2.5-7B-Instruct` | 模型 ID |
| `--device` | 自动检测 | `cuda` / `mps` / `cpu` |
| `--ga_pop` | 30 | GA 种群大小 |
| `--ga_gen` | 25 | GA 迭代代数 |
| `--target_compression` | 0.75 | 目标压缩比 |
| `--output` | `mixed_precision_config.pt` | 输出路径 |

### compare_real_quant.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_tokens` | 200 | 最大生成 token 数 |
| `--skip_original` | False | 跳过原始模型（节省内存） |
| `--q4km_path` | `models/Qwen2.5-7B-Instruct-Q4_K_M.gguf` | Q4_K_M 路径 |
| `--mixed_path` | `models/qwen2.5-7b-mixed.gguf` | 混合精度模型路径 |

---

## 🔬 技术原理

### 1. 遗传算法搜索

```
┌──────────────────────────────────────────────────────────────┐
│  遗传算法流程                                                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1️⃣ 初始化: 生成 N 个随机激活位宽配置                         │
│      染色体 = [a0, a1, a2, ..., a195]  (每个 ai ∈ {4, 8})    │
│                                                              │
│  2️⃣ 适应度评估:                                              │
│      fitness = -Σ(MSE_i × weight_i)                          │
│      MSE 越小 → 适应度越高 → 个体越优秀                       │
│                                                              │
│  3️⃣ 选择: 锦标赛选择 + 精英保留                              │
│                                                              │
│  4️⃣ 交叉: 两点交叉 / 均匀交叉                                │
│      父代A: [4,4,8,8,4,8,...]                                │
│      父代B: [8,4,4,4,8,4,...]                                │
│                   ↓                                          │
│      子代:  [4,4,4,4,8,4,...]                                │
│                                                              │
│  5️⃣ 变异: 智能变异（高敏感度层倾向 A8）                       │
│                                                              │
│  6️⃣ 迭代: 重复 2-5 直到收敛                                  │
└──────────────────────────────────────────────────────────────┘
```

### 2. SmoothQuant 技术

```
原始计算:  Y = X @ W

SmoothQuant 变换:
  s = (max|X|^α) / (max|W|^(1-α))   # α=0.5 时平衡
  X' = X / s      # 激活值缩小，更易量化
  W' = W * s      # 权重放大，保持计算结果
  Y = X' @ W'     # 结果等价，但两者都更易量化
```

### 3. 分组量化

```
权重矩阵 [out_features × in_features]
        ↓
按 group_size=128 分组
        ↓
每组独立计算 scale: scale_i = max(|group_i|) / (2^(n-1) - 1)
        ↓
量化: q_i = round(x_i / scale_i)
反量化: x_i ≈ q_i × scale_i
```

---

## 📊 性能对比

### Apple M4 Max 测试结果

| 模型 | 大小 | 加载时间 | 推理速度 | 加速比 |
|------|------|----------|----------|--------|
| 原始 (FP32) | ~14.2 GB | ~3s | 14.7 tok/s | 1.0x |
| Q4_K_M | 4.36 GB | ~1s | **68.5 tok/s** | **4.7x** |
| 混合精度 | 7.5 GB | ~2s | 53.5 tok/s | 3.6x |

### 压缩效果

- **Q4_K_M**: 内存减少 **69%**，速度提升 **4.7x**
- **混合精度**: 内存减少 **47%**，速度提升 **3.6x**

---

## ❓ 常见问题

### Q1: 为什么模拟量化更慢？

模拟量化在 FP32 基础上增加了量化/反量化操作，只是模拟精度损失。真正加速需要使用 GGUF 格式的真实量化推理。

### Q2: 如何节省内存运行对比测试？

```bash
python compare_real_quant.py --skip_original
```

### Q3: MPS 设备报错怎么办？

1. 确保使用 `torch.float32`（MPS 对 FP16 支持有限）
2. 更新 PyTorch 到最新版本
3. llama.cpp 编译时启用 Metal：
   ```bash
   CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall
   ```

### Q4: 输出质量下降怎么办？

调高目标压缩比（减少 A4 层比例）：
```bash
python mixed_precision_ptq.py --target_compression 0.85
```

---

## 💻 硬件要求

| 设备 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CUDA GPU | 16GB VRAM | 24GB+ (A100/4090) |
| Apple Silicon | M1 16GB | M2 Pro 32GB+ |
| CPU | 32GB RAM | 64GB+ RAM |

---

## 📚 参考文献

- [SmoothQuant: Accurate and Efficient Post-Training Quantization](https://arxiv.org/abs/2211.10438)
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)
- [Qwen2.5 Technical Report](https://github.com/QwenLM/Qwen2.5)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

---

## 📄 License

MIT License

---

<p align="center">
  如果这个项目对你有帮助，请给个 ⭐ Star！
</p>
