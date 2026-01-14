# Qwen2.5-7B 混合精度量化 (Mixed-Precision PTQ)

基于遗传算法优化的混合精度后训练量化框架，针对 Qwen2.5-7B-Instruct 大语言模型。

## ⚠️ 重要概念：模拟量化 vs 真实量化

在使用本项目之前，请务必理解这两种量化方式的区别：

### 🔬 模拟量化 (Simulated Quantization)

**用途**：验证量化精度、搜索最优位宽配置

```
┌─────────────────────────────────────────────────────────────┐
│  模拟量化流程 (本项目的 mixed_precision_ptq.py)              │
├─────────────────────────────────────────────────────────────┤
│  FP32权重 → 量化(round/clamp) → 反量化 → FP32权重(有损失)   │
│                                                             │
│  计算仍然是 FP32，只是模拟了低精度带来的精度损失            │
│  ❌ 不会加速，反而因为额外操作而变慢                         │
│  ✅ 用于评估量化对模型精度的影响                             │
└─────────────────────────────────────────────────────────────┘
```

### 🚀 真实量化 (Real Quantization)

**用途**：生产部署、实际加速推理

```
┌─────────────────────────────────────────────────────────────┐
│  真实量化流程 (llama.cpp / bitsandbytes / TensorRT)         │
├─────────────────────────────────────────────────────────────┤
│  FP32权重 → 转换为INT4/INT8 → 存储为GGUF/GPTQ格式           │
│                                                             │
│  计算直接使用低精度整数运算 (硬件加速)                       │
│  ✅ 真正加速推理 (5-10倍)                                    │
│  ✅ 大幅减少内存占用 (70-85%)                                │
└─────────────────────────────────────────────────────────────┘
```

### 📊 性能对比实测

| 方式 | 原始模型 (FP32) | 模拟量化 | 真实量化 (Q4_K_M) |
|------|-----------------|----------|-------------------|
| **推理速度** | 14.7 tok/s | 0.8 tok/s ❌ | **76.1 tok/s** ✅ |
| **内存占用** | 30.5 GB | 30.5 GB | **4.7 GB** ✅ |
| **加速比** | 1.0x | 0.05x | **5.6x** |
| **用途** | 基准 | 精度验证 | **生产部署** |

> 💡 **结论**：想要真正的加速效果，必须使用真实量化！

---

## 🎯 项目特点

- **智能混合精度**：自动识别每层的量化敏感度，敏感层使用高精度，非敏感层激进压缩
- **遗传算法优化**：全局搜索最优的逐层位宽配置
- **SmoothQuant技术**：通过激活值平滑减少量化误差
- **完整工作流**：从配置搜索到真实量化推理的端到端方案
- **多设备支持**：CUDA / MPS (Apple Silicon) / CPU

---

## 🚀 快速开始

### 环境安装

```bash
# 克隆项目
git clone https://github.com/yujiangsheng/Qwen2.5-7B_W2A8_Mixed_PTQ.git
cd Qwen2.5-7B_W2A8_Mixed_PTQ

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# (可选) 安装 llama-cpp-python 用于真实量化推理
# macOS (Metal 加速)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# Linux/Windows (CUDA 加速)
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
```

### 使用流程

本项目提供两条使用路径：

#### 路径 A：完整流程（搜索最优配置 + 真实量化）

```bash
# 第1步：运行混合精度搜索（模拟量化，用于找最优配置）
python mixed_precision_ptq.py

# 第2步：下载真实量化模型进行推理
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
    Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir models

# 第3步：运行真实量化推理对比
python compare_real_quant.py
```

#### 路径 B：快速体验（直接使用真实量化）

```bash
# 下载 GGUF 量化模型
huggingface-cli download bartowski/Qwen2.5-7B-Instruct-GGUF \
    Qwen2.5-7B-Instruct-Q4_K_M.gguf --local-dir models

# 运行真实量化对比测试（推荐！）
python compare_real_quant.py
```

---

## 📁 项目结构

```
Qwen2.5-7B_W2A8_Mixed_PTQ/
│
├── README.md                    # 项目说明文档（你正在阅读）
├── requirements.txt             # Python 依赖包
│
├── 【核心模块】
├── quant_utils.py              # 量化核心函数
│   ├── quantize_tensor()       # 模拟量化函数（分组对称/非对称）
│   └── MixedPrecisionLinear    # 混合精度线性层（替换 nn.Linear）
│
├── genetic_optim.py            # 遗传算法优化器
│   ├── MixedPrecisionGA        # 遗传算法主类
│   └── LayerSensitivityAnalyzer # 层敏感度分析器
│
├── data_utils.py               # 数据工具
│   ├── get_calib_dataset()     # 加载校准数据集
│   └── create_mock_input()     # 创建模拟输入
│
├── 【主程序】
├── mixed_precision_ptq.py      # 主量化程序（模拟量化，搜索配置）
│
├── 【测试脚本】
├── test_mixed_precision.py     # 模拟量化推理测试
├── compare_models.py           # 模拟量化 vs 原始模型对比
├── compare_real_quant.py       # 真实量化 vs 原始模型对比（推荐！）
│
├── 【输出文件】
├── mixed_precision_config.pt   # 量化配置文件（每层的位宽设置）
└── models/                     # GGUF 量化模型存放目录
```

---

## 🔧 命令行参数

### mixed_precision_ptq.py（主量化程序）

```bash
python mixed_precision_ptq.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \  # HuggingFace 模型 ID
    --device mps \                          # 计算设备: cuda, mps, cpu
    --n_layers 196 \                        # 量化层数
    --ga_pop 20 \                           # 遗传算法种群大小
    --ga_gen 12 \                           # 遗传算法迭代代数
    --target_compression 0.25 \             # 目标压缩比 (25%)
    --output mixed_precision_config.pt      # 输出配置文件
```

### compare_real_quant.py（真实量化对比）

```bash
python compare_real_quant.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \   # 原始模型 ID
    --gguf_path models/Qwen2.5-7B-Instruct-Q4_K_M.gguf \  # GGUF 模型路径
    --max_tokens 100                         # 最大生成 token 数
```

### test_mixed_precision.py（模拟量化测试）

```bash
python test_mixed_precision.py \
    --config mixed_precision_config.pt \    # 量化配置文件
    --prompt "你好，请介绍一下自己" \        # 自定义测试提示
    --max_tokens 100                         # 最大生成 token 数
```

---

## 📖 技术原理

### 1. 混合精度量化策略

不同层对量化的敏感度不同，我们为每层选择最合适的位宽：

| 层类型 | 敏感度 | 位宽 | 说明 |
|--------|--------|------|------|
| Attention Q/K 投影 | 高 | W8 | 对精度影响大，保守量化 |
| FFN 中间层 | 中 | W4 | 适中压缩 |
| Embedding 层 | 低 | W2 | 可激进压缩 |

### 2. 遗传算法优化

```
┌────────────────────────────────────────────────────────────┐
│  遗传算法搜索最优位宽配置                                   │
├────────────────────────────────────────────────────────────┤
│  1. 初始化：随机生成 N 个位宽配置（染色体）                 │
│  2. 适应度：计算每个配置的总 MSE（越小越好）               │
│  3. 选择：保留 Top 50% 的优秀个体                          │
│  4. 交叉：单点交叉产生子代                                 │
│  5. 变异：随机改变部分基因（位宽）                         │
│  6. 迭代：重复 2-5 直到收敛                                │
└────────────────────────────────────────────────────────────┘
```

### 3. SmoothQuant 技术

将激活值的量化难度转移到权重，使两者更容易量化：

```
原理: x' = x / s,  W' = W * s
其中: s = (max|x|^α) / (max|W|^(1-α))
α = 0.5 时，激活值和权重的量化难度平衡
```

### 4. 敏感度分析

根据量化前后的 MSE 误差判断敏感度：

| MSE 范围 | 敏感度 | 建议位宽 |
|----------|--------|----------|
| < 0.1 | 低 | W2 (2-bit) |
| 0.1 ~ 0.5 | 中 | W4 (4-bit) |
| > 0.5 | 高 | W8 (8-bit) |

---

## 💻 硬件要求

| 设备 | 最低要求 | 建议配置 | 备注 |
|------|----------|----------|------|
| CUDA GPU | 16GB VRAM | 24GB+ VRAM | 推荐 A100/4090 |
| Apple Silicon | M1 16GB | M2 Pro 32GB+ | 使用 Metal 加速 |
| CPU | 32GB RAM | 64GB+ RAM | 速度较慢 |

---

## 📝 输出文件说明

### mixed_precision_config.pt

量化配置文件，包含每层的详细配置：

```python
{
    "model.layers.0.self_attn.q_proj": {
        "w_bits": 4,        # 权重位数 (2/4/8)
        "a_bits": 8,        # 激活位数 (固定为8)
        "clip_ratio": 0.9,  # 权重裁剪比例
        "smooth_alpha": 0.5 # SmoothQuant 参数
    },
    "model.layers.0.self_attn.k_proj": { ... },
    # ... 共 196 层
}
```

---

## 🔍 常见问题

### Q: 为什么模拟量化后推理更慢了？

**A**: 这是正常的！模拟量化在 FP32 基础上增加了额外的 scale/clamp/round 操作，只是模拟精度损失，并不会加速。想要加速，必须使用**真实量化**（如 llama.cpp + GGUF）。

### Q: 如何获得真正的加速效果？

**A**: 运行真实量化对比测试：
```bash
python compare_real_quant.py
```

### Q: MPS 设备报错怎么办？

**A**: 
1. 确保使用 `torch.float32` 精度
2. 更新到最新版 PyTorch
3. 对于 llama.cpp，确保编译时启用 Metal：
   ```bash
   CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --force-reinstall
   ```

### Q: 量化后输出乱码？

**A**: W2 层过多可能导致精度损失。调整参数：
```bash
python mixed_precision_ptq.py --target_compression 0.35
```

---

## 📚 参考文献

- [SmoothQuant: Accurate and Efficient Post-Training Quantization](https://arxiv.org/abs/2211.10438)
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
- [Qwen2.5 Technical Report](https://github.com/QwenLM/Qwen2.5)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

---

## 👤 作者

**Jiangsheng Yu** - 作者 & 维护者

- GitHub: [@yujiangsheng](https://github.com/yujiangsheng)

---

## 📄 License

MIT License
