# Qwen2.5-7B 混合精度量化 (Mixed-Precision PTQ)

基于遗传算法优化的混合精度后训练量化框架，针对 Qwen2.5-7B-Instruct 大语言模型。

## 🎯 项目特点

- **智能混合精度**：自动识别每层的量化敏感度，敏感层使用高精度，非敏感层激进压缩
- **遗传算法优化**：全局搜索最优的逐层位宽配置
- **SmoothQuant技术**：通过激活值平滑减少量化误差
- **多设备支持**：自动检测 CUDA / MPS (Apple Silicon) / CPU

## 📊 量化效果

| 指标 | 原始模型 | 量化后 |
|------|----------|--------|
| 权重精度 | FP16 | W2/W4/W8混合 |
| 激活精度 | FP16 | A8 |
| 压缩比 | 100% | ~31% |
| 内存节省 | - | ~69% |
| 推理质量 | 基准 | 接近原始 |

典型配置（196层）:
- W2层: 39个 (低敏感度层)
- W4层: 87个 (中敏感度层)  
- W8层: 70个 (高敏感度层)

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
cd Qwen2.5-7B_W2A8

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行量化

```bash
# 基本用法 (自动检测设备)
python mixed_precision_ptq.py

# 指定参数
python mixed_precision_ptq.py \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --device mps \
    --n_layers 196 \
    --ga_pop 20 \
    --ga_gen 12 \
    --target_compression 0.25
```

### 3. 测试推理

```bash
# 默认测试
python test_mixed_precision.py

# 自定义提示
python test_mixed_precision.py --prompt "解释什么是量子计算"
```

## 📁 项目结构

```
Qwen2.5-7B_W2A8/
├── README.md                    # 项目说明文档
├── requirements.txt             # Python依赖
├── quant_utils.py              # 量化核心函数
│   ├── quantize_tensor()       # 模拟量化函数
│   └── MixedPrecisionLinear    # 混合精度线性层
├── genetic_optim.py            # 遗传算法优化器
│   ├── MixedPrecisionGA        # GA优化类
│   └── LayerSensitivityAnalyzer # 敏感度分析器
├── data_utils.py               # 数据加载工具
│   ├── get_calib_dataset()     # 加载校准数据
│   └── create_mock_input()     # 创建模拟输入
├── mixed_precision_ptq.py      # 主量化程序
└── test_mixed_precision.py     # 推理测试脚本
```

## 🔧 核心参数

### mixed_precision_ptq.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_id` | Qwen/Qwen2.5-7B-Instruct | HuggingFace模型ID |
| `--device` | 自动检测 | 计算设备: cuda, mps, cpu |
| `--n_layers` | 196 | 量化层数 |
| `--ga_pop` | 20 | 遗传算法种群大小 |
| `--ga_gen` | 12 | 遗传算法迭代代数 |
| `--target_compression` | 0.25 | 目标压缩比 |
| `--output` | mixed_precision_config.pt | 输出配置文件 |

### test_mixed_precision.py

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | mixed_precision_config.pt | 量化配置文件 |
| `--prompt` | (预设测试) | 自定义测试提示 |
| `--max_tokens` | 100 | 最大生成token数 |

## 📖 技术原理

### 1. 量化方案

- **权重量化**: 分组对称量化 (group_size=128)
  - W2: 2-bit，用于低敏感度层
  - W4: 4-bit，用于中敏感度层  
  - W8: 8-bit，用于高敏感度层

- **激活量化**: Per-tensor非对称量化
  - A8: 固定8-bit，适用于所有层

### 2. SmoothQuant 平滑

```
原理: 将激活值的量化难度转移到权重
x' = x / s,  W' = W * s
其中 s = (max|x|^α) / (max|W|^(1-α))
```

### 3. 遗传算法流程

```
1. 初始化: 随机生成N个位宽配置
2. 适应度: 计算总MSE (越小越好)
3. 选择: 保留Top 50%个体
4. 交叉: 单点交叉产生子代
5. 变异: 随机改变部分基因
6. 迭代: 重复2-5直到收敛
```

### 4. 敏感度分类

| MSE范围 | 敏感度 | 建议位宽 |
|---------|--------|----------|
| < 0.1 | 低 | W2 |
| 0.1 ~ 0.5 | 中 | W4 |
| > 0.5 | 高 | W8 |

## 💻 硬件要求

| 设备 | 最低要求 | 建议配置 |
|------|----------|----------|
| CUDA GPU | 24GB VRAM | 32GB+ VRAM |
| Apple Silicon | M1 (16GB) | M2 Pro (32GB+) |
| CPU | 32GB RAM | 64GB+ RAM |

**注意**: 7B模型较大，建议使用GPU或大内存设备。MPS设备请使用float32精度。

## 📝 输出文件

量化完成后生成 `mixed_precision_config.pt`，包含每层的配置:

```python
{
    "model.layers.0.self_attn.q_proj": {
        "w_bits": 4,        # 权重位数
        "a_bits": 8,        # 激活位数
        "clip_ratio": 0.9,  # 裁剪比例
        "smooth_alpha": 0.5 # 平滑参数
    },
    ...
}
```

## 🔍 常见问题

### Q: 量化后模型输出乱码?
A: 可能是W2层过多。尝试调高 `--target_compression` 到 0.35 或更高。

### Q: MPS设备报错?
A: MPS对某些操作支持有限。确保使用float32精度，并安装最新版PyTorch。

### Q: 如何自定义校准数据?
A: 修改 `data_utils.py` 中的 `get_calib_dataset()` 函数，传入自定义数据路径。

## 📚 参考文献

- [SmoothQuant: Accurate and Efficient Post-Training Quantization](https://arxiv.org/abs/2211.10438)
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323)
- [Qwen2.5 Technical Report](https://github.com/QwenLM/Qwen2.5)

## � 作者

**Jiangsheng Yu** - 作者 & 维护者

- GitHub: [@yujiangsheng](https://github.com/yujiangsheng)

## �📄 License

MIT License
