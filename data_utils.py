"""
校准数据工具模块 (Calibration Data Utilities)
=============================================

本模块负责加载和准备 PTQ 量化所需的校准数据集。

核心功能:
---------
- get_calib_dataset(): 加载校准数据集
- get_batch(): 批量数据生成器
- create_mock_input(): 创建模拟输入

校准数据的作用:
---------------
1. 收集激活值分布统计信息
2. 计算 SmoothQuant 缩放因子
3. 评估层敏感度

使用示例:
---------
>>> from data_utils import get_calib_dataset, create_mock_input
>>> 
>>> # 加载 WikiText-2 校准数据
>>> dataset = get_calib_dataset(n_samples=128, seq_len=2048)
>>> 
>>> # 创建模拟输入用于敏感度分析
>>> mock_input = create_mock_input(layer, batch_size=1, seq_len=128)
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import random
from typing import List, Optional, Generator


# ============================================================================
# 校准数据加载
# ============================================================================

def get_calib_dataset(
    data_path: Optional[str] = None,
    tokenizer_path: str = "Qwen/Qwen2.5-7B-Instruct",
    n_samples: int = 512,
    seq_len: int = 2048,
    seed: int = 42
) -> List[torch.Tensor]:
    """
    加载并准备校准数据集
    
    参数:
    -----
    data_path : str, optional
        本地数据文件路径（支持 .json/.jsonl/.txt）
        如果为 None，则使用 WikiText-2 数据集
    tokenizer_path : str
        分词器路径或 HuggingFace 模型 ID
    n_samples : int
        校准样本数量（建议 64-512）
    seq_len : int
        序列最大长度
    seed : int
        随机种子
    
    返回:
    ------
    List[torch.Tensor]
        校准数据列表，每个元素 shape: [1, seq_len]
    
    示例:
    ------
    >>> dataset = get_calib_dataset(n_samples=128)
    >>> print(f"加载了 {len(dataset)} 个样本")
    """
    # 设置随机种子确保可重复性
    random.seed(seed)
    
    # 加载分词器
    print(f"加载分词器: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # 加载数据
    if data_path:
        print(f"加载本地数据: {data_path}")
        data = _load_local_data(data_path)
        text_column = 'text'
    else:
        print("加载WikiText-2数据集...")
        data = load_dataset('wikitext', 'wikitext-2-v1', split='train')
        text_column = 'text'
    
    # 过滤过短的样本（至少50个字符）
    data = data.filter(lambda x: len(x[text_column]) > 50)
    print(f"过滤后剩余 {len(data)} 条数据")
    
    # 随机采样
    if len(data) > n_samples:
        indices = random.sample(range(len(data)), n_samples)
        data = data.select(indices)
    
    # 分词
    dataset = []
    print(f"正在分词 {len(data)} 个样本...")
    
    for example in data:
        text = example[text_column]
        encodings = tokenizer(
            text,
            return_tensors='pt',
            max_length=seq_len,
            truncation=True,
            padding='max_length'
        )
        
        # 过滤过短的序列（至少32个token）
        if encodings.input_ids.shape[1] < 32:
            continue
            
        dataset.append(encodings.input_ids)
    
    print(f"✓ 准备了 {len(dataset)} 个校准样本")
    return dataset


def _load_local_data(data_path: str):
    """
    加载本地数据文件
    
    参数：
    -----
    data_path : str
        本地文件路径
    
    返回：
    ------
    Dataset
        HuggingFace Dataset对象
    """
    if data_path.endswith('.json') or data_path.endswith('.jsonl'):
        return load_dataset('json', data_files=data_path, split='train')
    else:
        return load_dataset('text', data_files=data_path, split='train')


def get_batch(dataset: List[torch.Tensor], batch_size: int = 1) -> Generator:
    """
    将数据集按batch分批返回
    
    参数：
    -----
    dataset : List[torch.Tensor]
        校准数据列表
    batch_size : int
        批次大小，默认1
    
    返回：
    ------
    Generator
        batch数据的生成器
    
    示例：
    ------
    >>> dataset = get_calib_dataset(n_samples=128)
    >>> for batch in get_batch(dataset, batch_size=8):
    ...     # batch shape: [batch_size, seq_len]
    ...     process(batch)
    """
    for i in range(0, len(dataset), batch_size):
        yield torch.cat(dataset[i:i + batch_size], dim=0)


def create_mock_input(layer, batch_size: int = 1, seq_len: int = 128, 
                      device: str = 'cpu', dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    为指定层创建模拟输入
    
    用于层敏感度分析，无需实际运行完整模型
    
    参数：
    -----
    layer : nn.Linear
        目标线性层
    batch_size : int
        批次大小
    seq_len : int
        序列长度
    device : str
        设备 ('cpu', 'cuda', 'mps')
    dtype : torch.dtype
        数据类型
    
    返回：
    ------
    torch.Tensor
        模拟输入张量，shape: [batch_size, seq_len, in_features]
    """
    return torch.randn(
        batch_size, seq_len, layer.in_features,
        device=device, dtype=dtype
    )
