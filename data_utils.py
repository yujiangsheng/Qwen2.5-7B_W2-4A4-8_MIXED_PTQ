"""
校准数据工具 (Calibration Data Utilities)
========================================

功能:
  - get_calib_dataset(): 加载校准数据集 (WikiText-2 或自定义)
  - create_mock_input(): 为指定层创建模拟输入
  - get_batch(): 将数据集按 batch 分批

用途:
  - 收集激活值分布统计
  - 评估量化误差和层敏感度
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import random
from typing import List, Optional, Generator


def get_calib_dataset(
    data_path: Optional[str] = None,
    tokenizer_path: str = "Qwen/Qwen2.5-7B-Instruct",
    n_samples: int = 512,
    seq_len: int = 2048,
    seed: int = 42
) -> List[torch.Tensor]:
    """
    加载校准数据集
    
    Args:
        data_path: 本地数据文件 (.json/.jsonl/.txt)，None 则使用 WikiText-2
        tokenizer_path: 分词器路径
        n_samples: 样本数量
        seq_len: 序列长度
        seed: 随机种子
    
    Returns:
        List[torch.Tensor]: 校准数据列表，每个元素 shape=[1, seq_len]
    
    Example:
        >>> dataset = get_calib_dataset(n_samples=128)
        >>> print(f"加载了 {len(dataset)} 个样本")
    """
    random.seed(seed)
    
    # 加载分词器
    print(f"📝 加载分词器: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # 加载数据
    if data_path:
        print(f"📂 加载本地数据: {data_path}")
        if data_path.endswith(('.json', '.jsonl')):
            data = load_dataset('json', data_files=data_path, split='train')
        else:
            data = load_dataset('text', data_files=data_path, split='train')
    else:
        print("📂 加载 WikiText-2...")
        data = load_dataset('wikitext', 'wikitext-2-v1', split='train')
    
    text_column = 'text'
    
    # 过滤短样本
    data = data.filter(lambda x: len(x[text_column]) > 50)
    print(f"   过滤后: {len(data)} 条")
    
    # 随机采样
    if len(data) > n_samples:
        indices = random.sample(range(len(data)), n_samples)
        data = data.select(indices)
    
    # 分词
    dataset = []
    for example in data:
        encodings = tokenizer(
            example[text_column],
            return_tensors='pt',
            max_length=seq_len,
            truncation=True,
            padding='max_length'
        )
        if encodings.input_ids.shape[1] >= 32:
            dataset.append(encodings.input_ids)
    
    print(f"✅ 准备了 {len(dataset)} 个校准样本")
    return dataset


def get_batch(dataset: List[torch.Tensor], batch_size: int = 1) -> Generator:
    """
    按 batch 分批返回数据
    
    Args:
        dataset: 校准数据列表
        batch_size: 批次大小
    
    Yields:
        torch.Tensor: shape=[batch_size, seq_len]
    """
    for i in range(0, len(dataset), batch_size):
        yield torch.cat(dataset[i:i + batch_size], dim=0)


def create_mock_input(
    layer, 
    batch_size: int = 1, 
    seq_len: int = 128,
    device: str = 'cpu', 
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    为指定层创建模拟输入（用于敏感度分析）
    
    Args:
        layer: nn.Linear 层
        batch_size: 批次大小
        seq_len: 序列长度
        device: 设备
        dtype: 数据类型
    
    Returns:
        torch.Tensor: shape=[batch_size, seq_len, in_features]
    """
    return torch.randn(
        batch_size, seq_len, layer.in_features,
        device=device, dtype=dtype
    )
