"""
遗传算法优化器 (Genetic Algorithm Optimizer)
==========================================

用于搜索最优的混合精度配置 (W4 + A4/A8)。

核心思想:
  - 将每层的激活位宽 (4/8) 视为基因
  - 通过进化选择找到最优配置
  - 目标: 最小化量化误差 + 满足压缩率约束

主要类:
  - MixedPrecisionGA: 遗传算法主类
  - LayerSensitivityAnalyzer: 层敏感度分析器
"""

import numpy as np
from typing import Callable, Dict, List, Optional
import torch


class MixedPrecisionGA:
    """
    混合精度遗传算法优化器
    
    搜索最优的逐层激活位宽配置 (A4/A8)，权重固定使用 W4。
    
    Args:
        n_layers: 待优化的层数 (Qwen2.5-7B: 196)
        population_size: 种群大小
        n_generations: 迭代代数
        mutation_rate: 初始变异率
        elite_ratio: 精英保留比例
        adaptive_mutation: 是否启用自适应变异
    
    Example:
        >>> ga = MixedPrecisionGA(n_layers=196, population_size=30)
        >>> ga.set_layer_sensitivities(sensitivities, layer_names)
        >>> best_config = ga.optimize(fitness_func, target_compression=0.75)
    """
    
    def __init__(
        self, 
        n_layers: int, 
        population_size: int = 30,
        n_generations: int = 25, 
        mutation_rate: float = 0.12,
        elite_ratio: float = 0.15, 
        adaptive_mutation: bool = True
    ):
        self.n_layers = n_layers
        self.pop_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.initial_mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.adaptive_mutation = adaptive_mutation
        
        # 激活位宽选项: A4(低敏感度), A8(高敏感度)
        self.bit_options = [4, 8]
        
        # 层敏感度信息（由外部设置）
        self.layer_weights: Optional[np.ndarray] = None
        self.layer_sensitivity_ratios: Optional[np.ndarray] = None
        
        # 收敛追踪
        self.stagnation_counter = 0
        self.best_score_history = []
    
    def set_layer_sensitivities(
        self, 
        sensitivities: Dict[str, Dict[int, float]],
        layer_names: List[str]
    ):
        """
        设置层敏感度信息，用于智能初始化和加权变异
        
        Args:
            sensitivities: {layer_name: {4: mse_a4, 8: mse_a8}}
            layer_names: 按顺序排列的层名列表
        """
        n = len(layer_names)
        self.layer_weights = np.ones(n)
        self.layer_sensitivity_ratios = np.zeros(n)
        
        # 计算敏感度比例 (A4_MSE / A8_MSE)
        ratios = []
        for i, name in enumerate(layer_names):
            sens = sensitivities.get(name, {4: 0.1, 8: 0.01})
            ratio = sens.get(4, 0.1) / max(sens.get(8, 0.01), 1e-8)
            ratios.append(ratio)
            self.layer_sensitivity_ratios[i] = ratio
        
        # 对数变换归一化到 [0.5, 2.0]
        ratios = np.array(ratios)
        log_ratios = np.log1p(ratios)
        if log_ratios.max() > log_ratios.min():
            normalized = (log_ratios - log_ratios.min()) / (log_ratios.max() - log_ratios.min())
            self.layer_weights = 0.5 + 1.5 * normalized
        
        print(f"  敏感度比例: {ratios.min():.2f} - {ratios.max():.2f}")
    
    def initialize_population(self) -> List[np.ndarray]:
        """智能初始化种群（基于敏感度信息）"""
        population = []
        
        if self.layer_sensitivity_ratios is not None:
            median = np.median(self.layer_sensitivity_ratios)
            q25, q75 = np.percentile(self.layer_sensitivity_ratios, [25, 75])
            
            for i in range(self.pop_size):
                individual = np.zeros(self.n_layers, dtype=int)
                
                # 使用不同阈值创建多样性
                if i < self.pop_size // 4:
                    threshold = q25  # 激进压缩
                elif i < self.pop_size // 2:
                    threshold = median  # 平衡
                elif i < 3 * self.pop_size // 4:
                    threshold = q75  # 保守
                else:
                    threshold = np.random.uniform(q25, q75)  # 随机
                
                for j in range(self.n_layers):
                    ratio = self.layer_sensitivity_ratios[j]
                    noise = 1 + np.random.uniform(-0.2, 0.2)
                    individual[j] = 8 if ratio > threshold * noise else 4
                
                population.append(individual)
        else:
            # 无敏感度信息时随机初始化（略偏向 A4）
            for _ in range(self.pop_size):
                individual = np.random.choice([4, 8], size=self.n_layers, p=[0.55, 0.45])
                population.append(individual)
        
        return population
    
    def compute_model_size(self, individual: np.ndarray) -> float:
        """计算相对模型大小（相对于 W4A8 基准）"""
        avg_a_bits = np.mean(individual)
        return (4 + avg_a_bits) / (4 + 8)
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """交叉操作（两点交叉/均匀交叉）"""
        if np.random.rand() < 0.6:  # 两点交叉
            points = sorted(np.random.choice(range(1, self.n_layers), 2, replace=False))
            return np.concatenate([
                parent1[:points[0]], 
                parent2[points[0]:points[1]], 
                parent1[points[1]:]
            ])
        else:  # 均匀交叉
            mask = np.random.rand(self.n_layers) < 0.5
            return np.where(mask, parent1, parent2)
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """智能变异（高敏感度层倾向 A8）"""
        median_ratio = np.median(self.layer_sensitivity_ratios) if self.layer_sensitivity_ratios is not None else 0
        
        for i in range(self.n_layers):
            # 高权重层变异概率稍低
            mut_rate = self.mutation_rate
            if self.layer_weights is not None:
                mut_rate *= (2.0 - self.layer_weights[i]) / 1.5
            
            if np.random.rand() < mut_rate:
                if self.layer_sensitivity_ratios is not None:
                    ratio = self.layer_sensitivity_ratios[i]
                    # 高敏感度倾向 A8，低敏感度倾向 A4
                    p_a8 = 0.7 if ratio > median_ratio else 0.3
                    individual[i] = np.random.choice([4, 8], p=[1-p_a8, p_a8])
                else:
                    individual[i] = np.random.choice([4, 8])
        
        return individual
    
    def update_mutation_rate(self, improved: bool):
        """自适应调整变异率"""
        if not self.adaptive_mutation:
            return
        
        if improved:
            self.stagnation_counter = 0
            self.mutation_rate = max(0.05, self.mutation_rate * 0.95)
        else:
            self.stagnation_counter += 1
            if self.stagnation_counter >= 3:
                self.mutation_rate = min(0.3, self.mutation_rate * 1.5)
                self.stagnation_counter = 0
    
    def optimize(
        self, 
        fitness_func: Callable[[np.ndarray], float],
        target_compression: float = 0.75
    ) -> np.ndarray:
        """
        执行遗传算法优化
        
        Args:
            fitness_func: 适应度函数 (配置 -> 分数，越高越好)
            target_compression: 目标压缩比（相对于 W4A8）
        
        Returns:
            最优激活位宽配置
        """
        population = self.initialize_population()
        best_individual = None
        best_score = -float('inf')
        no_improve_count = 0
        n_elite = max(2, int(self.pop_size * self.elite_ratio))
        
        for gen in range(self.n_generations):
            # 计算适应度
            scores = []
            for indiv in population:
                quality = fitness_func(indiv)
                size_ratio = self.compute_model_size(indiv)
                
                # 大小惩罚
                if size_ratio > target_compression:
                    penalty = ((size_ratio - target_compression) * 15) ** 1.5
                else:
                    penalty = -((target_compression - size_ratio) * 2)
                
                scores.append(quality - penalty)
            
            scores = np.array(scores)
            sorted_idx = np.argsort(scores)[::-1]
            
            # 检查改进
            current_best = scores[sorted_idx[0]]
            improved = current_best > best_score
            
            if improved:
                best_score = current_best
                best_individual = population[sorted_idx[0]].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            self.update_mutation_rate(improved)
            
            # 早停
            if no_improve_count >= 10:
                print(f"  早停: 连续{no_improve_count}代无改进")
                break
            
            # 精英保留
            elite = [population[i].copy() for i in sorted_idx[:n_elite]]
            
            # 锦标赛选择
            def tournament_select(k=3):
                candidates = np.random.choice(len(population), k, replace=False)
                winner = candidates[np.argmax([scores[c] for c in candidates])]
                return population[winner]
            
            # 生成新种群
            new_pop = elite[:]
            while len(new_pop) < self.pop_size:
                child = self.crossover(tournament_select(), tournament_select())
                child = self.mutate(child)
                new_pop.append(child)
            
            population = new_pop
            
            # 打印进度
            a4_count = np.sum(best_individual == 4)
            a8_count = np.sum(best_individual == 8)
            size = self.compute_model_size(best_individual)
            mut = "↑" if self.mutation_rate > self.initial_mutation_rate else "↓"
            print(f"第{gen+1}代: 分数={best_score:.4f}, 大小={size:.1%}, "
                  f"A4={a4_count}, A8={a8_count}, 变异率={self.mutation_rate:.3f}{mut}")
        
        print(f"\n  最终变异率: {self.mutation_rate:.3f}")
        return best_individual


class LayerSensitivityAnalyzer:
    """
    层敏感度分析器
    
    评估每层对不同激活量化位宽的敏感程度（权重固定 W4）。
    
    敏感度 = MSE(量化输出, 原始输出)
    敏感度比例 = MSE_A4 / MSE_A8
    """
    
    def __init__(self, bit_options: List[int] = None):
        self.bit_options = bit_options or [4, 8]
    
    def analyze(
        self, 
        layer, 
        calib_input: torch.Tensor, 
        quantize_fn: Callable
    ) -> Dict[int, float]:
        """
        分析单层敏感度
        
        Args:
            layer: 线性层
            calib_input: 校准输入
            quantize_fn: 量化函数
        
        Returns:
            {4: mse_a4, 8: mse_a8}
        """
        original_output = layer(calib_input)
        sensitivity = {}
        
        # 固定 W4 权重量化
        w = layer.weight
        limit = w.abs().amax() * 0.9
        w_clipped = torch.clamp(w, -limit, limit)
        w_q = quantize_fn(w_clipped, n_bits=4, group_size=128, sym=True)
        
        for a_bits in self.bit_options:
            x_q = quantize_fn(calib_input, n_bits=a_bits, group_size=-1, sym=False)
            out_q = torch.nn.functional.linear(x_q, w_q, layer.bias)
            mse = torch.mean((out_q - original_output) ** 2).item()
            sensitivity[a_bits] = mse
        
        return sensitivity
    
    def classify(
        self, 
        mse_a4: float, 
        mse_a8: float, 
        is_edge_layer: bool = False
    ) -> str:
        """
        根据敏感度比例分类
        
        Args:
            mse_a4: A4 量化 MSE
            mse_a8: A8 量化 MSE
            is_edge_layer: 是否是首尾层
        
        Returns:
            分类描述
        """
        ratio = mse_a4 / max(mse_a8, 1e-8)
        threshold = 1.5 if is_edge_layer else 2.5
        
        if ratio > threshold:
            return f"高敏感度(A8) [比例:{ratio:.1f}]"
        else:
            return f"低敏感度(A4) [比例:{ratio:.1f}]"
