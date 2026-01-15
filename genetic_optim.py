"""
遗传算法优化器 (Genetic Algorithm Optimizer)
==========================================

本模块实现用于混合精度量化配置优化的遗传算法。

核心思想:
---------
将每层的量化位数视为基因，通过进化选择找到最优配置：
- 低敏感层 → W2 + A8 (2-bit权重 + 8-bit激活)
- 高敏感层 → W4 + A4 (4-bit权重 + 4-bit激活)

量化策略说明:
-----------
- W2 + A8: 低精度权重需要高精度激活来补偿信息损失
- W4 + A4: 权重精度足够时可使用低精度激活进一步压缩

算法流程:
---------
1. 初始化种群: 随机生成 N 个位宽配置
2. 适应度评估: 计算加权 MSE（越小越好）
3. 选择: 锦标赛选择保留优秀个体
4. 交叉: 两点/均匀交叉生成子代
5. 变异: 根据敏感度智能变异
6. 迭代: 重复 2-5 直到收敛

核心组件:
---------
- MixedPrecisionGA: 遗传算法主类
- LayerSensitivityAnalyzer: 层敏感度分析器

使用示例:
---------
>>> from genetic_optim import MixedPrecisionGA, LayerSensitivityAnalyzer
>>> 
>>> # 创建优化器
>>> ga = MixedPrecisionGA(n_layers=196, population_size=30)
>>> 
>>> # 执行优化
>>> best_config = ga.optimize(fitness_func, target_compression=0.25)
"""

import numpy as np
from typing import Callable, Dict, List, Tuple, Optional
from collections import defaultdict


# ============================================================================
# 遗传算法优化器
# ============================================================================

class MixedPrecisionGA:
    """
    混合精度遗传算法优化器
    
    用于搜索最优的逐层量化位宽配置，平衡压缩率和精度。
    
    特性:
    -----
    - 智能初始化: 基于敏感度信息初始化种群
    - 精英保留: 保留最优个体到下一代
    - 自适应变异: 根据收敛情况调整变异率
    - 早停机制: 连续多代无改进时提前停止
    
    参数:
    -----
    n_layers : int
        待优化的层数
    population_size : int
        种群大小（建议 25-40）
    n_generations : int
        迭代代数（建议 20-30）
    mutation_rate : float
        初始变异率（建议 0.1-0.15）
    elite_ratio : float
        精英保留比例（建议 0.1-0.2）
    adaptive_mutation : bool
        是否启用自适应变异率
    
    示例:
    ------
    >>> ga = MixedPrecisionGA(n_layers=196, population_size=30)
    >>> best_config = ga.optimize(fitness_func, target_compression=0.25)
    """
    
    def __init__(self, n_layers: int, population_size: int = 30, 
                 n_generations: int = 25, mutation_rate: float = 0.12,
                 elite_ratio: float = 0.15, adaptive_mutation: bool = True):
        """
        初始化遗传算法优化器（增强版）
        
        参数：
        -----
        n_layers : int
            待优化的层数（Qwen2.5-7B共196个线性层）
        population_size : int
            种群大小，建议25-40（增大以提高搜索覆盖）
        n_generations : int
            进化代数，建议20-30（增加以获得更好收敛）
        mutation_rate : float
            初始变异率，建议0.1-0.15
        elite_ratio : float
            精英保留比例，建议0.1-0.2
        adaptive_mutation : bool
            是否启用自适应变异率
        """
        self.n_layers = n_layers
        self.pop_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.initial_mutation_rate = mutation_rate
        self.elite_ratio = elite_ratio
        self.adaptive_mutation = adaptive_mutation
        
        # 可选位宽: W2(低敏感度), W4(高敏感度)
        self.bit_options = [2, 4]
        
        # 层敏感度权重（可由外部设置）
        self.layer_weights: Optional[np.ndarray] = None
        self.layer_sensitivity_ratios: Optional[np.ndarray] = None
        
        # 收敛追踪
        self.stagnation_counter = 0
        self.best_score_history = []
    
    def set_layer_sensitivities(self, sensitivities: Dict[str, Dict[int, float]], 
                                   layer_names: List[str]):
        """
        设置层敏感度信息，用于智能初始化和加权适应度
        
        参数：
        -----
        sensitivities : Dict
            层敏感度字典 {layer_name: {2: mse_w2, 4: mse_w4}}
        layer_names : List[str]
            按顺序排列的层名列表
        """
        n = len(layer_names)
        self.layer_weights = np.ones(n)
        self.layer_sensitivity_ratios = np.zeros(n)
        
        # 计算每层的敏感度比例 (W2_MSE / W4_MSE)
        ratios = []
        for i, name in enumerate(layer_names):
            sens = sensitivities.get(name, {2: 0.1, 4: 0.01})
            w2_mse = sens.get(2, 0.1)
            w4_mse = sens.get(4, 0.01)
            
            # 避免除零
            ratio = w2_mse / max(w4_mse, 1e-8)
            ratios.append(ratio)
            self.layer_sensitivity_ratios[i] = ratio
        
        # 根据敏感度比例计算层权重（比例越高，层越重要）
        ratios = np.array(ratios)
        # 使用对数变换平滑权重
        log_ratios = np.log1p(ratios)
        # 归一化到 [0.5, 2.0] 范围
        if log_ratios.max() > log_ratios.min():
            normalized = (log_ratios - log_ratios.min()) / (log_ratios.max() - log_ratios.min())
            self.layer_weights = 0.5 + 1.5 * normalized
        
        print(f"  敏感度比例范围: {ratios.min():.2f} - {ratios.max():.2f}")
        print(f"  层权重范围: {self.layer_weights.min():.2f} - {self.layer_weights.max():.2f}")
    
    def initialize_population(self) -> List[np.ndarray]:
        """
        智能初始化种群（基于敏感度信息）
        
        策略：
        - 如果有敏感度信息，根据敏感度比例决定初始位宽
        - 敏感度比例高的层更可能初始化为W4
        - 包含多样性个体以避免局部最优
        
        返回：
        ------
        List[np.ndarray]
            种群列表，每个个体是一个位宽配置数组
        """
        population = []
        
        # 如果有敏感度信息，使用智能初始化
        if self.layer_sensitivity_ratios is not None:
            # 计算自适应阈值（使用中位数作为分界）
            median_ratio = np.median(self.layer_sensitivity_ratios)
            q25 = np.percentile(self.layer_sensitivity_ratios, 25)
            q75 = np.percentile(self.layer_sensitivity_ratios, 75)
            
            print(f"  敏感度统计: Q25={q25:.2f}, 中位数={median_ratio:.2f}, Q75={q75:.2f}")
            
            for i in range(self.pop_size):
                individual = np.zeros(self.n_layers, dtype=int)
                
                # 使用不同阈值创建多样性
                if i < self.pop_size // 4:
                    # 激进压缩（使用Q25阈值）
                    threshold = q25
                elif i < self.pop_size // 2:
                    # 平衡（使用中位数阈值）
                    threshold = median_ratio
                elif i < 3 * self.pop_size // 4:
                    # 保守（使用Q75阈值）
                    threshold = q75
                else:
                    # 随机阈值
                    threshold = np.random.uniform(q25, q75)
                
                for j in range(self.n_layers):
                    ratio = self.layer_sensitivity_ratios[j]
                    # 敏感度比例高于阈值使用W4，否则使用W2
                    # 加入随机扰动
                    if ratio > threshold * (1 + np.random.uniform(-0.2, 0.2)):
                        individual[j] = 4
                    else:
                        individual[j] = 2
                
                population.append(individual)
        else:
            # 无敏感度信息时使用加权随机初始化
            for _ in range(self.pop_size):
                individual = np.random.choice(
                    self.bit_options, 
                    size=self.n_layers, 
                    p=[0.55, 0.45]  # 略偏向W2
                )
                population.append(individual)
        
        return population
    
    def compute_model_size(self, individual: np.ndarray) -> float:
        """
        计算相对模型大小
        
        参数：
        -----
        individual : np.ndarray
            位宽配置数组
        
        返回：
        ------
        float
            相对于FP16的大小比例 (0.0~1.0)
        """
        return np.sum(individual) / (self.n_layers * 16)  # 相对于FP16
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """
        多点交叉操作（增强版）
        
        使用两点交叉或均匀交叉，提高基因组合多样性
        
        参数：
        -----
        parent1, parent2 : np.ndarray
            两个父代个体
        
        返回：
        ------
        np.ndarray
            子代个体
        """
        crossover_type = np.random.choice(['two_point', 'uniform'], p=[0.6, 0.4])
        
        if crossover_type == 'two_point':
            # 两点交叉
            points = sorted(np.random.choice(range(1, self.n_layers), 2, replace=False))
            child = np.concatenate([
                parent1[:points[0]], 
                parent2[points[0]:points[1]], 
                parent1[points[1]:]
            ])
        else:
            # 均匀交叉（每个基因50%概率来自任一父代）
            mask = np.random.rand(self.n_layers) < 0.5
            child = np.where(mask, parent1, parent2)
        
        return child
    
    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        智能变异操作（增强版）
        
        策略：
        - 高敏感度层更倾向变异到W4
        - 低敏感度层更倾向变异到W2
        - 根据层权重调整变异概率
        
        参数：
        -----
        individual : np.ndarray
            待变异个体
        
        返回：
        ------
        np.ndarray
            变异后的个体
        """
        for i in range(self.n_layers):
            # 根据层权重调整变异概率（高权重层变异概率稍低）
            layer_mut_rate = self.mutation_rate
            if self.layer_weights is not None:
                # 权重高的层变异概率降低30%
                layer_mut_rate *= (2.0 - self.layer_weights[i]) / 1.5
            
            if np.random.rand() < layer_mut_rate:
                # 智能变异：根据敏感度比例决定变异方向
                if self.layer_sensitivity_ratios is not None:
                    ratio = self.layer_sensitivity_ratios[i]
                    median_ratio = np.median(self.layer_sensitivity_ratios)
                    
                    # 高敏感度层更倾向W4
                    if ratio > median_ratio:
                        individual[i] = np.random.choice([2, 4], p=[0.3, 0.7])
                    else:
                        individual[i] = np.random.choice([2, 4], p=[0.7, 0.3])
                else:
                    individual[i] = np.random.choice(self.bit_options)
        
        return individual
    
    def update_mutation_rate(self, generation: int, best_improved: bool):
        """
        自适应调整变异率
        
        策略：
        - 如果连续多代无改进，增加变异率以跳出局部最优
        - 如果有改进，逐渐降低变异率以精细搜索
        """
        if not self.adaptive_mutation:
            return
        
        if best_improved:
            self.stagnation_counter = 0
            # 缓慢降低变异率
            self.mutation_rate = max(0.05, self.mutation_rate * 0.95)
        else:
            self.stagnation_counter += 1
            if self.stagnation_counter >= 3:
                # 连续3代无改进，大幅增加变异率
                self.mutation_rate = min(0.3, self.mutation_rate * 1.5)
                self.stagnation_counter = 0
    
    def optimize(self, fitness_func: Callable[[np.ndarray], float], 
                 target_compression: float = 0.3) -> np.ndarray:
        """
        执行遗传算法优化（增强版）
        
        特性：
        - 精英保留策略
        - 自适应变异率
        - 锦标赛选择
        - 早停机制
        
        参数：
        -----
        fitness_func : Callable
            适应度函数，输入位宽配置，返回质量分数（越高越好）
            通常使用加权负MSE作为适应度
        target_compression : float
            目标压缩比（相对于FP16），默认0.3表示30%大小
        
        返回：
        ------
        np.ndarray
            最优位宽配置
        """
        population = self.initialize_population()
        best_individual = None
        best_score = -float('inf')
        no_improve_count = 0
        
        # 精英数量
        n_elite = max(2, int(self.pop_size * self.elite_ratio))
        
        for gen in range(self.n_generations):
            scores = []
            
            for indiv in population:
                # 计算加权质量分数
                quality = fitness_func(indiv)
                
                # 大小惩罚（使用平滑惩罚函数）
                size_ratio = self.compute_model_size(indiv)
                if size_ratio > target_compression:
                    # 超出目标时使用二次惩罚
                    size_penalty = ((size_ratio - target_compression) * 15) ** 1.5
                else:
                    # 低于目标时给予小奖励
                    size_penalty = -((target_compression - size_ratio) * 2)
                
                # 综合分数
                score = quality - size_penalty
                scores.append(score)
            
            scores = np.array(scores)
            sorted_idx = np.argsort(scores)[::-1]
            
            # 检查是否有改进
            current_best_score = scores[sorted_idx[0]]
            best_improved = current_best_score > best_score
            
            if best_improved:
                best_score = current_best_score
                best_individual = population[sorted_idx[0]].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # 自适应变异率
            self.update_mutation_rate(gen, best_improved)
            
            # 早停检查（连续10代无改进）
            if no_improve_count >= 10:
                print(f"  早停: 连续{no_improve_count}代无改进")
                break
            
            # === 精英保留 ===
            elite = [population[i].copy() for i in sorted_idx[:n_elite]]
            
            # === 锦标赛选择 ===
            def tournament_select(tournament_size=3):
                candidates = np.random.choice(len(population), tournament_size, replace=False)
                winner = candidates[np.argmax([scores[c] for c in candidates])]
                return population[winner]
            
            # === 生成新种群 ===
            new_pop = elite[:]  # 精英直接进入下一代
            
            while len(new_pop) < self.pop_size:
                # 锦标赛选择父代
                parent1 = tournament_select()
                parent2 = tournament_select()
                
                # 交叉
                child = self.crossover(parent1, parent2)
                # 变异
                child = self.mutate(child)
                new_pop.append(child)
            
            population = new_pop
            
            # 打印进度
            self._print_generation_stats(gen, best_individual, best_score)
            self.best_score_history.append(best_score)
        
        print(f"\n  最终变异率: {self.mutation_rate:.3f}")
        return best_individual
    
    def _print_generation_stats(self, gen: int, best: np.ndarray, score: float):
        """打印每代的统计信息"""
        w2_count = np.sum(best == 2)
        w4_count = np.sum(best == 4)
        size = self.compute_model_size(best)
        mut_indicator = "↑" if self.mutation_rate > self.initial_mutation_rate else "↓" if self.mutation_rate < self.initial_mutation_rate else "→"
        print(f"第{gen+1}代: 分数={score:.4f}, 大小={size:.1%}, "
              f"W2={w2_count}, W4={w4_count}, 变异率={self.mutation_rate:.3f}{mut_indicator}")


class LayerSensitivityAnalyzer:
    """
    层敏感度分析器（增强版）
    
    评估每层对不同量化位宽的敏感程度，用于指导混合精度配置。
    
    敏感度定义：
    -----------
    敏感度 = MSE(量化输出, 原始输出)
    敏感度比例 = MSE_W2 / MSE_W4（比例越高越敏感）
    
    敏感度越高的层应使用更高的位宽，敏感度低的层可使用W2激进压缩。
    
    敏感度分类（自适应阈值）：
    ------------------------
    - 使用敏感度比例（W2_MSE / W4_MSE）作为分类依据
    - 根据全局分布自适应计算阈值
    - 支持基于层位置的特殊处理（首尾层通常更敏感）
    """
    
    def __init__(self, bit_options: List[int] = None):
        """
        初始化敏感度分析器
        
        参数：
        -----
        bit_options : List[int]
            要测试的位宽列表，默认 [2, 4]
        """
        self.bit_options = bit_options or [2, 4]
    
    def analyze(self, layer, calib_input, quantize_fn: Callable) -> Dict[int, float]:
        """
        分析单层对不同位宽的敏感度
        
        参数：
        -----
        layer : nn.Module
            待分析的线性层
        calib_input : torch.Tensor
            校准输入
        quantize_fn : Callable
            量化函数
        
        返回：
        ------
        Dict[int, float]
            位宽到MSE的映射，如 {2: 0.15, 4: 0.05, 8: 0.01}
        """
        import torch
        
        # 获取原始输出
        original_output = layer(calib_input)
        
        sensitivity = {}
        for n_bits in self.bit_options:
            # 量化权重
            w = layer.weight
            limit = w.abs().amax() * 0.9  # 裁剪到90%
            w_clipped = torch.clamp(w, -limit, limit)
            w_q = quantize_fn(w_clipped, n_bits=n_bits, group_size=128, sym=True)
            
            # 量化激活（A8固定）
            x_q = quantize_fn(calib_input, n_bits=8, group_size=-1, sym=False)
            
            # 计算量化后输出
            out_q = torch.nn.functional.linear(x_q, w_q, layer.bias)
            
            # 计算MSE
            mse = torch.mean((out_q - original_output) ** 2).item()
            sensitivity[n_bits] = mse
        
        return sensitivity
    
    def classify_sensitivity(self, mse_w2: float, mse_w4: float = None, 
                              layer_idx: int = None, n_layers: int = None) -> str:
        """
        根据敏感度比例对层进行分类（增强版）
        
        使用敏感度比例（W2_MSE / W4_MSE）而非绝对MSE值进行分类，
        比例越高说明使用W2相比W4的精度损失越大，应使用W4。
        
        参数：
        -----
        mse_w2 : float
            W2量化的MSE
        mse_w4 : float, optional
            W4量化的MSE
        layer_idx : int, optional
            层索引（用于首尾层特殊处理）
        n_layers : int, optional
            总层数
        
        返回：
        ------
        str
            敏感度类别描述
        """
        # 计算敏感度比例
        if mse_w4 is not None and mse_w4 > 1e-8:
            ratio = mse_w2 / mse_w4
        else:
            ratio = mse_w2 * 100  # 如果W4 MSE很小，用绝对值
        
        # 首尾层特殊处理（前3层和后3层更敏感）
        is_edge_layer = False
        if layer_idx is not None and n_layers is not None:
            if layer_idx < 3 or layer_idx >= n_layers - 3:
                is_edge_layer = True
        
        # 自适应阈值
        # 比例 > 30 表示W2比W4差很多，应使用W4
        # 比例 < 15 表示W2和W4差距不大，可以使用W2
        if is_edge_layer:
            # 边缘层使用更严格的阈值
            threshold = 10
        else:
            threshold = 25
        
        if ratio > threshold:
            return f"高敏感度(使用W4) [比例:{ratio:.1f}]"
        else:
            return f"低敏感度(使用W2) [比例:{ratio:.1f}]"
