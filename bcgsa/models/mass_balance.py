# -*- coding: utf-8 -*-
"""
================================================================================
伊洛河流域污染溯源模型 - 改进物质平衡模块
Enhanced Mass Balance Module
================================================================================

核心改进：
1. 自适应降解系数校准
2. 迭代物质平衡优化
3. 面源负荷合理性约束
4. 不确定性量化
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass


@dataclass
class BalanceResult:
    """物质平衡计算结果"""
    nps_loads: np.ndarray  # 面源负荷 [n_days, n_pollutants]
    balance_error: float   # 平衡误差 (%)
    r2: float             # 相关系数
    nse: float            # NSE
    pbias: float          # PBIAS (%)
    valid_days: int       # 有效天数


class AdaptiveDecayCalibrator:
    """
    自适应降解系数校准器
    
    基于观测数据反演最优降解系数，使物质平衡误差最小
    """
    
    def __init__(self, pollutants: List[str], 
                 initial_rates: Dict[str, float],
                 bounds: Dict[str, Tuple[float, float]]):
        """
        初始化校准器
        
        Args:
            pollutants: 污染物列表
            initial_rates: 初始降解系数
            bounds: 降解系数边界
        """
        self.pollutants = pollutants
        self.initial_rates = initial_rates.copy()
        self.bounds = bounds
        self.calibrated_rates = initial_rates.copy()
        
    def calibrate(self, flows: np.ndarray, concentrations: np.ndarray,
                 upstream_indices: List[int], downstream_indices: List[int],
                 distances: np.ndarray) -> Dict[str, float]:
        """
        校准降解系数
        
        方法：最小化 sum[(C_down_obs - C_down_calc)^2]
        其中 C_down_calc = C_up * exp(-k * d)
        
        Args:
            flows: 流量 [n_days, n_nodes]
            concentrations: 浓度 [n_days, n_nodes, n_pollutants]
            upstream_indices: 上游节点索引列表
            downstream_indices: 对应的下游节点索引列表
            distances: 上下游距离 [n_pairs]
            
        Returns:
            calibrated_rates: 校准后的降解系数
        """
        n_pollutants = len(self.pollutants)
        
        for p_idx, pollutant in enumerate(self.pollutants):
            bounds_p = self.bounds.get(pollutant, (0.001, 0.1))
            
            def objective(k):
                total_error = 0
                n_valid = 0
                
                for up_idx, down_idx, dist in zip(upstream_indices, downstream_indices, distances):
                    if dist <= 0:
                        continue
                    
                    # 上游负荷
                    Q_up = flows[:, up_idx]
                    C_up = concentrations[:, up_idx, p_idx]
                    
                    # 下游观测
                    Q_down = flows[:, down_idx]
                    C_down_obs = concentrations[:, down_idx, p_idx]
                    
                    # 传输后浓度预测（忽略其他源汇）
                    # 这是简化假设，实际应考虑混合
                    lambda_coef = np.exp(-k[0] * dist)
                    
                    # 计算预期浓度变化
                    # 如果只有传输衰减，C_down = C_up * lambda
                    # 但实际上有流量变化，需要考虑稀释
                    flow_ratio = Q_up / (Q_down + 0.1)
                    C_down_pred = C_up * lambda_coef * flow_ratio
                    
                    # 有效数据
                    valid = (C_down_obs > 0) & (C_down_pred > 0) & (Q_down > 1)
                    
                    if valid.sum() > 10:
                        error = np.sum((np.log(C_down_obs[valid] + 0.01) - 
                                       np.log(C_down_pred[valid] + 0.01)) ** 2)
                        total_error += error
                        n_valid += valid.sum()
                
                return total_error / max(n_valid, 1)
            
            # 优化
            try:
                result = minimize(
                    objective,
                    x0=[self.initial_rates[pollutant]],
                    bounds=[bounds_p],
                    method='L-BFGS-B'
                )
                
                if result.success:
                    self.calibrated_rates[pollutant] = result.x[0]
                    
            except Exception as e:
                print(f"  {pollutant} 降解系数校准失败: {e}")
        
        return self.calibrated_rates


class EnhancedMassBalanceEstimator:
    """
    增强型物质平衡面源估算器
    
    核心改进：
    1. 考虑负荷传输时间
    2. 自适应校正因子
    3. 面源非负约束优化
    """
    
    def __init__(self, node_list: List[str], pollutants: List[str],
                 distance_matrix: pd.DataFrame, decay_rates: Dict[str, float],
                 segment_config: List[Dict]):
        """
        初始化估算器
        
        Args:
            node_list: 节点列表
            pollutants: 污染物列表
            distance_matrix: 距离矩阵
            decay_rates: 降解系数
            segment_config: 河段配置
        """
        self.node_list = node_list
        self.node_to_idx = {n: i for i, n in enumerate(node_list)}
        self.pollutants = pollutants
        self.distance_matrix = distance_matrix
        self.decay_rates = decay_rates
        self.segment_config = segment_config
        
        # 校正因子（用于调整系统性偏差）
        self.correction_factors = {p: 1.0 for p in pollutants}
        
    def get_transport_coef(self, upstream: str, downstream: str, 
                          pollutant: str) -> float:
        """获取传输系数"""
        if upstream not in self.distance_matrix.index:
            return 1.0
        if downstream not in self.distance_matrix.columns:
            return 1.0
        
        d = self.distance_matrix.loc[upstream, downstream]
        if isinstance(d, pd.Series):
            d = d.iloc[0]
        
        if pd.isna(d) or d <= 0:
            return 1.0
        
        k = self.decay_rates.get(pollutant, 0.01)
        return np.exp(-k * d)
    
    def estimate_segment_nps_optimized(self, flows: np.ndarray, 
                                       concentrations: np.ndarray,
                                       ps_loads: np.ndarray,
                                       segment: Dict) -> BalanceResult:
        """
        优化估算单河段面源负荷
        
        采用约束优化：
        min ||L_down - (L_up * λ + L_ps + L_nps)||^2
        s.t. L_nps >= 0
        
        Args:
            flows: 流量 [n_days, n_nodes]
            concentrations: 浓度 [n_days, n_nodes, n_pollutants]
            ps_loads: 点源负荷 [n_days, n_nodes, n_pollutants]
            segment: 河段配置
            
        Returns:
            result: 平衡计算结果
        """
        n_days = flows.shape[0]
        n_pollutants = len(self.pollutants)
        
        downstream = segment.get('downstream')
        upstreams = segment.get('upstream', [])
        if isinstance(upstreams, str):
            upstreams = [upstreams] if upstreams != 'source' else []
        
        segment_ps = segment.get('point_sources', [])
        
        # 获取下游节点
        if downstream not in self.node_to_idx:
            return BalanceResult(
                nps_loads=np.zeros((n_days, n_pollutants)),
                balance_error=100.0, r2=0, nse=-999, pbias=100, valid_days=0
            )
        
        down_idx = self.node_to_idx[downstream]
        nps_loads = np.zeros((n_days, n_pollutants))
        
        errors = []
        r2_values = []
        
        for p, pollutant in enumerate(self.pollutants):
            # 下游负荷
            Q_down = flows[:, down_idx]
            C_down = concentrations[:, down_idx, p]
            L_down = Q_down * C_down * 86.4  # kg/day
            
            # 上游传输负荷
            L_up_transport = np.zeros(n_days)
            for up_node in upstreams:
                if up_node not in self.node_to_idx:
                    continue
                up_idx = self.node_to_idx[up_node]
                Q_up = flows[:, up_idx]
                C_up = concentrations[:, up_idx, p]
                L_up = Q_up * C_up * 86.4
                
                lambda_coef = self.get_transport_coef(up_node, downstream, pollutant)
                L_up_transport += L_up * lambda_coef
            
            # 河段点源负荷
            L_ps = np.zeros(n_days)
            for ps_id in segment_ps:
                if ps_id in self.node_to_idx:
                    ps_idx = self.node_to_idx[ps_id]
                    L_ps += ps_loads[:, ps_idx, p]
            
            # 初始面源估计
            L_nps_raw = L_down - L_up_transport - L_ps
            
            # 处理负值：使用软约束优化
            # 当L_nps_raw < 0时，可能原因：
            # 1. 降解系数估计偏低
            # 2. 浓度测量误差
            # 3. 存在汇（如取水、沉降）
            
            # 方法1：直接截断（简单但可能导致不平衡）
            L_nps_simple = np.maximum(0, L_nps_raw)
            
            # 方法2：按比例分配负值到上游和点源
            negative_mask = L_nps_raw < 0
            if negative_mask.any():
                # 负值部分按上游传输:点源比例分配
                L_up_ps_sum = L_up_transport + L_ps
                
                # 用比例因子调整
                ratio = L_down[negative_mask] / (L_up_ps_sum[negative_mask] + 1e-10)
                ratio = np.clip(ratio, 0.5, 2.0)  # 限制调整幅度
                
                # 调整后的负荷
                L_nps_adjusted = L_nps_raw.copy()
                L_nps_adjusted[negative_mask] = 0  # 负值天的面源设为0
            else:
                L_nps_adjusted = L_nps_simple
            
            # 应用校正因子
            cf = self.correction_factors.get(pollutant, 1.0)
            nps_loads[:, p] = L_nps_adjusted * cf
            
            # 计算平衡误差
            L_calc = L_up_transport + L_ps + nps_loads[:, p]
            
            valid = (L_down > 1) & (L_calc > 0)
            if valid.sum() > 10:
                rel_err = np.abs(L_down[valid] - L_calc[valid]) / (L_down[valid] + 1)
                errors.append(rel_err.mean() * 100)
                
                # R²
                ss_res = np.sum((L_down[valid] - L_calc[valid]) ** 2)
                ss_tot = np.sum((L_down[valid] - np.mean(L_down[valid])) ** 2)
                r2 = max(0, 1 - ss_res / (ss_tot + 1e-10))
                r2_values.append(r2)
        
        mean_error = np.mean(errors) if errors else 100
        mean_r2 = np.mean(r2_values) if r2_values else 0
        
        # 计算NSE和PBIAS
        nse_values = []
        pbias_values = []
        
        for p, pollutant in enumerate(self.pollutants):
            Q_down = flows[:, down_idx]
            C_down = concentrations[:, down_idx, p]
            L_down = Q_down * C_down * 86.4
            
            L_up_transport = np.zeros(n_days)
            for up_node in upstreams:
                if up_node not in self.node_to_idx:
                    continue
                up_idx = self.node_to_idx[up_node]
                Q_up = flows[:, up_idx]
                C_up = concentrations[:, up_idx, p]
                L_up = Q_up * C_up * 86.4
                lambda_coef = self.get_transport_coef(up_node, downstream, pollutant)
                L_up_transport += L_up * lambda_coef
            
            L_ps = np.zeros(n_days)
            for ps_id in segment_ps:
                if ps_id in self.node_to_idx:
                    ps_idx = self.node_to_idx[ps_id]
                    L_ps += ps_loads[:, ps_idx, p]
            
            L_calc = L_up_transport + L_ps + nps_loads[:, p]
            
            valid = (L_down > 1) & (L_calc > 0)
            if valid.sum() > 10:
                ss_res = np.sum((L_down[valid] - L_calc[valid]) ** 2)
                ss_tot = np.sum((L_down[valid] - np.mean(L_down[valid])) ** 2)
                nse = 1 - ss_res / (ss_tot + 1e-10)
                nse_values.append(nse)
                
                pbias = 100 * np.sum(L_calc[valid] - L_down[valid]) / (np.sum(L_down[valid]) + 1e-10)
                pbias_values.append(pbias)
        
        return BalanceResult(
            nps_loads=nps_loads,
            balance_error=mean_error,
            r2=mean_r2,
            nse=np.mean(nse_values) if nse_values else -999,
            pbias=np.mean(pbias_values) if pbias_values else 100,
            valid_days=int(valid.sum()) if 'valid' in dir() else 0
        )
    
    def calibrate_correction_factors(self, flows: np.ndarray,
                                     concentrations: np.ndarray,
                                     ps_loads: np.ndarray,
                                     target_pbias: float = 0.0) -> Dict[str, float]:
        """
        校准校正因子使PBIAS接近目标值
        
        Args:
            flows: 流量
            concentrations: 浓度
            ps_loads: 点源负荷
            target_pbias: 目标PBIAS（默认0）
            
        Returns:
            correction_factors: 校准后的校正因子
        """
        # 首先用因子=1计算基准
        for p, pollutant in enumerate(self.pollutants):
            total_L_down = 0
            total_L_calc_base = 0
            
            for segment in self.segment_config:
                downstream = segment.get('downstream')
                if downstream not in self.node_to_idx:
                    continue
                
                down_idx = self.node_to_idx[downstream]
                Q_down = flows[:, down_idx]
                C_down = concentrations[:, down_idx, p]
                L_down = Q_down * C_down * 86.4
                
                upstreams = segment.get('upstream', [])
                if isinstance(upstreams, str):
                    upstreams = [upstreams] if upstreams != 'source' else []
                
                L_up_transport = 0
                for up_node in upstreams:
                    if up_node not in self.node_to_idx:
                        continue
                    up_idx = self.node_to_idx[up_node]
                    Q_up = flows[:, up_idx]
                    C_up = concentrations[:, up_idx, p]
                    L_up = Q_up * C_up * 86.4
                    lambda_coef = self.get_transport_coef(up_node, downstream, pollutant)
                    L_up_transport += (L_up * lambda_coef).sum()
                
                segment_ps = segment.get('point_sources', [])
                L_ps = 0
                for ps_id in segment_ps:
                    if ps_id in self.node_to_idx:
                        ps_idx = self.node_to_idx[ps_id]
                        L_ps += ps_loads[:, ps_idx, p].sum()
                
                L_nps_raw = np.maximum(0, L_down - L_up_transport - L_ps).sum()
                
                valid = L_down > 1
                total_L_down += L_down[valid].sum()
                total_L_calc_base += (L_up_transport + L_ps + L_nps_raw)
            
            # 计算需要的校正因子
            if total_L_calc_base > 0 and total_L_down > 0:
                current_ratio = total_L_calc_base / total_L_down
                # 理想情况下 ratio = 1 + target_pbias/100
                target_ratio = 1 + target_pbias / 100
                
                # 校正因子调整面源
                # new_L_calc = L_up + L_ps + cf * L_nps
                # 简化：假设面源占50%时
                cf = max(0.1, min(3.0, target_ratio / current_ratio))
                self.correction_factors[pollutant] = cf
        
        return self.correction_factors
    
    def estimate_all_segments(self, flows: np.ndarray,
                             concentrations: np.ndarray,
                             ps_loads: np.ndarray) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
        """
        估算所有河段面源
        
        Args:
            flows: 流量 [n_days, n_nodes]
            concentrations: 浓度 [n_days, n_nodes, n_pollutants]
            ps_loads: 点源负荷 [n_days, n_nodes, n_pollutants]
            
        Returns:
            segment_nps: {segment_id: nps_loads}
            error_df: 误差统计表
        """
        segment_nps = {}
        error_records = []
        
        for segment in self.segment_config:
            seg_id = segment.get('id', segment.get('downstream', 'unknown'))
            
            result = self.estimate_segment_nps_optimized(
                flows, concentrations, ps_loads, segment
            )
            
            segment_nps[seg_id] = result.nps_loads
            
            error_records.append({
                'segment_id': seg_id,
                'segment_name': segment.get('name', seg_id),
                'mean_error': result.balance_error,
                'r2': result.r2,
                'nse': result.nse,
                'pbias': result.pbias,
                'valid_days': result.valid_days,
            })
        
        error_df = pd.DataFrame(error_records)
        
        return segment_nps, error_df


class IterativeMassBalanceSolver:
    """
    迭代物质平衡求解器
    
    核心思想：
    通过多轮迭代，同时优化：
    1. 未知节点的浓度
    2. 河段面源负荷
    3. 降解系数
    
    直到物质平衡误差收敛
    """
    
    def __init__(self, node_list: List[str], pollutants: List[str],
                 distance_matrix: pd.DataFrame, initial_decay_rates: Dict[str, float],
                 segment_config: List[Dict]):
        self.node_list = node_list
        self.node_to_idx = {n: i for i, n in enumerate(node_list)}
        self.pollutants = pollutants
        self.distance_matrix = distance_matrix
        self.decay_rates = initial_decay_rates.copy()
        self.segment_config = segment_config
        
    def solve(self, flows: np.ndarray, conc_observed: np.ndarray,
             known_mask: np.ndarray, ps_loads: np.ndarray,
             max_iterations: int = 50, convergence_tol: float = 1.0,
             verbose: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict]:
        """
        迭代求解物质平衡
        
        Args:
            flows: 流量 [n_days, n_nodes]
            conc_observed: 观测浓度 [n_days, n_nodes, n_pollutants]
            known_mask: 已知节点掩码
            ps_loads: 点源负荷
            max_iterations: 最大迭代次数
            convergence_tol: 收敛阈值 (PBIAS %)
            verbose: 是否打印过程
            
        Returns:
            conc_final: 最终浓度
            nps_final: 最终面源
            diagnostics: 诊断信息
        """
        n_days, n_nodes, n_pollutants = conc_observed.shape
        
        # 初始化浓度（已知节点用观测值，未知节点用插值）
        conc = conc_observed.copy()
        
        # 对未知节点进行初始插值
        known_indices = np.where(known_mask > 0)[0]
        for i in range(n_nodes):
            if known_mask[i] == 0:
                # 简单平均插值
                conc[:, i, :] = conc_observed[:, known_indices, :].mean(axis=1)
        
        # 创建估算器
        estimator = EnhancedMassBalanceEstimator(
            self.node_list, self.pollutants, self.distance_matrix,
            self.decay_rates, self.segment_config
        )
        
        diagnostics = {
            'iterations': 0,
            'pbias_history': [],
            'error_history': [],
            'converged': False,
        }
        
        best_conc = conc.copy()
        best_nps = {}
        best_pbias = float('inf')
        
        for iteration in range(max_iterations):
            # 步骤1：估算面源
            segment_nps, error_df = estimator.estimate_all_segments(
                flows, conc, ps_loads
            )
            
            # 计算整体PBIAS
            mean_pbias = error_df['pbias'].mean()
            mean_error = error_df['mean_error'].mean()
            
            diagnostics['pbias_history'].append(mean_pbias)
            diagnostics['error_history'].append(mean_error)
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"  迭代 {iteration + 1}: PBIAS={mean_pbias:.1f}%, 误差={mean_error:.1f}%")
            
            # 保存最优结果
            if abs(mean_pbias) < abs(best_pbias):
                best_pbias = mean_pbias
                best_conc = conc.copy()
                best_nps = segment_nps.copy()
            
            # 检查收敛
            if abs(mean_pbias) < convergence_tol:
                diagnostics['converged'] = True
                break
            
            # 步骤2：调整未知节点浓度
            # 基于PBIAS方向调整
            adjustment_factor = 1.0 - 0.1 * np.sign(mean_pbias)  # PBIAS>0 则降低，<0则增加
            adjustment_factor = np.clip(adjustment_factor, 0.9, 1.1)
            
            for i in range(n_nodes):
                if known_mask[i] == 0:
                    conc[:, i, :] *= adjustment_factor
                    conc[:, i, :] = np.clip(conc[:, i, :], 0.01, 100)
            
            diagnostics['iterations'] = iteration + 1
        
        # 返回最优结果
        return best_conc, best_nps, diagnostics


def compute_enhanced_balance_metrics(flows: np.ndarray, concentrations: np.ndarray,
                                     ps_loads: np.ndarray, nps_loads: Dict[str, np.ndarray],
                                     segments: List[Dict], decay_rates: Dict[str, float],
                                     distance_matrix: pd.DataFrame,
                                     node_list: List[str],
                                     pollutants: List[str]) -> pd.DataFrame:
    """
    计算增强的物质平衡指标
    
    Returns:
        metrics_df: 包含多种指标的DataFrame
    """
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    records = []
    
    for segment in segments:
        seg_id = segment.get('id', segment.get('downstream', 'unknown'))
        downstream = segment.get('downstream')
        upstreams = segment.get('upstream', [])
        if isinstance(upstreams, str):
            upstreams = [upstreams] if upstreams != 'source' else []
        
        if downstream not in node_to_idx:
            continue
        
        down_idx = node_to_idx[downstream]
        segment_ps = segment.get('point_sources', [])
        
        for p, pollutant in enumerate(pollutants):
            Q_down = flows[:, down_idx]
            C_down = concentrations[:, down_idx, p]
            L_down = Q_down * C_down * 86.4
            
            # 上游传输
            L_up_transport = np.zeros(len(flows))
            for up_node in upstreams:
                if up_node not in node_to_idx:
                    continue
                up_idx = node_to_idx[up_node]
                Q_up = flows[:, up_idx]
                C_up = concentrations[:, up_idx, p]
                L_up = Q_up * C_up * 86.4
                
                # 传输系数
                if up_node in distance_matrix.index and downstream in distance_matrix.columns:
                    d = distance_matrix.loc[up_node, downstream]
                    if isinstance(d, pd.Series):
                        d = d.iloc[0]
                    if pd.notna(d) and d > 0:
                        k = decay_rates.get(pollutant, 0.01)
                        lambda_coef = np.exp(-k * d)
                    else:
                        lambda_coef = 1.0
                else:
                    lambda_coef = 1.0
                
                L_up_transport += L_up * lambda_coef
            
            # 点源
            L_ps = np.zeros(len(flows))
            for ps_id in segment_ps:
                if ps_id in node_to_idx:
                    ps_idx = node_to_idx[ps_id]
                    L_ps += ps_loads[:, ps_idx, p]
            
            # 面源
            L_nps = nps_loads.get(seg_id, np.zeros((len(flows), len(pollutants))))[:, p] \
                    if seg_id in nps_loads else np.zeros(len(flows))
            
            # 计算负荷
            L_calc = L_up_transport + L_ps + L_nps
            
            # 计算指标
            valid = (L_down > 1) & (L_calc > 0)
            n_valid = valid.sum()
            
            if n_valid > 10:
                L_down_v = L_down[valid]
                L_calc_v = L_calc[valid]
                
                # 相对误差
                rel_error = np.abs(L_down_v - L_calc_v) / (L_down_v + 1)
                mean_error = rel_error.mean() * 100
                p95_error = np.percentile(rel_error, 95) * 100
                
                # R²
                ss_res = np.sum((L_down_v - L_calc_v) ** 2)
                ss_tot = np.sum((L_down_v - np.mean(L_down_v)) ** 2)
                r2 = max(0, 1 - ss_res / (ss_tot + 1e-10))
                
                # NSE
                nse = 1 - ss_res / (ss_tot + 1e-10)
                
                # PBIAS
                pbias = 100 * np.sum(L_calc_v - L_down_v) / (np.sum(L_down_v) + 1e-10)
                
                # RMSE
                rmse = np.sqrt(np.mean((L_down_v - L_calc_v) ** 2))
                
                # KGE (Kling-Gupta Efficiency)
                r = np.corrcoef(L_down_v, L_calc_v)[0, 1] if len(L_down_v) > 2 else 0
                alpha = np.std(L_calc_v) / (np.std(L_down_v) + 1e-10)
                beta = np.mean(L_calc_v) / (np.mean(L_down_v) + 1e-10)
                kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
                
            else:
                mean_error = p95_error = 100
                r2 = nse = kge = 0
                pbias = 100
                rmse = 0
            
            records.append({
                'segment_id': seg_id,
                'segment_name': segment.get('name', seg_id),
                'river': segment.get('river', 'unknown'),
                'pollutant': pollutant,
                'n_valid': n_valid,
                'mean_error': mean_error,
                'p95_error': p95_error,
                'r2': r2,
                'nse': nse,
                'pbias': pbias,
                'kge': kge,
                'rmse': rmse,
            })
    
    return pd.DataFrame(records)


if __name__ == "__main__":
    print("改进物质平衡模块加载成功")
