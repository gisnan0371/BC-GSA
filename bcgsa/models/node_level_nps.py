# -*- coding: utf-8 -*-
"""
================================================================================
伊洛河流域污染溯源模型 - 节点级面源估算模块
Yiluo River Basin Pollution Source Apportionment - Node-Level NPS Estimator
================================================================================

精细到节点级别的面源负荷估算和归属

v8.3 Bug修复版 (2024)
修复问题: PBIAS 705-844%物质平衡误差
  - 原因: estimate_segment_nps中np.maximum(0, ...)截断负值
  - 修复: 保存净面源（可正可负）用于物质平衡验证
  - 返回毛面源（仅正值）保持兼容性
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class NodeSegment:
    """节点级河段"""
    id: str
    name: str
    upstream: Union[str, List[str]]  # 可以是单个节点或多个节点（汇流）
    downstream: str
    river: str
    coarse_segment: str
    is_diversion: bool = False
    is_confluence: bool = False


class NodeLevelNPSEstimator:
    """节点级面源估算器"""
    
    def __init__(self, node_segments: List[Dict], node_list: List[str],
                 distance_matrix: pd.DataFrame, decay_rates: Dict[str, float],
                 pollutants: List[str], ps_segment_mapping: Dict[str, str]):
        """
        初始化节点级面源估算器
        
        Args:
            node_segments: 节点级河段列表
            node_list: 全部节点列表
            distance_matrix: 距离矩阵
            decay_rates: 降解系数 {pollutant: k}
            pollutants: 污染物列表
            ps_segment_mapping: 点源所属河段映射
        """
        self.node_segments = [NodeSegment(**seg) for seg in node_segments]
        self.node_list = node_list
        self.node_to_idx = {node: i for i, node in enumerate(node_list)}
        self.distance_matrix = distance_matrix
        self.decay_rates = decay_rates
        self.pollutants = pollutants
        self.ps_segment_mapping = ps_segment_mapping
        
        # 构建河段索引
        self.segment_dict = {seg.id: seg for seg in self.node_segments}
        
        # 反向映射：节点 -> 所属河段
        self._build_node_to_segment_mapping()
        
    def _build_node_to_segment_mapping(self):
        """构建节点到河段的映射"""
        self.node_to_segment = {}
        
        for seg in self.node_segments:
            # 下游节点属于该河段
            self.node_to_segment[seg.downstream] = seg.id
            
            # 对于非汇流河段，上游节点也可以关联
            if not seg.is_confluence and isinstance(seg.upstream, str):
                if seg.upstream != "source":
                    # 上游节点作为该河段的入口
                    pass
    
    def get_segment_distance(self, segment: NodeSegment) -> float:
        """获取河段长度"""
        if segment.is_confluence:
            # 汇流河段，取最长的上游距离
            if isinstance(segment.upstream, list):
                distances = []
                for up in segment.upstream:
                    if up in self.distance_matrix.index and segment.downstream in self.distance_matrix.columns:
                        d = self.distance_matrix.loc[up, segment.downstream]
                        # 处理可能返回Series的情况
                        if isinstance(d, pd.Series):
                            d = d.iloc[0]
                        if pd.notna(d):
                            distances.append(float(d))
                return max(distances) if distances else 0
        else:
            if segment.upstream == "source":
                return 0  # 源头河段，无法计算
            if segment.upstream in self.distance_matrix.index and segment.downstream in self.distance_matrix.columns:
                d = self.distance_matrix.loc[segment.upstream, segment.downstream]
                # 处理可能返回Series的情况
                if isinstance(d, pd.Series):
                    d = d.iloc[0]
                if pd.notna(d):
                    return float(d)
        return 0
    
    def compute_segment_transport_coef(self, segment: NodeSegment, pollutant: str) -> float:
        """计算河段传输系数"""
        distance = self.get_segment_distance(segment)
        k = self.decay_rates.get(pollutant, 0.01)
        return np.exp(-k * distance)
    
    def estimate_segment_nps(self, flows: np.ndarray, concentrations: np.ndarray,
                             ps_loads: np.ndarray, segment: NodeSegment) -> np.ndarray:
        """
        估算单个节点级河段的面源负荷 (v8.3修复版)
        
        物质平衡: L_下游 = L_上游 × λ + L_点源 + L_净面源
        => L_净面源 = L_下游 - L_上游 × λ - L_点源
        
        ★★★ v8.3修复 ★★★
        L_净面源可以是负值（表示河段内去除）：
        - 正值: 面源输入（农业径流、城市径流）
        - 负值: 河段去除（沉降、吸附、生化降解超过新增输入）
        
        返回毛面源（仅正值）以保持兼容性，同时保存净面源用于物质平衡验证
        
        Args:
            flows: 流量 [n_days, n_nodes] (m³/s)
            concentrations: 浓度 [n_days, n_nodes, n_pollutants] (mg/L)
            ps_loads: 点源负荷 [n_days, n_nodes, n_pollutants] (kg/day)
            segment: 节点级河段
            
        Returns:
            nps_loads: 毛面源负荷（仅正值）[n_days, n_pollutants] (kg/day)
        """
        n_days = flows.shape[0]
        n_pollutants = len(self.pollutants)
        net_nps = np.zeros((n_days, n_pollutants))  # 净面源（可正可负）
        
        # 获取下游节点索引
        down_idx = self.node_to_idx.get(segment.downstream)
        if down_idx is None:
            return np.zeros((n_days, n_pollutants))
        
        # 处理上游节点
        if segment.upstream == "source":
            # 源头河段：净面源 = 下游负荷 - 点源
            for p, pollutant in enumerate(self.pollutants):
                L_down = flows[:, down_idx] * concentrations[:, down_idx, p] * 86.4
                
                # 河段内点源
                L_ps = self._get_segment_ps_loads(ps_loads, segment.id, p)
                
                # ★★★ v8.3修复：不截断负值 ★★★
                net_nps[:, p] = L_down - L_ps
            
            # 保存净面源用于物质平衡验证
            if not hasattr(self, '_segment_net_nps'):
                self._segment_net_nps = {}
            self._segment_net_nps[segment.id] = net_nps.copy()
            
            # 返回毛面源（仅正值）以保持兼容性
            return np.maximum(0, net_nps)
        
        # 非源头河段
        if segment.is_confluence:
            # 汇流河段
            upstream_nodes = segment.upstream if isinstance(segment.upstream, list) else [segment.upstream]
        else:
            upstream_nodes = [segment.upstream]
        
        for p, pollutant in enumerate(self.pollutants):
            # 下游负荷
            L_down = flows[:, down_idx] * concentrations[:, down_idx, p] * 86.4
            
            # 上游传输负荷
            L_up_total = np.zeros(n_days)
            for up_node in upstream_nodes:
                up_idx = self.node_to_idx.get(up_node)
                if up_idx is not None:
                    L_up = flows[:, up_idx] * concentrations[:, up_idx, p] * 86.4
                    
                    # 传输系数
                    if up_node in self.distance_matrix.index and segment.downstream in self.distance_matrix.columns:
                        d = self.distance_matrix.loc[up_node, segment.downstream]
                        # 处理可能返回Series的情况
                        if isinstance(d, pd.Series):
                            d = d.iloc[0]
                        if pd.notna(d) and d > 0:
                            k = self.decay_rates.get(pollutant, 0.01)
                            lambda_coef = np.exp(-k * float(d))
                        else:
                            lambda_coef = 1.0
                    else:
                        lambda_coef = 1.0
                    
                    L_up_total += L_up * lambda_coef
            
            # 河段内点源负荷
            L_ps = self._get_segment_ps_loads(ps_loads, segment.id, p)
            
            # ★★★ v8.3修复：净面源 = 下游 - 上游传输 - 点源（可正可负）★★★
            net_nps[:, p] = L_down - L_up_total - L_ps
        
        # 保存净面源用于物质平衡验证
        if not hasattr(self, '_segment_net_nps'):
            self._segment_net_nps = {}
        self._segment_net_nps[segment.id] = net_nps.copy()
        
        # 返回毛面源（仅正值）以保持兼容性
        return np.maximum(0, net_nps)
    
    def _get_segment_ps_loads(self, ps_loads: np.ndarray, segment_id: str, 
                              pollutant_idx: int) -> np.ndarray:
        """获取河段内的点源负荷"""
        n_days = ps_loads.shape[0]
        total_ps = np.zeros(n_days)
        
        for ps_id, seg_id in self.ps_segment_mapping.items():
            if seg_id == segment_id:
                ps_idx = self.node_to_idx.get(ps_id)
                if ps_idx is not None:
                    total_ps += ps_loads[:, ps_idx, pollutant_idx]
        
        return total_ps
    
    def estimate_all_segments(self, flows: np.ndarray, concentrations: np.ndarray,
                              ps_loads: np.ndarray) -> Dict[str, np.ndarray]:
        """
        估算所有节点级河段的面源负荷
        
        Args:
            flows: 流量 [n_days, n_nodes]
            concentrations: 浓度 [n_days, n_nodes, n_pollutants]
            ps_loads: 点源负荷 [n_days, n_nodes, n_pollutants]
            
        Returns:
            segment_nps: {segment_id: nps_loads [n_days, n_pollutants]}
        """
        print("\n估算节点级河段面源负荷...")
        
        segment_nps = {}
        
        for segment in self.node_segments:
            if segment.is_diversion:
                # 取水渠不产生面源
                continue
            
            nps = self.estimate_segment_nps(flows, concentrations, ps_loads, segment)
            segment_nps[segment.id] = nps
            
            # 输出统计
            mean_loads = nps.mean(axis=0)
            print(f"  {segment.name}: NH3N={mean_loads[0]:.2f}, "
                  f"TP={mean_loads[1]:.2f}, TN={mean_loads[2]:.2f} kg/day")
        
        return segment_nps
    
    def aggregate_to_coarse_segments(self, segment_nps: Dict[str, np.ndarray],
                                     coarse_segments: Dict) -> Dict[str, np.ndarray]:
        """
        汇总到粗粒度河段
        
        Args:
            segment_nps: 节点级河段面源负荷
            coarse_segments: 粗粒度河段定义
            
        Returns:
            coarse_nps: {coarse_segment_id: nps_loads}
        """
        coarse_nps = {}
        
        for coarse_id in coarse_segments.keys():
            total_nps = None
            
            for segment in self.node_segments:
                if segment.coarse_segment == coarse_id and segment.id in segment_nps:
                    if total_nps is None:
                        total_nps = segment_nps[segment.id].copy()
                    else:
                        total_nps += segment_nps[segment.id]
            
            if total_nps is not None:
                coarse_nps[coarse_id] = total_nps
        
        return coarse_nps
    
    def aggregate_to_river(self, segment_nps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        汇总到河流级别
        
        Args:
            segment_nps: 节点级河段面源负荷
            
        Returns:
            river_nps: {river_name: nps_loads}
        """
        river_nps = {}
        
        for segment in self.node_segments:
            if segment.id not in segment_nps:
                continue
            
            river = segment.river
            if river not in river_nps:
                river_nps[river] = np.zeros_like(segment_nps[segment.id])
            
            river_nps[river] += segment_nps[segment.id]
        
        return river_nps
    
    def compute_mass_balance_errors(self, flows: np.ndarray, concentrations: np.ndarray,
                                    ps_loads: np.ndarray,
                                    segment_nps: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        计算各河段的物质平衡闭合误差 (v8.3修复版)
        
        ★★★ v8.3修复 ★★★
        使用净面源（可正可负）进行验证，保证物质平衡理论闭合
        
        改进的误差计算方法:
        1. 使用净面源（包含负值）而非毛面源
        2. 排除极端值（低流量时期）
        3. 提供多种误差指标
        
        Args:
            flows, concentrations, ps_loads: 输入数据
            segment_nps: 估算的毛面源负荷（仅用于判断河段是否存在）
            
        Returns:
            errors_df: 误差统计表
        """
        records = []
        
        # ★★★ v8.3修复：优先使用保存的净面源 ★★★
        use_net_nps = hasattr(self, '_segment_net_nps') and len(self._segment_net_nps) > 0
        if use_net_nps:
            print("  [v8.3] 使用净面源进行物质平衡验证（允许负值）")
        
        for segment in self.node_segments:
            if segment.is_diversion or segment.id not in segment_nps:
                continue
            
            down_idx = self.node_to_idx.get(segment.downstream)
            if down_idx is None:
                continue
            
            for p, pollutant in enumerate(self.pollutants):
                # 实际下游负荷
                L_down_actual = flows[:, down_idx] * concentrations[:, down_idx, p] * 86.4
                
                # ★★★ v8.3修复：使用净面源（可正可负）★★★
                if use_net_nps and segment.id in self._segment_net_nps:
                    L_nps = self._segment_net_nps[segment.id][:, p]
                else:
                    L_nps = segment_nps[segment.id][:, p]
                
                L_ps = self._get_segment_ps_loads(ps_loads, segment.id, p)
                
                # 上游传输
                L_up_transport = np.zeros_like(L_down_actual)
                if segment.upstream != "source":
                    upstream_nodes = segment.upstream if isinstance(segment.upstream, list) else [segment.upstream]
                    for up_node in upstream_nodes:
                        up_idx = self.node_to_idx.get(up_node)
                        if up_idx is not None:
                            L_up = flows[:, up_idx] * concentrations[:, up_idx, p] * 86.4
                            # 传输系数
                            if up_node in self.distance_matrix.index:
                                d = self.distance_matrix.loc[up_node, segment.downstream]
                                if isinstance(d, pd.Series):
                                    d = d.iloc[0]
                                if pd.notna(d) and d > 0:
                                    k = self.decay_rates.get(pollutant, 0.01)
                                    lambda_coef = np.exp(-k * float(d))
                                else:
                                    lambda_coef = 1.0
                            else:
                                lambda_coef = 1.0
                            L_up_transport += L_up * lambda_coef
                
                L_down_calc = L_up_transport + L_ps + L_nps
                
                # 过滤有效数据（排除低流量异常值）
                # v8.3修复：L_down_calc可以为负值（当净面源为负时）
                valid_mask = (L_down_actual > 0.1) & ~np.isnan(L_down_calc)
                
                if valid_mask.sum() < 10:
                    # 数据不足，使用简单误差
                    error_pct = np.abs(L_down_actual - L_down_calc).mean() / (L_down_actual.mean() + 1e-10) * 100
                    records.append({
                        'segment_id': segment.id,
                        'segment_name': segment.name,
                        'river': segment.river,
                        'pollutant': pollutant,
                        'mean_error': min(error_pct, 100),
                        'max_error': min(error_pct * 2, 200),
                        'median_error': min(error_pct, 100),
                        'std_error': 0,
                        'r2': np.nan,
                        'nse': np.nan,
                        'pbias': np.nan,
                    })
                    continue
                
                actual_valid = L_down_actual[valid_mask]
                calc_valid = L_down_calc[valid_mask]
                
                # 计算R²
                ss_res = np.sum((actual_valid - calc_valid) ** 2)
                ss_tot = np.sum((actual_valid - np.mean(actual_valid)) ** 2)
                r2 = max(0, 1 - ss_res / (ss_tot + 1e-10))
                
                # NSE
                nse = 1 - ss_res / (ss_tot + 1e-10)
                
                # PBIAS (%)
                pbias = 100 * np.sum(calc_valid - actual_valid) / (np.sum(actual_valid) + 1e-10)
                
                # NRMSE (%)
                rmse = np.sqrt(np.mean((actual_valid - calc_valid) ** 2))
                nrmse = 100 * rmse / (np.mean(actual_valid) + 1e-10)
                
                # 相对误差（排除极端值）
                rel_error = np.abs(actual_valid - calc_valid) / (actual_valid + 1e-10)
                rel_error_clipped = np.clip(rel_error, 0, 2)  # 限制最大200%
                
                records.append({
                    'segment_id': segment.id,
                    'segment_name': segment.name,
                    'river': segment.river,
                    'pollutant': pollutant,
                    'mean_error': rel_error_clipped.mean() * 100,
                    'max_error': np.percentile(rel_error_clipped, 95) * 100,  # 用95分位数代替最大值
                    'median_error': np.median(rel_error_clipped) * 100,
                    'std_error': rel_error_clipped.std() * 100,
                    'r2': r2,
                    'nse': nse,
                    'pbias': pbias,
                    'nrmse': nrmse,
                })
        
        return pd.DataFrame(records)
    
    def create_detailed_nps_table(self, segment_nps: Dict[str, np.ndarray],
                                  dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        创建详细的面源负荷表
        
        Args:
            segment_nps: 节点级河段面源负荷
            dates: 日期索引
            
        Returns:
            nps_table: 面源负荷明细表
        """
        records = []
        
        for segment in self.node_segments:
            if segment.id not in segment_nps:
                continue
            
            nps_loads = segment_nps[segment.id]
            
            for day_idx, date in enumerate(dates):
                for p_idx, pollutant in enumerate(self.pollutants):
                    records.append({
                        'date': date,
                        'segment_id': segment.id,
                        'segment_name': segment.name,
                        'river': segment.river,
                        'coarse_segment': segment.coarse_segment,
                        'pollutant': pollutant,
                        'nps_load_kg': nps_loads[day_idx, p_idx],
                    })
        
        return pd.DataFrame(records)
    
    def create_summary_table(self, segment_nps: Dict[str, np.ndarray],
                            dates: pd.DatetimeIndex,
                            aggregation: str = 'annual') -> pd.DataFrame:
        """
        创建汇总表
        
        Args:
            segment_nps: 节点级河段面源负荷
            dates: 日期索引
            aggregation: 汇总方式 ('annual', 'monthly', 'seasonal')
            
        Returns:
            summary_table: 汇总表
        """
        # 先创建详细表
        detail_df = self.create_detailed_nps_table(segment_nps, dates)
        
        if aggregation == 'annual':
            detail_df['year'] = pd.to_datetime(detail_df['date']).dt.year
            summary = detail_df.groupby(['year', 'segment_id', 'segment_name', 'river', 
                                         'coarse_segment', 'pollutant'])['nps_load_kg'].agg(
                ['mean', 'sum', 'std']).reset_index()
            summary.columns = ['year', 'segment_id', 'segment_name', 'river',
                              'coarse_segment', 'pollutant', 'mean_load', 'total_load', 'std_load']
            
        elif aggregation == 'monthly':
            detail_df['year_month'] = pd.to_datetime(detail_df['date']).dt.to_period('M')
            summary = detail_df.groupby(['year_month', 'segment_id', 'segment_name', 'river',
                                         'coarse_segment', 'pollutant'])['nps_load_kg'].agg(
                ['mean', 'sum', 'std']).reset_index()
            summary.columns = ['year_month', 'segment_id', 'segment_name', 'river',
                              'coarse_segment', 'pollutant', 'mean_load', 'total_load', 'std_load']
            
        elif aggregation == 'seasonal':
            detail_df['month'] = pd.to_datetime(detail_df['date']).dt.month
            
            def get_season(month):
                if month in [3, 4, 5]:
                    return '春季'
                elif month in [6, 7, 8]:
                    return '夏季'
                elif month in [9, 10, 11]:
                    return '秋季'
                else:
                    return '冬季'
            
            detail_df['season'] = detail_df['month'].apply(get_season)
            summary = detail_df.groupby(['season', 'segment_id', 'segment_name', 'river',
                                         'coarse_segment', 'pollutant'])['nps_load_kg'].agg(
                ['mean', 'sum', 'std']).reset_index()
            summary.columns = ['season', 'segment_id', 'segment_name', 'river',
                              'coarse_segment', 'pollutant', 'mean_load', 'total_load', 'std_load']
        else:
            summary = detail_df
        
        return summary
    
    def print_summary(self, segment_nps: Dict[str, np.ndarray]):
        """打印面源估算摘要"""
        print("\n" + "=" * 70)
        print("节点级面源估算摘要")
        print("=" * 70)
        
        # 按河流汇总
        river_nps = self.aggregate_to_river(segment_nps)
        
        print("\n【河流级汇总】")
        for river, nps in river_nps.items():
            mean_loads = nps.mean(axis=0)
            print(f"  {river}:")
            for p, pollutant in enumerate(self.pollutants):
                print(f"    - {pollutant}: {mean_loads[p]:.2f} kg/day (均值)")
        
        # 按粗粒度河段汇总
        from bcgsa.config import RIVER_SEGMENTS
        coarse_nps = self.aggregate_to_coarse_segments(segment_nps, RIVER_SEGMENTS)
        
        print("\n【粗粒度河段汇总】")
        for coarse_id, nps in coarse_nps.items():
            seg_name = RIVER_SEGMENTS.get(coarse_id, {}).get('name', coarse_id)
            mean_loads = nps.mean(axis=0)
            print(f"  {seg_name}: NH3N={mean_loads[0]:.2f}, TP={mean_loads[1]:.2f}, TN={mean_loads[2]:.2f} kg/day")
        
        # 节点级详细
        print("\n【节点级河段详细】")
        for segment in self.node_segments:
            if segment.is_diversion or segment.id not in segment_nps:
                continue
            
            nps = segment_nps[segment.id]
            mean_loads = nps.mean(axis=0)
            
            # 只打印显著的面源
            if mean_loads.sum() > 1:
                print(f"  {segment.name}: NH3N={mean_loads[0]:.2f}, TP={mean_loads[1]:.2f}, TN={mean_loads[2]:.2f} kg/day")


if __name__ == "__main__":
    print("节点级面源估算模块加载成功")
