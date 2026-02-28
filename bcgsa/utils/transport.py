# -*- coding: utf-8 -*-
"""
================================================================================
伊洛河流域污染溯源模型 - 传输系数模块
Yiluo River Basin Pollution Source Apportionment - Transport Coefficient Module
================================================================================

计算污染物从各源到目标断面的传输系数
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class TransportModel:
    """传输系数模型"""
    
    def __init__(self, distance_matrix: pd.DataFrame, node_list: List[str],
                 decay_rates: Dict[str, float], pollutants: List[str],
                 target_station: str = 'QiLiPu'):
        """
        初始化传输模型
        
        Args:
            distance_matrix: 河道距离矩阵 (DataFrame)
            node_list: 节点列表
            decay_rates: 降解系数 {pollutant: k (1/km)}
            pollutants: 污染物列表
            target_station: 目标断面
        """
        self.distance_matrix = distance_matrix
        self.node_list = node_list
        self.node_to_idx = {node: i for i, node in enumerate(node_list)}
        self.decay_rates = decay_rates
        self.pollutants = pollutants
        self.target_station = target_station
        
        # 计算到目标站点的距离
        self._compute_distances_to_target()
        
    def _compute_distances_to_target(self):
        """计算各节点到目标站点的距离"""
        self.distances_to_target = {}
        
        for node in self.node_list:
            if node == self.target_station:
                self.distances_to_target[node] = 0.0
            elif node in self.distance_matrix.index and self.target_station in self.distance_matrix.columns:
                d = self.distance_matrix.loc[node, self.target_station]
                if pd.notna(d):
                    self.distances_to_target[node] = d
                else:
                    self.distances_to_target[node] = np.inf
            else:
                self.distances_to_target[node] = np.inf
    
    def compute_transport_coefficients(self, flows: Optional[np.ndarray] = None,
                                       use_velocity: bool = True) -> Dict[str, np.ndarray]:
        """
        计算传输系数
        
        传输系数 λ = exp(-k × d / v) 或 λ = exp(-k × d) (无流速时)
        
        Args:
            flows: 流量数据 [n_days, n_nodes] (m³/s)，用于计算流速
            use_velocity: 是否使用流速修正
            
        Returns:
            transport_coefs: {pollutant: [n_days, n_nodes]} 传输系数
        """
        n_nodes = len(self.node_list)
        n_days = flows.shape[0] if flows is not None else 1
        
        transport_coefs = {}
        
        for pollutant in self.pollutants:
            k = self.decay_rates.get(pollutant, 0.01)
            coefs = np.zeros((n_days, n_nodes))
            
            for i, node in enumerate(self.node_list):
                d = self.distances_to_target.get(node, np.inf)
                
                if d == 0:
                    # 目标站点本身
                    coefs[:, i] = 1.0
                elif d < np.inf:
                    if use_velocity and flows is not None:
                        # 使用流速修正: λ = exp(-k × d / v)
                        # 流速估算: v = Q / A, 假设断面面积与流量的0.5次方成正比
                        Q = np.maximum(flows[:, i], 0.1)  # 避免零流量
                        # 假设典型流速 2-5 km/h = 48-120 km/day
                        v = 2.0 * (Q ** 0.3) * 24  # 简化的流速估算 (km/day)
                        travel_time = d / v  # days
                        # 使用时间衰减
                        coefs[:, i] = np.exp(-k * d)  # 仍用距离衰减，但可以调整
                    else:
                        # 仅距离衰减: λ = exp(-k × d)
                        coefs[:, i] = np.exp(-k * d)
                else:
                    # 不可达
                    coefs[:, i] = 0.0
            
            transport_coefs[pollutant] = coefs
        
        return transport_coefs
    
    def compute_static_coefficients(self) -> Dict[str, np.ndarray]:
        """
        计算静态传输系数（不考虑时变流速）
        
        Returns:
            transport_coefs: {pollutant: [n_nodes]} 传输系数
        """
        n_nodes = len(self.node_list)
        transport_coefs = {}
        
        for pollutant in self.pollutants:
            k = self.decay_rates.get(pollutant, 0.01)
            coefs = np.zeros(n_nodes)
            
            for i, node in enumerate(self.node_list):
                d = self.distances_to_target.get(node, np.inf)
                
                if d == 0:
                    coefs[i] = 1.0
                elif d < np.inf:
                    coefs[i] = np.exp(-k * d)
                else:
                    coefs[i] = 0.0
            
            transport_coefs[pollutant] = coefs
        
        return transport_coefs
    
    def get_distance_to_target(self, node: str) -> float:
        """获取节点到目标站点的距离"""
        return self.distances_to_target.get(node, np.inf)
    
    def get_reachable_nodes(self) -> List[str]:
        """获取可到达目标站点的节点列表"""
        return [node for node, d in self.distances_to_target.items() if d < np.inf]
    
    def print_transport_summary(self):
        """打印传输系数摘要"""
        print("\n" + "=" * 60)
        print("传输系数摘要")
        print("=" * 60)
        
        # 计算静态系数
        static_coefs = self.compute_static_coefficients()
        
        # 按类型统计
        node_types = {
            'WQ': [],
            'H': [],
            'PS': [],
        }
        
        for node in self.node_list:
            if node.startswith('YL'):
                node_types['PS'].append(node)
            elif node in ['LingKou', 'LuoNingChangShui', 'GaoYaZhai', 'BaiMaSi',
                         'TanTou', 'LongMenDaQiao', 'QiLiPu']:
                node_types['WQ'].append(node)
            else:
                node_types['H'].append(node)
        
        for pollutant in self.pollutants:
            print(f"\n【{pollutant}】 (k = {self.decay_rates.get(pollutant, 0.01):.4f} 1/km)")
            coefs = static_coefs[pollutant]
            
            for ntype, nodes in node_types.items():
                if not nodes:
                    continue
                
                type_coefs = [coefs[self.node_to_idx[n]] for n in nodes 
                              if n in self.node_to_idx and self.node_to_idx[n] < len(coefs)]
                
                if type_coefs:
                    valid_coefs = [c for c in type_coefs if c > 0]
                    if valid_coefs:
                        print(f"  {ntype}: 平均λ = {np.mean(valid_coefs):.4f}, "
                              f"范围 [{min(valid_coefs):.4f}, {max(valid_coefs):.4f}]")


class ArrivedLoadCalculator:
    """到达负荷计算器"""
    
    def __init__(self, transport_model: TransportModel, node_list: List[str],
                 pollutants: List[str]):
        """
        初始化到达负荷计算器
        
        Args:
            transport_model: 传输模型
            node_list: 节点列表
            pollutants: 污染物列表
        """
        self.transport_model = transport_model
        self.node_list = node_list
        self.node_to_idx = {node: i for i, node in enumerate(node_list)}
        self.pollutants = pollutants
        
    def compute_arrived_loads(self, source_loads: np.ndarray,
                              transport_coefs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        计算到达负荷
        
        到达负荷 = 源负荷 × 传输系数
        
        Args:
            source_loads: 源负荷 [n_days, n_nodes, n_pollutants] (kg/day)
            transport_coefs: 传输系数 {pollutant: [n_days, n_nodes] 或 [n_nodes]}
            
        Returns:
            arrived_loads: {pollutant: [n_days, n_nodes]} 到达负荷
        """
        n_days, n_nodes, n_pollutants = source_loads.shape
        arrived_loads = {}
        
        for p_idx, pollutant in enumerate(self.pollutants):
            coefs = transport_coefs.get(pollutant)
            if coefs is None:
                continue
            
            # 处理维度
            if coefs.ndim == 1:
                coefs = np.tile(coefs, (n_days, 1))
            
            # 到达负荷 = 源负荷 × 传输系数
            arrived = source_loads[:, :, p_idx] * coefs
            arrived_loads[pollutant] = arrived
        
        return arrived_loads
    
    def aggregate_by_source_type(self, arrived_loads: Dict[str, np.ndarray],
                                 source_types: Dict[str, str]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        按源类型汇总到达负荷
        
        Args:
            arrived_loads: 到达负荷 {pollutant: [n_days, n_nodes]}
            source_types: 节点类型 {node: type}
            
        Returns:
            aggregated: {pollutant: {source_type: [n_days]}}
        """
        aggregated = {}
        
        for pollutant, loads in arrived_loads.items():
            aggregated[pollutant] = {}
            
            type_loads = {}
            for node, stype in source_types.items():
                idx = self.node_to_idx.get(node)
                if idx is None or idx >= loads.shape[1]:
                    continue
                
                if stype not in type_loads:
                    type_loads[stype] = np.zeros(loads.shape[0])
                type_loads[stype] += loads[:, idx]
            
            aggregated[pollutant] = type_loads
        
        return aggregated


class ContributionCalculator:
    """贡献率计算器"""
    
    def __init__(self, node_list: List[str], pollutants: List[str],
                 source_types: Dict[str, str]):
        """
        初始化贡献率计算器
        
        Args:
            node_list: 节点列表
            pollutants: 污染物列表
            source_types: 源类型 {node: type}
        """
        self.node_list = node_list
        self.node_to_idx = {node: i for i, node in enumerate(node_list)}
        self.pollutants = pollutants
        self.source_types = source_types
        
    def compute_contributions(self, arrived_loads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        计算贡献率
        
        贡献率 = 到达负荷_i / Σ到达负荷
        
        Args:
            arrived_loads: 到达负荷 {pollutant: [n_days, n_nodes]}
            
        Returns:
            contributions: {pollutant: [n_days, n_nodes]} 贡献率
        """
        contributions = {}
        
        for pollutant, loads in arrived_loads.items():
            # 总到达负荷
            total = loads.sum(axis=1, keepdims=True)
            # 避免除零
            total = np.maximum(total, 1e-10)
            # 贡献率
            contrib = loads / total
            contributions[pollutant] = contrib
        
        return contributions
    
    def compute_type_contributions(self, contributions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        按类型汇总贡献率
        
        Args:
            contributions: 节点贡献率
            
        Returns:
            type_contributions: {pollutant: {type: [n_days]}}
        """
        type_contributions = {}
        
        for pollutant, contrib in contributions.items():
            type_contributions[pollutant] = {}
            
            for stype in set(self.source_types.values()):
                type_indices = [self.node_to_idx[n] for n, t in self.source_types.items() 
                               if t == stype and n in self.node_to_idx]
                
                if type_indices:
                    type_contrib = contrib[:, type_indices].sum(axis=1)
                    type_contributions[pollutant][stype] = type_contrib
        
        return type_contributions
    
    def create_contribution_table(self, contributions: Dict[str, np.ndarray],
                                  dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        创建贡献率表
        
        Args:
            contributions: 节点贡献率
            dates: 日期索引
            
        Returns:
            table: 贡献率DataFrame
        """
        records = []
        
        for pollutant, contrib in contributions.items():
            for day_idx, date in enumerate(dates):
                for node_idx, node in enumerate(self.node_list):
                    if node_idx >= contrib.shape[1]:
                        continue
                    
                    stype = self.source_types.get(node, 'unknown')
                    
                    records.append({
                        'date': date,
                        'pollutant': pollutant,
                        'node': node,
                        'source_type': stype,
                        'contribution': contrib[day_idx, node_idx],
                    })
        
        return pd.DataFrame(records)
    
    def create_summary_table(self, type_contributions: Dict[str, Dict[str, np.ndarray]],
                            dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        创建类型贡献率汇总表
        """
        records = []
        
        for pollutant, type_contribs in type_contributions.items():
            for stype, contrib in type_contribs.items():
                # 日均值
                for day_idx, date in enumerate(dates):
                    records.append({
                        'date': date,
                        'pollutant': pollutant,
                        'source_type': stype,
                        'contribution': contrib[day_idx],
                    })
        
        return pd.DataFrame(records)
    
    def compute_annual_summary(self, type_contributions: Dict[str, Dict[str, np.ndarray]],
                               dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        计算年度汇总
        """
        records = []
        
        for pollutant, type_contribs in type_contributions.items():
            for stype, contrib in type_contribs.items():
                # 按年汇总
                df = pd.DataFrame({'date': dates, 'contribution': contrib})
                df['year'] = df['date'].dt.year
                
                annual = df.groupby('year')['contribution'].mean()
                
                for year, mean_contrib in annual.items():
                    records.append({
                        'year': year,
                        'pollutant': pollutant,
                        'source_type': stype,
                        'mean_contribution': mean_contrib,
                    })
        
        return pd.DataFrame(records)


if __name__ == "__main__":
    print("传输系数模块加载成功")
