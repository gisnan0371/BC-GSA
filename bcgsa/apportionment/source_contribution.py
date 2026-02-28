# -*- coding: utf-8 -*-
"""
================================================================================
伊洛河流域污染溯源模型 - 贡献解析模块
Yiluo River Basin Pollution Source Apportionment - Source Contribution Module
================================================================================

整合所有模块，计算各源对目标断面的污染贡献
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class SourceApportionment:
    """污染源贡献解析"""
    
    def __init__(self, topology: Dict, river_segments: Dict,
                 pollutants: List[str], decay_rates: Dict[str, float],
                 target_station: str = 'QiLiPu',
                 node_segments: Optional[List[Dict]] = None,
                 ps_segment_mapping: Optional[Dict[str, str]] = None):
        """
        初始化贡献解析器
        
        Args:
            topology: 拓扑信息
            river_segments: 河段定义
            pollutants: 目标污染物
            decay_rates: 降解系数
            target_station: 目标断面
            node_segments: 节点级河段定义（可选）
            ps_segment_mapping: 点源所属河段映射（可选）
        """
        self.topology = topology
        self.river_segments = river_segments
        self.pollutants = pollutants
        self.decay_rates = decay_rates
        self.target_station = target_station
        self.node_segments = node_segments or []
        self.ps_segment_mapping = ps_segment_mapping or {}
        
        self.node_list = topology.get('node_list', [])
        self.node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        self.distance_to_target = topology.get('distance_to_target', {})
        
        # 源分类
        self._classify_sources()
        
    def _classify_sources(self):
        """分类源节点"""
        self.source_types = {}
        self.sources_by_type = {
            'point_source': [],
            'nps': [],
            'upstream': [],
        }
        
        # 节点级面源河段
        self.nps_segments = {}
        for seg in self.node_segments:
            seg_id = seg.get('id', '')
            if not seg.get('is_diversion', False):
                self.nps_segments[seg_id] = seg
        
        for node in self.node_list:
            if node.startswith('YL'):
                self.source_types[node] = 'point_source'
                self.sources_by_type['point_source'].append(node)
            elif node in ['LingKou', 'TanTou', 'LuanChuan']:
                # 上游边界站点
                self.source_types[node] = 'upstream'
                self.sources_by_type['upstream'].append(node)
            else:
                self.source_types[node] = 'other'
        
        # 为每个节点级面源河段创建虚拟源
        for seg_id, seg in self.nps_segments.items():
            seg_name = f"NPS_{seg_id}"
            self.source_types[seg_name] = 'nps'
            self.sources_by_type['nps'].append(seg_name)
    
    def compute_source_loads(self, flows: np.ndarray, concentrations: np.ndarray,
                             ps_loads: np.ndarray) -> np.ndarray:
        """
        计算各源负荷
        
        Args:
            flows: 流量 [n_days, n_nodes] (m³/s)
            concentrations: 浓度 [n_days, n_nodes, n_pollutants] (mg/L)
            ps_loads: 点源负荷 [n_days, n_nodes, n_pollutants] (kg/day)
            
        Returns:
            source_loads: [n_days, n_nodes, n_pollutants] (kg/day)
        """
        n_days, n_nodes = flows.shape
        n_pollutants = len(self.pollutants)
        
        source_loads = np.zeros((n_days, n_nodes, n_pollutants))
        
        for i, node in enumerate(self.node_list):
            stype = self.source_types.get(node, 'other')
            
            if stype == 'point_source':
                # 点源：直接使用点源负荷
                source_loads[:, i, :] = ps_loads[:, i, :]
            elif stype == 'upstream':
                # 上游来水：L = Q × C × 86.4
                for p in range(n_pollutants):
                    source_loads[:, i, p] = flows[:, i] * concentrations[:, i, p] * 86.4
            else:
                # 其他节点：暂不作为源
                pass
        
        return source_loads
    
    def compute_transport_coefficients(self, flows: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        计算传输系数
        
        Args:
            flows: 流量 [n_days, n_nodes]
            
        Returns:
            transport_coefs: {pollutant: [n_nodes]}
        """
        n_nodes = len(self.node_list)
        transport_coefs = {}
        
        for pollutant in self.pollutants:
            k = self.decay_rates.get(pollutant, 0.01)
            coefs = np.zeros(n_nodes)
            
            for i, node in enumerate(self.node_list):
                d = self.distance_to_target.get(node, np.inf)
                
                if d == 0:
                    coefs[i] = 1.0
                elif d < np.inf:
                    coefs[i] = np.exp(-k * d)
                else:
                    coefs[i] = 0.0
            
            transport_coefs[pollutant] = coefs
        
        return transport_coefs
    
    def compute_arrived_loads(self, source_loads: np.ndarray,
                              transport_coefs: Dict[str, np.ndarray]) -> np.ndarray:
        """
        计算到达负荷
        
        Args:
            source_loads: 源负荷 [n_days, n_nodes, n_pollutants]
            transport_coefs: 传输系数 {pollutant: [n_nodes]}
            
        Returns:
            arrived_loads: [n_days, n_nodes, n_pollutants]
        """
        n_days, n_nodes, n_pollutants = source_loads.shape
        arrived_loads = np.zeros_like(source_loads)
        
        for p, pollutant in enumerate(self.pollutants):
            coefs = transport_coefs.get(pollutant, np.ones(n_nodes))
            arrived_loads[:, :, p] = source_loads[:, :, p] * coefs
        
        return arrived_loads
    
    def compute_contributions(self, arrived_loads: np.ndarray) -> np.ndarray:
        """
        计算贡献率
        
        Args:
            arrived_loads: 到达负荷 [n_days, n_nodes, n_pollutants]
            
        Returns:
            contributions: [n_days, n_nodes, n_pollutants]
        """
        # 总到达负荷
        total = arrived_loads.sum(axis=1, keepdims=True)
        total = np.maximum(total, 1e-10)
        
        contributions = arrived_loads / total
        
        return contributions
    
    def run_apportionment(self, flows: np.ndarray, concentrations: np.ndarray,
                          ps_loads: np.ndarray, nps_loads: Dict[str, np.ndarray],
                          dates: pd.DatetimeIndex) -> Dict:
        """
        运行完整的贡献解析
        
        Args:
            flows: 流量 [n_days, n_nodes]
            concentrations: 浓度 [n_days, n_nodes, n_pollutants]
            ps_loads: 点源负荷 [n_days, n_nodes, n_pollutants]
            nps_loads: 面源负荷 {segment: [n_days, n_pollutants]}
            dates: 日期索引
            
        Returns:
            results: 解析结果
        """
        n_days = len(dates)
        n_nodes = len(self.node_list)
        n_pollutants = len(self.pollutants)
        
        print("\n" + "=" * 60)
        print("运行污染源贡献解析...")
        print("=" * 60)
        
        # 1. 计算源负荷
        print("\n[1/4] 计算源负荷...")
        source_loads = self.compute_source_loads(flows, concentrations, ps_loads)
        
        # 添加面源负荷
        # 将面源分配到对应的虚拟节点（简化处理：暂时不添加）
        
        # 2. 计算传输系数
        print("[2/4] 计算传输系数...")
        transport_coefs = self.compute_transport_coefficients(flows)
        
        # 3. 计算到达负荷
        print("[3/4] 计算到达负荷...")
        arrived_loads = self.compute_arrived_loads(source_loads, transport_coefs)
        
        # 4. 计算贡献率
        print("[4/4] 计算贡献率...")
        contributions = self.compute_contributions(arrived_loads)
        
        # 汇总结果
        results = {
            'source_loads': source_loads,
            'transport_coefs': transport_coefs,
            'arrived_loads': arrived_loads,
            'contributions': contributions,
            'dates': dates,
            'node_list': self.node_list,
            'pollutants': self.pollutants,
        }
        
        # 按类型汇总
        results['type_contributions'] = self._aggregate_by_type(contributions)
        
        print("\n解析完成！")
        
        return results
    
    def _aggregate_by_type(self, contributions: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """按源类型汇总贡献率"""
        n_days, n_nodes, n_pollutants = contributions.shape
        
        type_contributions = {}
        
        for p, pollutant in enumerate(self.pollutants):
            type_contributions[pollutant] = {}
            
            for stype, nodes in self.sources_by_type.items():
                if not nodes:
                    continue
                
                indices = [self.node_to_idx[n] for n in nodes if n in self.node_to_idx]
                if indices:
                    type_contrib = contributions[:, indices, p].sum(axis=1)
                    type_contributions[pollutant][stype] = type_contrib
        
        return type_contributions
    
    def create_contribution_report(self, results: Dict, output_dir: Path) -> None:
        """
        生成贡献报告
        
        Args:
            results: 解析结果
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dates = results['dates']
        contributions = results['contributions']
        type_contributions = results['type_contributions']
        
        print("\n生成贡献报告...")
        
        # 1. 日贡献表
        print("  - 生成日贡献表...")
        daily_records = []
        
        for day_idx, date in enumerate(dates):
            for p_idx, pollutant in enumerate(self.pollutants):
                for node_idx, node in enumerate(self.node_list):
                    stype = self.source_types.get(node, 'other')
                    contrib = contributions[day_idx, node_idx, p_idx]
                    
                    if contrib > 0.001:  # 只记录贡献 > 0.1%
                        daily_records.append({
                            'date': date,
                            'pollutant': pollutant,
                            'source': node,
                            'source_type': stype,
                            'contribution': contrib,
                        })
        
        daily_df = pd.DataFrame(daily_records)
        daily_df.to_csv(output_dir / 'daily_contributions.csv', index=False)
        
        # 2. 类型汇总表
        print("  - 生成类型汇总表...")
        type_records = []
        
        for pollutant, type_contribs in type_contributions.items():
            for stype, contribs in type_contribs.items():
                for day_idx, date in enumerate(dates):
                    type_records.append({
                        'date': date,
                        'pollutant': pollutant,
                        'source_type': stype,
                        'contribution': contribs[day_idx],
                    })
        
        type_df = pd.DataFrame(type_records)
        type_df.to_csv(output_dir / 'type_contributions.csv', index=False)
        
        # 3. 月均汇总
        print("  - 生成月均汇总...")
        type_df['year_month'] = pd.to_datetime(type_df['date']).dt.to_period('M')
        monthly = type_df.groupby(['year_month', 'pollutant', 'source_type'])['contribution'].mean()
        monthly.to_csv(output_dir / 'monthly_contributions.csv')
        
        # 4. 年均汇总
        print("  - 生成年均汇总...")
        type_df['year'] = pd.to_datetime(type_df['date']).dt.year
        annual = type_df.groupby(['year', 'pollutant', 'source_type'])['contribution'].mean()
        annual.to_csv(output_dir / 'annual_contributions.csv')
        
        # 5. 点源排名
        print("  - 生成点源排名...")
        ps_nodes = self.sources_by_type.get('point_source', [])
        
        for p_idx, pollutant in enumerate(self.pollutants):
            ps_contribs = {}
            for node in ps_nodes:
                idx = self.node_to_idx.get(node)
                if idx is not None:
                    ps_contribs[node] = contributions[:, idx, p_idx].mean()
            
            # 排序
            sorted_ps = sorted(ps_contribs.items(), key=lambda x: x[1], reverse=True)
            
            ranking_df = pd.DataFrame(sorted_ps, columns=['source', 'mean_contribution'])
            ranking_df['rank'] = range(1, len(ranking_df) + 1)
            ranking_df.to_csv(output_dir / f'ps_ranking_{pollutant}.csv', index=False)
        
        print(f"\n报告已保存至: {output_dir}")
    
    def print_summary(self, results: Dict):
        """打印结果摘要"""
        type_contributions = results['type_contributions']
        
        print("\n" + "=" * 60)
        print("贡献解析结果摘要")
        print("=" * 60)
        
        for pollutant in self.pollutants:
            print(f"\n【{pollutant}】")
            
            type_contribs = type_contributions.get(pollutant, {})
            for stype, contribs in type_contribs.items():
                mean_contrib = contribs.mean() * 100
                std_contrib = contribs.std() * 100
                print(f"  - {stype}: {mean_contrib:.2f}% (±{std_contrib:.2f}%)")


class ExtremeEventAnalyzer:
    """极端事件分析器"""
    
    def __init__(self, results: Dict, precip_data: pd.DataFrame,
                 percentile: float = 95):
        """
        初始化极端事件分析器
        
        Args:
            results: 贡献解析结果
            precip_data: 降水数据
            percentile: 极端事件阈值百分位
        """
        self.results = results
        self.precip_data = precip_data
        self.percentile = percentile
        
        # 识别极端事件
        self._identify_extreme_events()
    
    def _identify_extreme_events(self):
        """识别极端降水事件"""
        precip = self.precip_data['basin_precipitation'].values
        
        # 计算阈值
        threshold = np.percentile(precip[precip > 0], self.percentile)
        
        # 标记极端事件
        self.is_extreme = precip >= threshold
        self.threshold = threshold
        
        # 统计
        n_extreme = self.is_extreme.sum()
        print(f"极端事件识别: P{self.percentile}阈值={threshold:.2f}mm, 共{n_extreme}天")
    
    def compare_extreme_vs_normal(self) -> pd.DataFrame:
        """比较极端事件与正常时期的贡献差异"""
        contributions = self.results['contributions']
        type_contributions = self.results['type_contributions']
        
        records = []
        
        for pollutant, type_contribs in type_contributions.items():
            for stype, contribs in type_contribs.items():
                # 极端时期
                extreme_mean = contribs[self.is_extreme].mean()
                extreme_std = contribs[self.is_extreme].std()
                
                # 正常时期
                normal_mean = contribs[~self.is_extreme].mean()
                normal_std = contribs[~self.is_extreme].std()
                
                # 变化率
                change_rate = (extreme_mean - normal_mean) / (normal_mean + 1e-10) * 100
                
                records.append({
                    'pollutant': pollutant,
                    'source_type': stype,
                    'normal_mean': normal_mean,
                    'normal_std': normal_std,
                    'extreme_mean': extreme_mean,
                    'extreme_std': extreme_std,
                    'change_rate': change_rate,
                })
        
        return pd.DataFrame(records)
    
    def analyze_specific_event(self, event_start: str, event_end: str,
                               pre_days: int = 3, post_days: int = 7) -> pd.DataFrame:
        """
        分析特定事件
        
        Args:
            event_start: 事件开始日期
            event_end: 事件结束日期
            pre_days: 事件前天数
            post_days: 事件后天数
        """
        dates = self.results['dates']
        type_contributions = self.results['type_contributions']
        
        event_start = pd.to_datetime(event_start)
        event_end = pd.to_datetime(event_end)
        
        # 定义时期
        pre_start = event_start - pd.Timedelta(days=pre_days)
        post_end = event_end + pd.Timedelta(days=post_days)
        
        records = []
        
        for day_idx, date in enumerate(dates):
            if date < pre_start or date > post_end:
                continue
            
            if date < event_start:
                period = 'pre_event'
            elif date <= event_end:
                period = 'event'
            else:
                period = 'post_event'
            
            for pollutant, type_contribs in type_contributions.items():
                for stype, contribs in type_contribs.items():
                    records.append({
                        'date': date,
                        'period': period,
                        'pollutant': pollutant,
                        'source_type': stype,
                        'contribution': contribs[day_idx],
                    })
        
        return pd.DataFrame(records)


if __name__ == "__main__":
    print("贡献解析模块加载成功")
