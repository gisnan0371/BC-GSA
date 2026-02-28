# -*- coding: utf-8 -*-
"""
================================================================================
伊洛河流域污染溯源模型 - 河网拓扑构建模块
Yiluo River Basin Pollution Source Apportionment - River Network Topology Builder
================================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import networkx as nx


class TopologyBuilder:
    """河网拓扑构建器"""
    
    def __init__(self, distance_matrix: pd.DataFrame,
                 wq_stations: Dict, hydro_stations: Dict, 
                 point_sources: List[str], river_segments: Dict):
        """
        初始化拓扑构建器
        
        Args:
            distance_matrix: 河道距离矩阵
            wq_stations: 水质站点配置
            hydro_stations: 水文站点配置
            point_sources: 点源列表
            river_segments: 河段定义
        """
        self.distance_matrix = distance_matrix
        self.wq_stations = wq_stations
        self.hydro_stations = hydro_stations
        self.point_sources = point_sources
        self.river_segments = river_segments
        
        # 构建的数据结构
        self.node_list = []
        self.node_types = {}
        self.node_info = {}
        self.graph = None
        self.adjacency_matrix = None
        self.directed_distance = None
        
    def build(self) -> Dict:
        """构建完整的河网拓扑"""
        print("\n" + "=" * 60)
        print("构建河网拓扑...")
        print("=" * 60)
        
        # 1. 构建节点列表
        self._build_node_list()
        
        # 2. 构建有向图
        self._build_directed_graph()
        
        # 3. 构建邻接矩阵
        self._build_adjacency_matrix()
        
        # 4. 提取节点到目标站的距离
        self._compute_distances_to_target()
        
        # 5. 构建河段-节点映射
        self._build_segment_mapping()
        
        print("\n拓扑构建完成！")
        print(f"  - 总节点数: {len(self.node_list)}")
        print(f"  - 水质站: {len([n for n, t in self.node_types.items() if t == 'WQ'])}")
        print(f"  - 水文站: {len([n for n, t in self.node_types.items() if t == 'H'])}")
        print(f"  - 点源: {len([n for n, t in self.node_types.items() if t == 'PS'])}")
        
        return {
            'node_list': self.node_list,
            'node_types': self.node_types,
            'node_info': self.node_info,
            'graph': self.graph,
            'adjacency_matrix': self.adjacency_matrix,
            'distance_to_target': self.distance_to_target,
            'segment_nodes': self.segment_nodes,
        }
    
    def _build_node_list(self):
        """构建节点列表"""
        print("\n[1/5] 构建节点列表...")
        
        # 从距离矩阵获取所有节点
        all_nodes = list(self.distance_matrix.index)
        
        # 分类节点
        wq_nodes = set(self.wq_stations.keys())
        hydro_nodes = set(self.hydro_stations.keys())
        ps_nodes = set(self.point_sources)
        
        # 处理重叠节点（水质站和水文站同名）
        overlap = wq_nodes & hydro_nodes
        if overlap:
            print(f"  - 水质站与水文站重叠: {overlap}")
        
        # 构建有序节点列表
        self.node_list = []
        self.node_types = {}
        self.node_info = {}
        
        # 添加水质站
        for node in all_nodes:
            if node in wq_nodes:
                self.node_list.append(node)
                self.node_types[node] = 'WQ'
                self.node_info[node] = {
                    'type': 'water_quality',
                    'river': self.wq_stations[node].get('river', 'unknown'),
                    'lon': self.wq_stations[node].get('lon', 0),
                    'lat': self.wq_stations[node].get('lat', 0),
                }
        
        # 添加水文站（排除已添加的重叠站点）
        for node in all_nodes:
            if node in hydro_nodes and node not in self.node_list:
                self.node_list.append(node)
                self.node_types[node] = 'H'
                station_info = self.hydro_stations.get(node, {})
                self.node_info[node] = {
                    'type': 'hydrological',
                    'subtype': 'river' if station_info.get('type', 1) == 1 else 'diversion',
                    'river': station_info.get('river', 'unknown'),
                    'lon': station_info.get('lon', 0),
                    'lat': station_info.get('lat', 0),
                    'area': station_info.get('area', 0),
                }
        
        # 添加点源
        for node in all_nodes:
            if node in ps_nodes and node not in self.node_list:
                self.node_list.append(node)
                self.node_types[node] = 'PS'
                self.node_info[node] = {
                    'type': 'point_source',
                }
        
        # 添加其他节点（可能是距离矩阵中有但配置中没有的）
        for node in all_nodes:
            if node not in self.node_list:
                self.node_list.append(node)
                self.node_types[node] = 'OTHER'
                self.node_info[node] = {'type': 'other'}
        
        # 创建节点索引映射
        self.node_to_idx = {node: i for i, node in enumerate(self.node_list)}
        self.idx_to_node = {i: node for i, node in enumerate(self.node_list)}
        
        print(f"  - 总节点数: {len(self.node_list)}")
        print(f"  - 节点类型分布: {self._count_node_types()}")
    
    def _count_node_types(self) -> Dict[str, int]:
        """统计节点类型分布"""
        counts = {}
        for node_type in self.node_types.values():
            counts[node_type] = counts.get(node_type, 0) + 1
        return counts
    
    def _build_directed_graph(self):
        """构建有向图"""
        print("\n[2/5] 构建有向图...")
        
        self.graph = nx.DiGraph()
        
        # 添加所有节点
        for node in self.node_list:
            self.graph.add_node(node, **self.node_info.get(node, {}))
        
        # 添加有向边（基于距离矩阵）
        edge_count = 0
        for src in self.node_list:
            if src not in self.distance_matrix.index:
                continue
            for dst in self.node_list:
                if dst not in self.distance_matrix.columns:
                    continue
                if src == dst:
                    continue
                
                # 处理可能返回Series的情况（当有重复列名时）
                distance = self.distance_matrix.loc[src, dst]
                if isinstance(distance, pd.Series):
                    distance = distance.iloc[0]  # 取第一个值
                
                if pd.notna(distance) and distance > 0:
                    self.graph.add_edge(src, dst, distance=float(distance))
                    edge_count += 1
        
        print(f"  - 有向边数: {edge_count}")
        print(f"  - 图是否连通: {nx.is_weakly_connected(self.graph)}")
    
    def _build_adjacency_matrix(self):
        """构建邻接矩阵"""
        print("\n[3/5] 构建邻接矩阵...")
        
        n_nodes = len(self.node_list)
        self.adjacency_matrix = np.zeros((n_nodes, n_nodes))
        self.edge_weights = np.zeros((n_nodes, n_nodes))
        
        for src in self.node_list:
            src_idx = self.node_to_idx[src]
            if src not in self.distance_matrix.index:
                continue
                
            for dst in self.node_list:
                dst_idx = self.node_to_idx[dst]
                if dst not in self.distance_matrix.columns:
                    continue
                
                if src == dst:
                    self.adjacency_matrix[src_idx, dst_idx] = 1  # 自环
                    self.edge_weights[src_idx, dst_idx] = 0.5
                else:
                    # 处理可能返回Series的情况（当有重复列名时）
                    distance = self.distance_matrix.loc[src, dst]
                    if isinstance(distance, pd.Series):
                        distance = distance.iloc[0]  # 取第一个值
                    
                    if pd.notna(distance) and distance > 0:
                        self.adjacency_matrix[src_idx, dst_idx] = 1
                        # 权重与距离成反比
                        self.edge_weights[src_idx, dst_idx] = np.exp(-0.01 * float(distance))
        
        n_edges = int(self.adjacency_matrix.sum()) - n_nodes  # 减去自环
        print(f"  - 邻接矩阵维度: {self.adjacency_matrix.shape}")
        print(f"  - 有效边数（不含自环）: {n_edges}")
    
    def _compute_distances_to_target(self, target: str = 'QiLiPu'):
        """计算各节点到目标站点的距离"""
        print(f"\n[4/5] 计算到目标站点 ({target}) 的距离...")
        
        self.target_station = target
        self.distance_to_target = {}
        
        if target not in self.distance_matrix.columns:
            print(f"  警告: 目标站点 {target} 不在距离矩阵中")
            return
        
        for node in self.node_list:
            if node == target:
                self.distance_to_target[node] = 0.0
            elif node in self.distance_matrix.index:
                distance = self.distance_matrix.loc[node, target]
                # 处理可能返回Series的情况
                if isinstance(distance, pd.Series):
                    distance = distance.iloc[0]
                
                if pd.notna(distance) and distance > 0:
                    self.distance_to_target[node] = float(distance)
                else:
                    # 尝试从其他列获取距离
                    self.distance_to_target[node] = self._estimate_distance_to_target(node, target)
            else:
                # 节点不在距离矩阵中，尝试估算
                self.distance_to_target[node] = self._estimate_distance_to_target(node, target)
        
        # 统计可达节点
        reachable = sum(1 for d in self.distance_to_target.values() if d < np.inf)
        print(f"  - 可达目标站点的节点数: {reachable}/{len(self.node_list)}")
        
        # 打印距离范围
        valid_distances = [d for d in self.distance_to_target.values() if d < np.inf and d > 0]
        if valid_distances:
            print(f"  - 距离范围: {min(valid_distances):.2f} ~ {max(valid_distances):.2f} km")
    
    def _estimate_distance_to_target(self, node: str, target: str) -> float:
        """估算节点到目标的距离（用于不在距离矩阵中的节点）"""
        # 对于点源，根据其所属河段估算距离
        if node.startswith('YL'):
            # 从POINT_SOURCE_SEGMENT_MAPPING获取其所属河段
            from ..config import POINT_SOURCE_SEGMENT_MAPPING, NODE_LEVEL_SEGMENTS
            
            segment_id = None
            for seg_id, ps_list in POINT_SOURCE_SEGMENT_MAPPING.items():
                if node in ps_list:
                    segment_id = seg_id
                    break
            
            if segment_id:
                # 找到该河段的下游站点距离
                for seg in NODE_LEVEL_SEGMENTS:
                    if seg.get('id') == segment_id:
                        downstream = seg.get('downstream')
                        if downstream and downstream in self.distance_to_target:
                            # 点源到目标的距离 ≈ 下游站点到目标的距离 + 河段内距离/2
                            d_down = self.distance_to_target.get(downstream, 50)
                            return d_down + 10  # 简单估算
                        break
            
            # 默认使用中等距离
            return 100.0
        
        # 其他类型节点，使用默认距离
        return 100.0
    
    def _build_segment_mapping(self):
        """构建河段-节点映射"""
        print("\n[5/5] 构建河段-节点映射...")
        
        self.segment_nodes = {}
        
        for seg_id, seg_info in self.river_segments.items():
            segment_stations = seg_info.get('stations', [])
            # 找到属于该河段的节点
            nodes_in_segment = []
            for station in segment_stations:
                if station in self.node_list:
                    nodes_in_segment.append(station)
            
            # 同时找到河段内的点源
            ps_in_segment = []
            for ps in self.point_sources:
                if ps in self.node_list and ps in self.distance_to_target:
                    # 判断点源是否在该河段范围内（简化：基于距离）
                    # 这里需要更精确的判断，暂时跳过
                    pass
            
            self.segment_nodes[seg_id] = {
                'name': seg_info.get('name', seg_id),
                'river': seg_info.get('river', 'unknown'),
                'stations': nodes_in_segment,
                'point_sources': ps_in_segment,
            }
            
            print(f"  - {seg_info.get('name', seg_id)}: {len(nodes_in_segment)} 个站点")
    
    def get_upstream_nodes(self, node: str) -> List[str]:
        """获取某节点的上游节点"""
        if node not in self.graph:
            return []
        return list(self.graph.predecessors(node))
    
    def get_downstream_nodes(self, node: str) -> List[str]:
        """获取某节点的下游节点"""
        if node not in self.graph:
            return []
        return list(self.graph.successors(node))
    
    def get_path_to_target(self, source: str, target: str = 'QiLiPu') -> Tuple[List[str], float]:
        """获取从源节点到目标节点的路径"""
        if source not in self.graph or target not in self.graph:
            return [], np.inf
        
        try:
            path = nx.shortest_path(self.graph, source, target, weight='distance')
            distance = nx.shortest_path_length(self.graph, source, target, weight='distance')
            return path, distance
        except nx.NetworkXNoPath:
            return [], np.inf
    
    def get_nodes_by_type(self, node_type: str) -> List[str]:
        """获取指定类型的所有节点"""
        return [node for node, ntype in self.node_types.items() if ntype == node_type]
    
    def get_nodes_by_river(self, river: str) -> List[str]:
        """获取指定河流的所有节点"""
        return [node for node, info in self.node_info.items() 
                if info.get('river') == river]
    
    def compute_transport_distance_matrix(self) -> np.ndarray:
        """计算传输距离矩阵（用于传输系数计算）"""
        n_nodes = len(self.node_list)
        dist_matrix = np.full((n_nodes, n_nodes), np.inf)
        
        for i, src in enumerate(self.node_list):
            for j, dst in enumerate(self.node_list):
                if src == dst:
                    dist_matrix[i, j] = 0
                elif src in self.distance_matrix.index and dst in self.distance_matrix.columns:
                    d = self.distance_matrix.loc[src, dst]
                    # 处理可能返回Series的情况
                    if isinstance(d, pd.Series):
                        d = d.iloc[0]
                    if pd.notna(d):
                        dist_matrix[i, j] = float(d)
        
        return dist_matrix
    
    def get_river_for_node(self, node: str) -> str:
        """获取节点所属河流"""
        return self.node_info.get(node, {}).get('river', 'unknown')
    
    def is_diversion(self, node: str) -> bool:
        """判断节点是否为取水渠"""
        info = self.node_info.get(node, {})
        return info.get('subtype') == 'diversion'
    
    def get_diversion_stations(self) -> List[str]:
        """获取所有取水渠站点"""
        return [node for node, info in self.node_info.items() 
                if info.get('subtype') == 'diversion']
    
    def print_topology_summary(self):
        """打印拓扑摘要"""
        print("\n" + "=" * 60)
        print("河网拓扑摘要")
        print("=" * 60)
        
        print("\n【节点统计】")
        type_counts = self._count_node_types()
        for ntype, count in type_counts.items():
            print(f"  - {ntype}: {count}")
        
        print("\n【河流统计】")
        river_nodes = {}
        for node, info in self.node_info.items():
            river = info.get('river', 'unknown')
            if river not in river_nodes:
                river_nodes[river] = []
            river_nodes[river].append(node)
        for river, nodes in river_nodes.items():
            print(f"  - {river}: {len(nodes)} 个节点")
        
        print("\n【目标站点可达性】")
        if hasattr(self, 'distance_to_target'):
            reachable = {k: v for k, v in self.distance_to_target.items() if v < np.inf}
            print(f"  - 可达节点数: {len(reachable)}")
            if reachable:
                print(f"  - 最近节点: {min(reachable, key=reachable.get)} ({reachable[min(reachable, key=reachable.get)]:.2f} km)")
                print(f"  - 最远节点: {max(reachable, key=reachable.get)} ({reachable[max(reachable, key=reachable.get)]:.2f} km)")


class WaterBalanceTopology:
    """水量平衡拓扑"""
    
    def __init__(self, topology: TopologyBuilder):
        """
        初始化水量平衡拓扑
        
        Args:
            topology: 拓扑构建器实例
        """
        self.topology = topology
        self.balance_equations = {}
        
    def build_balance_equations(self):
        """构建水量平衡方程"""
        print("\n构建水量平衡方程...")
        
        # 获取所有河流断面（type=1）和取水渠（type=2）
        river_stations = []
        diversion_stations = []
        
        for node, info in self.topology.node_info.items():
            if info.get('type') == 'hydrological':
                if info.get('subtype') == 'diversion':
                    diversion_stations.append(node)
                else:
                    river_stations.append(node)
        
        print(f"  - 河流断面: {len(river_stations)}")
        print(f"  - 取水渠: {len(diversion_stations)}")
        
        # 为每个河流断面构建水量平衡方程
        for station in river_stations:
            upstream = self.topology.get_upstream_nodes(station)
            
            # 分类上游节点
            upstream_river = [n for n in upstream if n in river_stations]
            upstream_diversion = [n for n in upstream if n in diversion_stations]
            upstream_ps = [n for n in upstream if self.topology.node_types.get(n) == 'PS']
            
            self.balance_equations[station] = {
                'upstream_inflow': upstream_river,      # 上游河流来水
                'diversions': upstream_diversion,       # 取水（流出）
                'point_sources': upstream_ps,           # 点源排放（流入）
                'interval_inflow': True,                # 区间入流（待估算）
            }
        
        return self.balance_equations
    
    def get_balance_equation(self, station: str) -> Dict:
        """获取指定站点的水量平衡方程"""
        return self.balance_equations.get(station, {})


if __name__ == "__main__":
    from bcgsa.config import (DATA_PATHS, WATER_QUALITY_STATIONS, HYDRO_STATIONS, 
                       POINT_SOURCES, RIVER_SEGMENTS)
    
    # 加载距离矩阵
    distance_matrix = pd.read_csv(DATA_PATHS["distance_matrix"], index_col=0)
    
    # 构建拓扑
    builder = TopologyBuilder(
        distance_matrix=distance_matrix,
        wq_stations=WATER_QUALITY_STATIONS,
        hydro_stations=HYDRO_STATIONS,
        point_sources=POINT_SOURCES,
        river_segments=RIVER_SEGMENTS,
    )
    
    topology = builder.build()
    builder.print_topology_summary()
    
    # 构建水量平衡拓扑
    wb_topo = WaterBalanceTopology(builder)
    equations = wb_topo.build_balance_equations()
