# -*- coding: utf-8 -*-
"""
================================================================================
伊洛河流域污染溯源模型 - 可视化模块
Yiluo River Basin Pollution Source Apportionment - Visualization Module
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ContributionPlotter:
    """贡献率可视化"""
    
    def __init__(self, output_dir: Path, dpi: int = 300, figsize: Tuple = (12, 6)):
        """
        初始化绘图器
        
        Args:
            output_dir: 输出目录
            dpi: 图像分辨率
            figsize: 默认图像大小
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.figsize = figsize
        
        # 颜色方案
        self.colors = {
            'point_source': '#e74c3c',    # 红色 - 点源
            'nps': '#27ae60',             # 绿色 - 面源
            'upstream': '#3498db',        # 蓝色 - 上游
            'other': '#95a5a6',           # 灰色 - 其他
        }
        
        # 污染物名称
        self.pollutant_names = {
            'NH3N': '氨氮',
            'TP': '总磷',
            'TN': '总氮',
        }
        
    def plot_time_series(self, type_contributions: Dict[str, Dict[str, np.ndarray]],
                         dates: pd.DatetimeIndex, pollutant: str,
                         title: Optional[str] = None) -> str:
        """
        绘制贡献率时间序列图
        
        Args:
            type_contributions: {pollutant: {type: contributions}}
            dates: 日期索引
            pollutant: 污染物名称
            title: 图标题
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        pollutant_name = self.pollutant_names.get(pollutant, pollutant)
        
        if pollutant not in type_contributions:
            print(f"Warning: {pollutant} not found in contributions")
            return ""
        
        type_contribs = type_contributions[pollutant]
        
        # 堆叠面积图
        bottom = np.zeros(len(dates))
        
        for stype in ['upstream', 'nps', 'point_source']:
            if stype in type_contribs:
                contrib = type_contribs[stype] * 100  # 转为百分比
                ax.fill_between(dates, bottom, bottom + contrib,
                               label=self._get_type_name(stype),
                               color=self.colors.get(stype, '#95a5a6'),
                               alpha=0.7)
                bottom += contrib
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('贡献率 (%)', fontsize=12)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 设置日期格式
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        if title:
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(f'{pollutant_name}污染源贡献时间序列', fontsize=14)
        
        plt.tight_layout()
        
        # 保存
        filepath = self.output_dir / f'contribution_timeseries_{pollutant}.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_pie_chart(self, type_contributions: Dict[str, Dict[str, np.ndarray]],
                       pollutant: str, period: str = 'annual') -> str:
        """
        绘制贡献率饼图
        
        Args:
            type_contributions: 类型贡献率
            pollutant: 污染物
            period: 时间段描述
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        pollutant_name = self.pollutant_names.get(pollutant, pollutant)
        
        if pollutant not in type_contributions:
            return ""
        
        type_contribs = type_contributions[pollutant]
        
        # 计算平均贡献
        labels = []
        sizes = []
        colors = []
        
        for stype in ['point_source', 'nps', 'upstream']:
            if stype in type_contribs:
                mean_contrib = type_contribs[stype].mean() * 100
                if mean_contrib > 0.5:  # 只显示 > 0.5% 的
                    labels.append(f'{self._get_type_name(stype)}\n({mean_contrib:.1f}%)')
                    sizes.append(mean_contrib)
                    colors.append(self.colors.get(stype, '#95a5a6'))
        
        if not sizes:
            return ""
        
        # 绘制饼图
        wedges, texts = ax.pie(sizes, colors=colors, startangle=90,
                               wedgeprops={'edgecolor': 'white', 'linewidth': 2})
        
        # 添加图例
        ax.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0.5),
                 fontsize=11)
        
        ax.set_title(f'{pollutant_name}污染源贡献构成 ({period})', fontsize=14)
        
        plt.tight_layout()
        
        filepath = self.output_dir / f'contribution_pie_{pollutant}_{period}.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_seasonal_comparison(self, type_contributions: Dict[str, Dict[str, np.ndarray]],
                                 dates: pd.DatetimeIndex, pollutant: str) -> str:
        """
        绘制季节对比图
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        pollutant_name = self.pollutant_names.get(pollutant, pollutant)
        
        if pollutant not in type_contributions:
            return ""
        
        type_contribs = type_contributions[pollutant]
        
        # 创建DataFrame
        df = pd.DataFrame({'date': dates})
        for stype, contrib in type_contribs.items():
            df[stype] = contrib
        df['month'] = pd.to_datetime(df['date']).dt.month
        
        # 定义季节
        seasons = {
            '春季 (3-5月)': [3, 4, 5],
            '夏季 (6-8月)': [6, 7, 8],
            '秋季 (9-11月)': [9, 10, 11],
            '冬季 (12-2月)': [12, 1, 2],
        }
        
        for i, (season_name, months) in enumerate(seasons.items()):
            ax = axes[i]
            season_df = df[df['month'].isin(months)]
            
            # 计算季节平均
            means = []
            labels = []
            colors_list = []
            
            for stype in ['point_source', 'nps', 'upstream']:
                if stype in season_df.columns:
                    mean = season_df[stype].mean() * 100
                    means.append(mean)
                    labels.append(self._get_type_name(stype))
                    colors_list.append(self.colors.get(stype, '#95a5a6'))
            
            if means:
                bars = ax.bar(labels, means, color=colors_list, edgecolor='white')
                ax.set_ylabel('贡献率 (%)')
                ax.set_title(season_name)
                ax.set_ylim(0, max(means) * 1.2)
                
                # 添加数值标签
                for bar, val in zip(bars, means):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{val:.1f}%', ha='center', fontsize=10)
        
        fig.suptitle(f'{pollutant_name}污染源贡献季节变化', fontsize=14, y=1.02)
        plt.tight_layout()
        
        filepath = self.output_dir / f'contribution_seasonal_{pollutant}.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_extreme_event_response(self, event_data: pd.DataFrame,
                                    pollutant: str, event_name: str) -> str:
        """
        绘制极端事件响应图
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        pollutant_name = self.pollutant_names.get(pollutant, pollutant)
        
        # 筛选污染物数据
        df = event_data[event_data['pollutant'] == pollutant].copy()
        if df.empty:
            return ""
        
        # 按类型绘制
        for stype in ['point_source', 'nps', 'upstream']:
            type_df = df[df['source_type'] == stype]
            if not type_df.empty:
                ax.plot(type_df['date'], type_df['contribution'] * 100,
                       label=self._get_type_name(stype),
                       color=self.colors.get(stype, '#95a5a6'),
                       linewidth=2, marker='o', markersize=4)
        
        # 标记事件期间
        periods = df['period'].unique()
        if 'event' in periods:
            event_dates = df[df['period'] == 'event']['date']
            if len(event_dates) > 0:
                ax.axvspan(event_dates.min(), event_dates.max(),
                          alpha=0.2, color='red', label='事件期间')
        
        ax.set_xlabel('日期', fontsize=12)
        ax.set_ylabel('贡献率 (%)', fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{event_name} - {pollutant_name}污染源贡献响应', fontsize=14)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        filepath = self.output_dir / f'event_{event_name}_{pollutant}.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def plot_point_source_ranking(self, contributions: np.ndarray,
                                  node_list: List[str], pollutant: str,
                                  top_n: int = 10) -> str:
        """
        绘制点源排名图
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        pollutant_name = self.pollutant_names.get(pollutant, pollutant)
        
        # 筛选点源
        ps_contributions = {}
        for i, node in enumerate(node_list):
            if node.startswith('YL'):
                mean_contrib = contributions[:, i].mean() * 100
                ps_contributions[node] = mean_contrib
        
        # 排序
        sorted_ps = sorted(ps_contributions.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        if not sorted_ps:
            return ""
        
        nodes, contribs = zip(*sorted_ps)
        y_pos = np.arange(len(nodes))
        
        bars = ax.barh(y_pos, contribs, color=self.colors['point_source'], 
                      edgecolor='white', height=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(nodes)
        ax.set_xlabel('平均贡献率 (%)', fontsize=12)
        ax.set_title(f'{pollutant_name} - 点源贡献排名 (Top {top_n})', fontsize=14)
        ax.invert_yaxis()
        
        # 添加数值标签
        for bar, val in zip(bars, contribs):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                   f'{val:.2f}%', va='center', fontsize=10)
        
        plt.tight_layout()
        
        filepath = self.output_dir / f'ps_ranking_{pollutant}.png'
        plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return str(filepath)
    
    def _get_type_name(self, stype: str) -> str:
        """获取类型中文名"""
        names = {
            'point_source': '点源',
            'nps': '面源',
            'upstream': '上游来水',
            'other': '其他',
        }
        return names.get(stype, stype)
    
    def generate_all_plots(self, results: Dict, dates: pd.DatetimeIndex,
                          pollutants: List[str]) -> List[str]:
        """
        生成所有图表
        
        Args:
            results: 解析结果
            dates: 日期索引
            pollutants: 污染物列表
            
        Returns:
            filepaths: 生成的图表文件路径列表
        """
        filepaths = []
        
        type_contributions = results.get('type_contributions', {})
        contributions = results.get('contributions')
        node_list = results.get('node_list', [])
        
        for pollutant in pollutants:
            # 时间序列图
            fp = self.plot_time_series(type_contributions, dates, pollutant)
            if fp:
                filepaths.append(fp)
            
            # 饼图
            fp = self.plot_pie_chart(type_contributions, pollutant)
            if fp:
                filepaths.append(fp)
            
            # 季节对比图
            fp = self.plot_seasonal_comparison(type_contributions, dates, pollutant)
            if fp:
                filepaths.append(fp)
            
            # 点源排名
            if contributions is not None:
                p_idx = pollutants.index(pollutant)
                fp = self.plot_point_source_ranking(
                    contributions[:, :, p_idx], node_list, pollutant
                )
                if fp:
                    filepaths.append(fp)
        
        print(f"\n共生成 {len(filepaths)} 张图表")
        return filepaths


if __name__ == "__main__":
    print("可视化模块加载成功")
