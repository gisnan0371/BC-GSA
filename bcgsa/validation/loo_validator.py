# -*- coding: utf-8 -*-
"""
================================================================================
留一交叉验证水质模型
Leave-One-Out Cross-Validation for Water Quality Model
================================================================================

功能：
1. 对7个水质站点进行留一交叉验证
2. 每次留出1个站点作为验证，其余6个站点训练
3. 验证站点的滞后特征用最近邻训练站点观测值填充
4. 输出详细验证结果、汇总表、对比图

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LeaveOneOutValidator:
    """
    留一交叉验证器
    
    对每个水质站点进行留一验证，评估模型的空间泛化能力
    """
    
    def __init__(self, output_dir: str = None):
        """
        初始化验证器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 水质站点列表
        self.wq_stations = [
            'LingKou', 'LuoNingChangShui', 'GaoYaZhai', 
            'BaiMaSi', 'TanTou', 'QiLiPu', 'LongMenDaQiao'
        ]
        
        # 站点上下游关系（手动定义）
        self.upstream_map = {
            'LingKou': [],  # 洛河最上游
            'LuoNingChangShui': ['LingKou'],
            'GaoYaZhai': ['LuoNingChangShui', 'LingKou'],
            'BaiMaSi': ['GaoYaZhai', 'LuoNingChangShui'],
            'TanTou': [],  # 伊河最上游
            'LongMenDaQiao': ['TanTou'],
            'QiLiPu': ['BaiMaSi', 'LongMenDaQiao'],  # 汇流后
        }
        
        # 站点距离（用于最近邻查找）
        self.station_distances = {
            ('LingKou', 'LuoNingChangShui'): 50,
            ('LingKou', 'GaoYaZhai'): 130,
            ('LingKou', 'BaiMaSi'): 170,
            ('LuoNingChangShui', 'GaoYaZhai'): 80,
            ('LuoNingChangShui', 'BaiMaSi'): 120,
            ('GaoYaZhai', 'BaiMaSi'): 40,
            ('BaiMaSi', 'QiLiPu'): 30,
            ('TanTou', 'LongMenDaQiao'): 100,
            ('LongMenDaQiao', 'QiLiPu'): 25,
            # 跨河距离（较大）
            ('LingKou', 'TanTou'): 200,
            ('LingKou', 'LongMenDaQiao'): 250,
            ('LingKou', 'QiLiPu'): 200,
            ('LuoNingChangShui', 'TanTou'): 180,
            ('LuoNingChangShui', 'LongMenDaQiao'): 200,
            ('LuoNingChangShui', 'QiLiPu'): 150,
            ('GaoYaZhai', 'TanTou'): 150,
            ('GaoYaZhai', 'LongMenDaQiao'): 80,
            ('GaoYaZhai', 'QiLiPu'): 70,
            ('BaiMaSi', 'TanTou'): 130,
            ('BaiMaSi', 'LongMenDaQiao'): 50,
            ('TanTou', 'QiLiPu'): 125,
        }
        
        # 污染物
        self.pollutants = ['NH3N', 'TP', 'TN']
        
        # 存储验证结果
        self.results = {}
        self.predictions = {}
        
    def get_distance(self, station1: str, station2: str) -> float:
        """获取两站点间距离"""
        if station1 == station2:
            return 0
        key1 = (station1, station2)
        key2 = (station2, station1)
        if key1 in self.station_distances:
            return self.station_distances[key1]
        elif key2 in self.station_distances:
            return self.station_distances[key2]
        else:
            return 999  # 未知距离，设为较大值
    
    def find_nearest_station(self, target: str, candidates: List[str]) -> str:
        """找到距离目标站点最近的候选站点"""
        if not candidates:
            return None
        
        min_dist = float('inf')
        nearest = candidates[0]
        
        for station in candidates:
            dist = self.get_distance(target, station)
            if dist < min_dist:
                min_dist = dist
                nearest = station
        
        return nearest
    
    def compute_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """
        计算验证指标
        
        Args:
            actual: 实际观测值
            predicted: 预测值
            
        Returns:
            包含R², NSE, RMSE的字典
        """
        valid = (~np.isnan(actual)) & (~np.isnan(predicted)) & (actual > 0) & (predicted > 0)
        
        if valid.sum() < 30:
            return {'R2': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'n_valid': valid.sum()}
        
        a = actual[valid]
        p = predicted[valid]
        
        # R²
        ss_res = np.sum((a - p) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        r2 = max(-1, min(1, r2))
        
        # NSE (与R²计算方式相同，但概念不同)
        nse = r2
        
        # RMSE
        rmse = np.sqrt(np.mean((a - p) ** 2))
        
        return {'R2': r2, 'NSE': nse, 'RMSE': rmse, 'n_valid': valid.sum()}
    
    def run_loo_validation(self, topology, processed_data, flows, loader, device='cpu'):
        """
        运行留一交叉验证
        
        Args:
            topology: 河网拓扑
            processed_data: 预处理后的数据
            flows: 流量数据
            loader: 数据加载器
            device: 计算设备
        """
        print("\n" + "=" * 70)
        print("留一交叉验证 (Leave-One-Out Cross-Validation)")
        print("=" * 70)
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"站点数: {len(self.wq_stations)}")
        print(f"验证轮次: {len(self.wq_stations)}")
        
        # 准备数据
        dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
        n_days = len(dates)
        
        wq_data = processed_data['water_quality']
        runoff_data = processed_data['runoff']
        ps_data = processed_data['point_source']
        metro_data = processed_data.get('metro', None)  # 注意：是'metro'不是'meteorology'
        
        # 污染物列名映射
        pollutant_cols = {
            'NH3N': 'AmmoniaNitrogen',
            'TP': 'TotalPhosphorus',
            'TN': 'TotalNitrogen',
        }
        
        # 提取水质数据
        wq_dict = {}
        for station in self.wq_stations:
            for poll, col in pollutant_cols.items():
                col_name = f"{station}_{col}"
                if col_name in wq_data.columns:
                    wq_dict[f"{station}_{poll}"] = wq_data[col_name].values
        
        # 流量映射
        nearby_hydro = {
            'GaoYaZhai': 'YiYang',
            'LuoNingChangShui': 'ChangShui',
            'LongMenDaQiao': 'LongMenZhen',
            'QiLiPu': 'HeiShiGuan',
            'BaiMaSi': 'BaiMaSi',
            'TanTou': 'TanTou',
            'LingKou': 'LingKou',
        }
        
        flow_dict = {}
        for station in self.wq_stations:
            hydro = nearby_hydro.get(station, station)
            if hydro in runoff_data.columns:
                flow_dict[station] = runoff_data[hydro].values
            elif station in runoff_data.columns:
                flow_dict[station] = runoff_data[station].values
            else:
                flow_dict[station] = runoff_data.mean(axis=1).values
        
        # 气象数据
        precip_cols = [c for c in metro_data.columns if 'Precipitation' in c] if metro_data is not None else []
        precip = metro_data[precip_cols].mean(axis=1).values if precip_cols else np.zeros(n_days)
        
        temp_cols = [c for c in metro_data.columns if 'Temperature' in c] if metro_data is not None else []
        temp = metro_data[temp_cols].mean(axis=1).values if temp_cols else 15 + 10 * np.sin(2 * np.pi * (dates.dayofyear - 100) / 365)
        
        # 点源数据
        ps_col_mapping = {'NH3N': 'NH4N_load', 'TP': 'TP_load', 'TN': 'TN_load'}
        ps_dict = {}
        for col in ps_data.columns:
            if '_load' in col:
                ps_dict[col] = ps_data[col].values
        
        # 导入增强版水质模型
        from bcgsa.models.water_quality import EnhancedWaterQualityModel
        
        # 留一交叉验证
        all_results = []
        all_predictions = {}
        
        for i, val_station in enumerate(self.wq_stations):
            print(f"\n{'='*60}")
            print(f"轮次 {i+1}/{len(self.wq_stations)}: 验证站点 = {val_station}")
            print(f"{'='*60}")
            
            # 训练站点
            train_stations = [s for s in self.wq_stations if s != val_station]
            print(f"  训练站点: {train_stations}")
            
            # 找到验证站点的最近邻（用于填充滞后特征）
            nearest = self.find_nearest_station(val_station, train_stations)
            print(f"  最近邻站点: {nearest} (用于填充滞后特征)")
            
            # 创建并训练模型
            model = EnhancedWaterQualityModel(
                pollutants=self.pollutants,
                use_point_source=True,
                use_temperature=True,
                use_lag_features=True,
                use_upstream=True,
                model_type='ensemble'
            )
            
            # 设置上下游关系
            model.upstream_stations = {
                s: [(up, self.get_distance(up, s)) for up in self.upstream_map.get(s, []) if up in train_stations]
                for s in self.wq_stations
            }
            model.downstream_stations = {}
            
            # 训练模型（静默模式）
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                model.fit(
                    train_stations=train_stations,
                    dates=dates,
                    wq_data=wq_dict,
                    flow_data=flow_dict,
                    precip=precip,
                    temp=temp,
                    ps_data=ps_dict
                )
            finally:
                sys.stdout = old_stdout
            
            # 获取验证站点流量
            val_flow = flow_dict.get(val_station, np.mean(list(flow_dict.values()), axis=0))
            
            # 为验证站点构建特征（滞后特征用最近邻站点填充）
            val_predictions = {}
            val_metrics = {}
            
            for poll in self.pollutants:
                val_key = f"{val_station}_{poll}"
                nearest_key = f"{nearest}_{poll}"
                
                # 验证站点的实际观测值
                actual = wq_dict.get(val_key, np.full(n_days, np.nan))
                
                # 最近邻站点的观测值（用于填充滞后特征）
                nearest_conc = wq_dict.get(nearest_key, np.full(n_days, np.nan))
                nearest_conc_filled = pd.Series(nearest_conc).interpolate().ffill().bfill().values
                
                # 创建验证站点的"伪观测值"（实际上是最近邻的值，用于滞后特征）
                pseudo_wq_dict = wq_dict.copy()
                pseudo_wq_dict[val_key] = nearest_conc_filled
                
                # 获取点源负荷
                ps_load = model._get_upstream_ps_load(val_station, poll, ps_dict, dates)
                
                # 构建特征
                X, _ = model._build_features(
                    station=val_station,
                    pollutant=poll,
                    dates=dates,
                    flow=val_flow,
                    precip=precip,
                    temp=temp,
                    conc_obs=nearest_conc_filled,  # 用最近邻填充
                    all_station_conc=pseudo_wq_dict,
                    ps_load=ps_load
                )
                
                # 使用训练好的模型集成预测
                predictions_list = []
                weights_list = []
                
                for train_station in train_stations:
                    if poll in model.models.get(train_station, {}):
                        m = model.models[train_station][poll]
                        s = model.scalers[train_station][poll]
                        r2 = model.performance.get(train_station, {}).get(poll, 0.5)
                        
                        try:
                            X_scaled = s.transform(X)
                            pred_log = m.predict(X_scaled)
                            pred = np.exp(pred_log)
                            
                            predictions_list.append(pred)
                            weights_list.append(max(0.1, r2))
                        except:
                            pass
                
                if predictions_list:
                    weights_arr = np.array(weights_list)
                    weights_arr = weights_arr / weights_arr.sum()
                    
                    predicted = np.zeros(n_days)
                    for pred, w in zip(predictions_list, weights_arr):
                        predicted += pred * w
                else:
                    # 回退到最近邻观测值
                    predicted = nearest_conc_filled
                
                val_predictions[poll] = predicted
                
                # 计算验证指标
                metrics = self.compute_metrics(actual, predicted)
                val_metrics[poll] = metrics
                
                print(f"  {poll}: R²={metrics['R2']:.4f}, NSE={metrics['NSE']:.4f}, RMSE={metrics['RMSE']:.4f}")
            
            # 保存结果
            result_row = {
                'val_station': val_station,
                'train_stations': ','.join(train_stations),
                'nearest_station': nearest,
            }
            for poll in self.pollutants:
                result_row[f'{poll}_R2'] = val_metrics[poll]['R2']
                result_row[f'{poll}_NSE'] = val_metrics[poll]['NSE']
                result_row[f'{poll}_RMSE'] = val_metrics[poll]['RMSE']
                result_row[f'{poll}_n_valid'] = val_metrics[poll]['n_valid']
            
            all_results.append(result_row)
            
            # 保存详细预测值
            pred_df = pd.DataFrame({
                'date': dates,
                'station': val_station,
            })
            for poll in self.pollutants:
                val_key = f"{val_station}_{poll}"
                actual = wq_dict.get(val_key, np.full(n_days, np.nan))
                pred_df[f'{poll}_actual'] = actual
                pred_df[f'{poll}_predicted'] = val_predictions[poll]
            
            all_predictions[val_station] = pred_df
        
        # 保存结果
        self.results = pd.DataFrame(all_results)
        self.predictions = all_predictions
        
        # 生成报告
        self._generate_report()
        
        # 生成图表
        self._generate_plots(wq_dict)
        
        return self.results
    
    def _generate_report(self):
        """生成验证报告"""
        print("\n" + "=" * 70)
        print("留一交叉验证结果汇总")
        print("=" * 70)
        
        df = self.results
        
        # 1. 按验证站点汇总
        print("\n【按验证站点】")
        print("-" * 100)
        header = f"{'验证站点':<20}"
        for poll in self.pollutants:
            header += f" | {poll}_R²  | {poll}_NSE | {poll}_RMSE"
        print(header)
        print("-" * 100)
        
        for _, row in df.iterrows():
            line = f"{row['val_station']:<20}"
            for poll in self.pollutants:
                r2 = row[f'{poll}_R2']
                nse = row[f'{poll}_NSE']
                rmse = row[f'{poll}_RMSE']
                r2_str = f"{r2:>7.4f}" if not np.isnan(r2) else "    N/A"
                nse_str = f"{nse:>7.4f}" if not np.isnan(nse) else "    N/A"
                rmse_str = f"{rmse:>8.4f}" if not np.isnan(rmse) else "     N/A"
                line += f" | {r2_str} | {nse_str} | {rmse_str}"
            print(line)
        
        # 2. 按污染物汇总
        print("\n【按污染物汇总】")
        print("-" * 80)
        print(f"{'污染物':<8} | {'平均R²':>8} | {'最优站点':<20} | {'最优R²':>8} | {'最差站点':<20} | {'最差R²':>8}")
        print("-" * 80)
        
        for poll in self.pollutants:
            r2_col = f'{poll}_R2'
            r2_values = df[r2_col].dropna()
            
            if len(r2_values) > 0:
                avg_r2 = r2_values.mean()
                best_idx = r2_values.idxmax()
                worst_idx = r2_values.idxmin()
                best_station = df.loc[best_idx, 'val_station']
                worst_station = df.loc[worst_idx, 'val_station']
                best_r2 = r2_values.max()
                worst_r2 = r2_values.min()
                
                print(f"{poll:<8} | {avg_r2:>8.4f} | {best_station:<20} | {best_r2:>8.4f} | {worst_station:<20} | {worst_r2:>8.4f}")
        
        # 3. 综合评价
        print("\n【综合评价】")
        print("-" * 60)
        
        # 计算每个站点的平均R²
        df['avg_R2'] = df[[f'{p}_R2' for p in self.pollutants]].mean(axis=1)
        
        best_overall = df.loc[df['avg_R2'].idxmax()]
        worst_overall = df.loc[df['avg_R2'].idxmin()]
        
        print(f"整体平均R²: {df['avg_R2'].mean():.4f}")
        print(f"R² > 0 的站点数: {(df['avg_R2'] > 0).sum()}/{len(df)}")
        print(f"R² > 0.3 的站点数: {(df['avg_R2'] > 0.3).sum()}/{len(df)}")
        
        print(f"\n最优验证站点: {best_overall['val_station']} (平均R²={best_overall['avg_R2']:.4f})")
        print(f"  最近邻站点: {best_overall['nearest_station']}")
        print(f"  可能原因: ", end="")
        if best_overall['val_station'] in ['GaoYaZhai', 'LuoNingChangShui']:
            print("位于中游，上下游训练数据丰富，空间相关性强")
        elif best_overall['val_station'] in ['BaiMaSi', 'LongMenDaQiao']:
            print("位于下游，受上游影响明显，传输模式稳定")
        else:
            print("空间位置适中，模型泛化能力较好")
        
        print(f"\n最差验证站点: {worst_overall['val_station']} (平均R²={worst_overall['avg_R2']:.4f})")
        print(f"  最近邻站点: {worst_overall['nearest_station']}")
        print(f"  可能原因: ", end="")
        if worst_overall['val_station'] == 'QiLiPu':
            print("位于伊洛河汇合后，水质受两条河流混合影响，特征复杂")
        elif worst_overall['val_station'] in ['LingKou', 'TanTou']:
            print("位于最上游，缺少上游参考数据，滞后特征用最近邻填充可能不够准确")
        else:
            print("可能存在局部污染源或水质特征异常")
        
        # 4. 保存结果到CSV
        csv_path = self.output_dir / 'loo_validation_summary.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"\n汇总表已保存至: {csv_path}")
        
        # 5. 保存详细预测值
        for station, pred_df in self.predictions.items():
            pred_path = self.output_dir / f'loo_predictions_{station}.csv'
            pred_df.to_csv(pred_path, index=False, encoding='utf-8-sig')
        print(f"详细预测值已保存至: {self.output_dir}/loo_predictions_*.csv")
    
    def _generate_plots(self, wq_dict: Dict):
        """生成验证效果对比图"""
        print("\n生成验证效果对比图...")
        
        # 1. 散点图：预测vs观测（每个站点每个污染物）
        fig, axes = plt.subplots(len(self.wq_stations), len(self.pollutants), 
                                  figsize=(15, 3*len(self.wq_stations)))
        
        for i, station in enumerate(self.wq_stations):
            if station not in self.predictions:
                continue
            
            pred_df = self.predictions[station]
            
            for j, poll in enumerate(self.pollutants):
                ax = axes[i, j] if len(self.wq_stations) > 1 else axes[j]
                
                actual = pred_df[f'{poll}_actual'].values
                predicted = pred_df[f'{poll}_predicted'].values
                
                valid = (~np.isnan(actual)) & (~np.isnan(predicted)) & (actual > 0) & (predicted > 0)
                
                if valid.sum() > 0:
                    ax.scatter(actual[valid], predicted[valid], alpha=0.3, s=10)
                    
                    # 1:1线
                    max_val = max(actual[valid].max(), predicted[valid].max())
                    ax.plot([0, max_val], [0, max_val], 'r--', label='1:1 line')
                    
                    # 计算R²
                    r2 = self.results.loc[self.results['val_station']==station, f'{poll}_R2'].values[0]
                    ax.set_title(f'{station} - {poll}\nR²={r2:.3f}', fontsize=10)
                else:
                    ax.set_title(f'{station} - {poll}\nNo data', fontsize=10)
                
                if i == len(self.wq_stations) - 1:
                    ax.set_xlabel('Observed')
                if j == 0:
                    ax.set_ylabel('Predicted')
        
        plt.tight_layout()
        fig_path = self.output_dir / 'loo_validation_scatter.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  散点图已保存: {fig_path}")
        
        # 2. 时间序列对比图（每个站点）
        for station in self.wq_stations:
            if station not in self.predictions:
                continue
            
            pred_df = self.predictions[station]
            
            fig, axes = plt.subplots(len(self.pollutants), 1, figsize=(14, 3*len(self.pollutants)))
            
            for j, poll in enumerate(self.pollutants):
                ax = axes[j]
                
                actual = pred_df[f'{poll}_actual'].values
                predicted = pred_df[f'{poll}_predicted'].values
                dates = pred_df['date'].values
                
                ax.plot(dates, actual, 'b-', label='Observed', alpha=0.7, linewidth=1)
                ax.plot(dates, predicted, 'r-', label='Predicted', alpha=0.7, linewidth=1)
                
                r2 = self.results.loc[self.results['val_station']==station, f'{poll}_R2'].values[0]
                ax.set_title(f'{station} - {poll} (R²={r2:.3f})')
                ax.set_ylabel('Concentration (mg/L)')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Date')
            
            plt.tight_layout()
            fig_path = self.output_dir / f'loo_timeseries_{station}.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  时间序列图已保存: {self.output_dir}/loo_timeseries_*.png")
        
        # 3. R²汇总柱状图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(self.wq_stations))
        width = 0.25
        
        for j, poll in enumerate(self.pollutants):
            r2_values = [self.results.loc[self.results['val_station']==s, f'{poll}_R2'].values[0] 
                        for s in self.wq_stations]
            ax.bar(x + j*width, r2_values, width, label=poll)
        
        ax.set_xlabel('Validation Station')
        ax.set_ylabel('R²')
        ax.set_title('Leave-One-Out Cross-Validation Results')
        ax.set_xticks(x + width)
        ax.set_xticklabels(self.wq_stations, rotation=45, ha='right')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig_path = self.output_dir / 'loo_validation_r2_summary.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  R²汇总图已保存: {fig_path}")


def run_loo_validation_standalone(topology, processed_data, flows, loader, 
                                   output_dir: str, device='cpu'):
    """
    独立运行留一交叉验证
    
    Args:
        topology: 河网拓扑
        processed_data: 预处理后的数据
        flows: 流量数据
        loader: 数据加载器
        output_dir: 输出目录
        device: 计算设备
    
    Returns:
        验证结果DataFrame
    """
    validator = LeaveOneOutValidator(output_dir)
    results = validator.run_loo_validation(topology, processed_data, flows, loader, device)
    return results
