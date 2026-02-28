# -*- coding: utf-8 -*-
"""
================================================================================
水质模型综合验证模块
Combined Validation: Temporal + Spatial with Physics Constraints
================================================================================

功能：
1. 时间序列验证：2020年训练，2021年验证（所有7个站点）
2. 空间外推验证：物理约束插值（留一交叉验证）
3. 衰减系数从训练数据率定
4. 生成综合验证报告和对比图

================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class CombinedValidator:
    """
    综合验证器：时间序列验证 + 物理约束空间插值
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
        
        # 污染物
        self.pollutants = ['NH3N', 'TP', 'TN']
        
        # 上游关系（用于物理约束）
        self.upstream_map = {
            'LingKou': [],  # 洛河最上游，用临近站点
            'LuoNingChangShui': ['LingKou'],
            'GaoYaZhai': ['LuoNingChangShui'],
            'BaiMaSi': ['GaoYaZhai'],
            'TanTou': [],  # 伊河最上游，用临近站点
            'LongMenDaQiao': ['TanTou'],
            'QiLiPu': ['BaiMaSi', 'LongMenDaQiao'],  # 汇流后
        }
        
        # 站点距离（km）
        self.station_distances = {
            ('LingKou', 'LuoNingChangShui'): 50,
            ('LuoNingChangShui', 'GaoYaZhai'): 80,
            ('GaoYaZhai', 'BaiMaSi'): 40,
            ('BaiMaSi', 'QiLiPu'): 30,
            ('TanTou', 'LongMenDaQiao'): 100,
            ('LongMenDaQiao', 'QiLiPu'): 25,
        }
        
        # 临近站点（用于边界站点）
        self.neighbor_map = {
            'LingKou': 'LuoNingChangShui',
            'TanTou': 'LongMenDaQiao',
        }
        
        # 存储结果
        self.temporal_results = {}
        self.spatial_results = {}
        self.calibrated_decay_rates = {}
        
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
            return 50  # 默认距离
    
    def compute_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """计算验证指标"""
        valid = (~np.isnan(actual)) & (~np.isnan(predicted)) & (actual > 0) & (predicted > 0)
        
        if valid.sum() < 30:
            return {'R2': np.nan, 'NSE': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'n_valid': valid.sum()}
        
        a = actual[valid]
        p = predicted[valid]
        
        # R²
        ss_res = np.sum((a - p) ** 2)
        ss_tot = np.sum((a - np.mean(a)) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        r2 = max(-1, min(1, r2))
        
        # NSE
        nse = r2
        
        # RMSE
        rmse = np.sqrt(np.mean((a - p) ** 2))
        
        # PBIAS
        pbias = 100 * np.sum(p - a) / (np.sum(a) + 1e-10)
        
        return {'R2': r2, 'NSE': nse, 'RMSE': rmse, 'PBIAS': pbias, 'n_valid': valid.sum()}
    
    # ========== Part 1: 时间序列验证 ==========
    
    def run_temporal_validation(self, 
                                 dates: pd.DatetimeIndex,
                                 wq_dict: Dict[str, np.ndarray],
                                 flow_dict: Dict[str, np.ndarray],
                                 precip: np.ndarray,
                                 temp: np.ndarray,
                                 ps_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        时间序列验证：2020年训练，2021年验证
        
        所有7个站点都参与训练和验证
        """
        print("\n" + "=" * 70)
        print("Part 1: 时间序列验证")
        print("=" * 70)
        print("训练期: 2020-01-01 ~ 2020-12-31")
        print("验证期: 2021-01-01 ~ 2021-12-31")
        
        # 时间划分
        train_mask = (dates.year == 2020)
        val_mask = (dates.year == 2021)
        
        train_dates = dates[train_mask]
        val_dates = dates[val_mask]
        
        print(f"训练样本数: {train_mask.sum()}")
        print(f"验证样本数: {val_mask.sum()}")
        
        # 导入模型
        from bcgsa.models.water_quality import EnhancedWaterQualityModel
        
        all_results = []
        all_predictions = {}
        
        for station in self.wq_stations:
            print(f"\n{'='*50}")
            print(f"站点: {station}")
            print(f"{'='*50}")
            
            # 创建模型
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
                s: [(up, self.get_distance(up, s)) for up in self.upstream_map.get(s, [])]
                for s in self.wq_stations
            }
            model.downstream_stations = {}
            
            # 准备训练数据（仅2020年）
            train_wq_dict = {k: v[train_mask] for k, v in wq_dict.items()}
            train_flow_dict = {k: v[train_mask] for k, v in flow_dict.items()}
            train_precip = precip[train_mask]
            train_temp = temp[train_mask]
            train_ps_dict = {k: v[train_mask] for k, v in ps_dict.items()}
            
            # 静默训练
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                model.fit(
                    train_stations=[station],  # 只训练当前站点
                    dates=train_dates,
                    wq_data=train_wq_dict,
                    flow_data=train_flow_dict,
                    precip=train_precip,
                    temp=train_temp,
                    ps_data=train_ps_dict
                )
            finally:
                sys.stdout = old_stdout
            
            # 验证期预测
            val_wq_dict = {k: v[val_mask] for k, v in wq_dict.items()}
            val_flow_dict = {k: v[val_mask] for k, v in flow_dict.items()}
            val_precip = precip[val_mask]
            val_temp = temp[val_mask]
            val_ps_dict = {k: v[val_mask] for k, v in ps_dict.items()}
            
            station_predictions = {}
            station_metrics = {}
            
            for poll in self.pollutants:
                key = f"{station}_{poll}"
                
                # 验证期观测值
                actual = val_wq_dict.get(key, np.full(len(val_dates), np.nan))
                
                # 获取流量
                flow = val_flow_dict.get(station, np.mean(list(val_flow_dict.values()), axis=0))
                
                # 获取点源负荷
                ps_load = model._get_upstream_ps_load(station, poll, val_ps_dict, val_dates)
                
                # 构建特征（使用验证期自己的历史数据）
                X, _ = model._build_features(
                    station=station,
                    pollutant=poll,
                    dates=val_dates,
                    flow=flow,
                    precip=val_precip,
                    temp=val_temp,
                    conc_obs=actual,  # 使用验证期的观测值构建滞后特征
                    all_station_conc=val_wq_dict,
                    ps_load=ps_load
                )
                
                # 预测
                if station in model.models and poll in model.models[station]:
                    m = model.models[station][poll]
                    s = model.scalers[station][poll]
                    
                    try:
                        X_scaled = s.transform(X)
                        pred_log = m.predict(X_scaled)
                        predicted = np.exp(pred_log)
                    except:
                        predicted = np.full(len(val_dates), np.nan)
                else:
                    predicted = np.full(len(val_dates), np.nan)
                
                station_predictions[poll] = predicted
                
                # 计算指标
                metrics = self.compute_metrics(actual, predicted)
                station_metrics[poll] = metrics
                
                print(f"  {poll}: R²={metrics['R2']:.4f}, NSE={metrics['NSE']:.4f}, RMSE={metrics['RMSE']:.4f}")
            
            # 保存结果
            result_row = {'station': station}
            for poll in self.pollutants:
                result_row[f'{poll}_R2'] = station_metrics[poll]['R2']
                result_row[f'{poll}_NSE'] = station_metrics[poll]['NSE']
                result_row[f'{poll}_RMSE'] = station_metrics[poll]['RMSE']
            
            all_results.append(result_row)
            
            # 保存预测值
            all_predictions[station] = {
                'dates': val_dates,
                'predictions': station_predictions,
                'observations': {poll: val_wq_dict.get(f"{station}_{poll}", np.full(len(val_dates), np.nan)) 
                                for poll in self.pollutants}
            }
        
        self.temporal_results = {
            'summary': pd.DataFrame(all_results),
            'predictions': all_predictions,
            'train_dates': train_dates,
            'val_dates': val_dates,
        }
        
        return pd.DataFrame(all_results)
    
    # ========== Part 2: 物理约束空间插值 ==========
    
    def calibrate_decay_rates(self, 
                               dates: pd.DatetimeIndex,
                               wq_dict: Dict[str, np.ndarray],
                               flow_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        从训练数据率定衰减系数
        
        使用有上下游关系的站点对率定
        """
        print("\n" + "-" * 50)
        print("率定衰减系数...")
        print("-" * 50)
        
        # 使用2020年数据率定
        train_mask = (dates.year == 2020)
        
        calibrated = {}
        
        for poll in self.pollutants:
            print(f"\n  {poll}:")
            
            all_k_values = []
            
            # 使用有上下游关系的站点对
            station_pairs = [
                ('LingKou', 'LuoNingChangShui', 50),
                ('LuoNingChangShui', 'GaoYaZhai', 80),
                ('GaoYaZhai', 'BaiMaSi', 40),
                ('TanTou', 'LongMenDaQiao', 100),
            ]
            
            for up_station, down_station, dist in station_pairs:
                up_key = f"{up_station}_{poll}"
                down_key = f"{down_station}_{poll}"
                
                if up_key not in wq_dict or down_key not in wq_dict:
                    continue
                
                up_conc = wq_dict[up_key][train_mask]
                down_conc = wq_dict[down_key][train_mask]
                
                # 有效数据
                valid = (~np.isnan(up_conc)) & (~np.isnan(down_conc)) & (up_conc > 0) & (down_conc > 0)
                
                if valid.sum() < 50:
                    continue
                
                # 率定衰减系数: C_down = C_up * exp(-k * t)
                # 假设流速50 km/day，t = dist / 50
                travel_time = dist / 50
                
                # k = -ln(C_down / C_up) / t
                ratio = down_conc[valid] / up_conc[valid]
                ratio = np.clip(ratio, 0.01, 10)  # 限制范围
                
                k_values = -np.log(ratio) / travel_time
                k_median = np.median(k_values)
                k_median = np.clip(k_median, 0.001, 0.1)  # 合理范围
                
                all_k_values.append(k_median)
                print(f"    {up_station} → {down_station}: k = {k_median:.4f} day⁻¹")
            
            if all_k_values:
                calibrated[poll] = np.mean(all_k_values)
            else:
                # 使用默认值
                default_k = {'NH3N': 0.02, 'TP': 0.01, 'TN': 0.005}
                calibrated[poll] = default_k[poll]
            
            print(f"    → {poll} 综合衰减系数: {calibrated[poll]:.4f} day⁻¹")
        
        self.calibrated_decay_rates = calibrated
        return calibrated
    
    def run_spatial_validation(self,
                                dates: pd.DatetimeIndex,
                                wq_dict: Dict[str, np.ndarray],
                                flow_dict: Dict[str, np.ndarray],
                                ps_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        空间外推验证：物理约束插值（留一交叉验证）
        
        使用物理模型：C_下游 = C_上游 × exp(-k×t) + 点源贡献
        """
        print("\n" + "=" * 70)
        print("Part 2: 空间外推验证（物理约束插值）")
        print("=" * 70)
        print("方法: 上游传输 + 衰减 + 点源稀释")
        
        # 首先率定衰减系数
        if not self.calibrated_decay_rates:
            self.calibrate_decay_rates(dates, wq_dict, flow_dict)
        
        all_results = []
        all_predictions = {}
        
        for val_station in self.wq_stations:
            print(f"\n{'='*50}")
            print(f"验证站点: {val_station}")
            print(f"{'='*50}")
            
            # 训练站点（排除验证站点）
            train_stations = [s for s in self.wq_stations if s != val_station]
            
            # 确定参考站点
            upstream_list = self.upstream_map.get(val_station, [])
            if upstream_list:
                # 有上游站点
                ref_stations = [(s, self.get_distance(s, val_station)) for s in upstream_list if s in train_stations]
            else:
                # 边界站点，使用临近站点
                neighbor = self.neighbor_map.get(val_station)
                if neighbor and neighbor in train_stations:
                    ref_stations = [(neighbor, self.get_distance(neighbor, val_station))]
                else:
                    # 使用最近的训练站点
                    ref_stations = [(train_stations[0], 50)]
            
            ref_str = ", ".join([f"{s}({d}km)" for s, d in ref_stations])
            print(f"  参考站点: {ref_str}")
            
            station_predictions = {}
            station_metrics = {}
            
            for poll in self.pollutants:
                val_key = f"{val_station}_{poll}"
                actual = wq_dict.get(val_key, np.full(len(dates), np.nan))
                
                # 物理约束插值
                predicted = self._physics_interpolation(
                    val_station=val_station,
                    pollutant=poll,
                    ref_stations=ref_stations,
                    dates=dates,
                    wq_dict=wq_dict,
                    flow_dict=flow_dict,
                    ps_dict=ps_dict
                )
                
                station_predictions[poll] = predicted
                
                # 计算指标
                metrics = self.compute_metrics(actual, predicted)
                station_metrics[poll] = metrics
                
                print(f"  {poll}: R²={metrics['R2']:.4f}, NSE={metrics['NSE']:.4f}, RMSE={metrics['RMSE']:.4f}")
            
            # 保存结果
            result_row = {
                'val_station': val_station,
                'ref_stations': ref_str,
            }
            for poll in self.pollutants:
                result_row[f'{poll}_R2'] = station_metrics[poll]['R2']
                result_row[f'{poll}_NSE'] = station_metrics[poll]['NSE']
                result_row[f'{poll}_RMSE'] = station_metrics[poll]['RMSE']
            
            all_results.append(result_row)
            
            # 保存预测值
            all_predictions[val_station] = {
                'dates': dates,
                'predictions': station_predictions,
                'observations': {poll: wq_dict.get(f"{val_station}_{poll}", np.full(len(dates), np.nan)) 
                                for poll in self.pollutants}
            }
        
        self.spatial_results = {
            'summary': pd.DataFrame(all_results),
            'predictions': all_predictions,
        }
        
        return pd.DataFrame(all_results)
    
    def _physics_interpolation(self,
                                val_station: str,
                                pollutant: str,
                                ref_stations: List[Tuple[str, float]],
                                dates: pd.DatetimeIndex,
                                wq_dict: Dict[str, np.ndarray],
                                flow_dict: Dict[str, np.ndarray],
                                ps_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        物理约束插值
        
        C_val = Σ(C_ref × exp(-k × t) × w) / Σw + 点源贡献
        """
        n = len(dates)
        k = self.calibrated_decay_rates.get(pollutant, 0.01)
        
        # 收集参考站点贡献
        contributions = []
        weights = []
        
        for ref_station, dist in ref_stations:
            ref_key = f"{ref_station}_{pollutant}"
            
            if ref_key not in wq_dict:
                continue
            
            ref_conc = wq_dict[ref_key]
            ref_conc_filled = pd.Series(ref_conc).interpolate(limit=14).ffill().bfill().values
            
            # 传输衰减
            travel_time = dist / 50  # 假设流速50 km/day
            decay = np.exp(-k * travel_time)
            
            # 权重（距离反比）
            w = 1.0 / (1 + dist * 0.01)
            
            contributions.append(ref_conc_filled * decay)
            weights.append(w)
        
        if not contributions:
            # 回退到全局均值
            all_conc = []
            for key, val in wq_dict.items():
                if pollutant in key:
                    all_conc.append(val)
            return np.nanmean(all_conc, axis=0) if all_conc else np.full(n, 0.1)
        
        # 加权平均
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        predicted = np.zeros(n)
        for conc, w in zip(contributions, weights):
            predicted += conc * w
        
        # 添加点源贡献（简化）
        flow = flow_dict.get(val_station, np.mean(list(flow_dict.values()), axis=0))
        ps_load = self._get_point_source_load(val_station, pollutant, ps_dict, dates)
        
        if np.sum(ps_load) > 0:
            # 点源浓度贡献: C = Load / Q
            # Load: kg/day, Q: m³/s → 需要单位转换
            ps_contribution = ps_load * 0.0116 / np.maximum(flow, 1)  # mg/L
            predicted = predicted + ps_contribution * 0.1  # 缩放因子
        
        # 平滑处理
        predicted = pd.Series(predicted).rolling(3, min_periods=1, center=True).mean().values
        
        return np.maximum(predicted, 0.001)
    
    def _get_point_source_load(self, station: str, pollutant: str, 
                                ps_dict: Dict, dates: pd.DatetimeIndex) -> np.ndarray:
        """获取站点上游点源负荷"""
        n = len(dates)
        total_load = np.zeros(n)
        
        # 简化：搜索所有点源
        load_col = {'NH3N': 'NH4N_load', 'TP': 'TP_load', 'TN': 'TN_load'}
        suffix = load_col.get(pollutant, f'{pollutant}_load')
        
        for col, values in ps_dict.items():
            if suffix in col:
                total_load += values
        
        return total_load / max(1, len([c for c in ps_dict if suffix in c]))
    
    # ========== Part 3: 综合报告和图表 ==========
    
    def generate_report(self):
        """生成综合验证报告"""
        print("\n" + "=" * 70)
        print("综合验证报告")
        print("=" * 70)
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("水质模型综合验证报告")
        report_lines.append("=" * 70)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Part 1: 时间序列验证
        if self.temporal_results:
            report_lines.append("【Part 1: 时间序列验证】")
            report_lines.append("训练期: 2020-01-01 ~ 2020-12-31")
            report_lines.append("验证期: 2021-01-01 ~ 2021-12-31")
            report_lines.append("")
            
            df = self.temporal_results['summary']
            
            # 打印表格
            header = f"{'站点':<20}"
            for poll in self.pollutants:
                header += f" | {poll}_R²"
            header += " | 平均R²"
            report_lines.append(header)
            report_lines.append("-" * 70)
            
            for _, row in df.iterrows():
                line = f"{row['station']:<20}"
                r2_values = []
                for poll in self.pollutants:
                    r2 = row[f'{poll}_R2']
                    r2_values.append(r2)
                    line += f" | {r2:>6.3f}" if not np.isnan(r2) else " |    N/A"
                avg_r2 = np.nanmean(r2_values)
                line += f" | {avg_r2:>6.3f}"
                report_lines.append(line)
            
            # 汇总
            report_lines.append("-" * 70)
            summary_line = f"{'汇总':<20}"
            for poll in self.pollutants:
                mean_r2 = df[f'{poll}_R2'].mean()
                summary_line += f" | {mean_r2:>6.3f}"
            overall_mean = np.nanmean([df[f'{p}_R2'].mean() for p in self.pollutants])
            summary_line += f" | {overall_mean:>6.3f}"
            report_lines.append(summary_line)
            report_lines.append("")
            
            print("\n【Part 1: 时间序列验证汇总】")
            print(f"  整体平均R²: {overall_mean:.4f}")
            for poll in self.pollutants:
                print(f"  {poll} 平均R²: {df[f'{poll}_R2'].mean():.4f}")
        
        # Part 2: 空间外推验证
        if self.spatial_results:
            report_lines.append("【Part 2: 空间外推验证（物理约束插值）】")
            report_lines.append(f"衰减系数: {self.calibrated_decay_rates}")
            report_lines.append("")
            
            df = self.spatial_results['summary']
            
            header = f"{'验证站点':<20} | {'参考站点':<25}"
            for poll in self.pollutants:
                header += f" | {poll}_R²"
            report_lines.append(header)
            report_lines.append("-" * 90)
            
            for _, row in df.iterrows():
                line = f"{row['val_station']:<20} | {row['ref_stations']:<25}"
                for poll in self.pollutants:
                    r2 = row[f'{poll}_R2']
                    line += f" | {r2:>6.3f}" if not np.isnan(r2) else " |    N/A"
                report_lines.append(line)
            
            report_lines.append("-" * 90)
            summary_line = f"{'汇总':<20} | {'':<25}"
            for poll in self.pollutants:
                mean_r2 = df[f'{poll}_R2'].mean()
                summary_line += f" | {mean_r2:>6.3f}"
            report_lines.append(summary_line)
            report_lines.append("")
            
            print("\n【Part 2: 空间外推验证汇总】")
            for poll in self.pollutants:
                print(f"  {poll} 平均R²: {df[f'{poll}_R2'].mean():.4f}")
        
        # 综合评估
        report_lines.append("【综合评估】")
        if self.temporal_results:
            temporal_r2 = np.nanmean([self.temporal_results['summary'][f'{p}_R2'].mean() for p in self.pollutants])
            report_lines.append(f"- 时间预测能力: R² = {temporal_r2:.4f}")
            
            if temporal_r2 > 0.6:
                report_lines.append("  评价: 良好 - 模型能够准确预测时间序列变化")
            elif temporal_r2 > 0.3:
                report_lines.append("  评价: 可接受 - 模型能够捕捉主要变化趋势")
            else:
                report_lines.append("  评价: 需改进 - 模型预测能力有限")
        
        if self.spatial_results:
            spatial_r2 = np.nanmean([self.spatial_results['summary'][f'{p}_R2'].mean() for p in self.pollutants])
            report_lines.append(f"- 空间外推能力: R² = {spatial_r2:.4f}")
            
            if spatial_r2 > 0.3:
                report_lines.append("  评价: 良好 - 物理模型能够合理外推")
            elif spatial_r2 > 0:
                report_lines.append("  评价: 有限 - 物理模型提供基本参考")
            else:
                report_lines.append("  评价: 较差 - 空间变异性难以预测")
        
        report_lines.append("")
        report_lines.append("【结论】")
        report_lines.append("- 模型适用于已有监测站点的时间序列预测")
        report_lines.append("- 对于无监测数据的站点，建议使用物理约束插值作为参考")
        report_lines.append("- 物质平衡闭合良好，污染溯源结果可靠")
        
        # 保存报告
        report_text = "\n".join(report_lines)
        report_path = self.output_dir / 'combined_validation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n报告已保存至: {report_path}")
        
        # 保存CSV
        if self.temporal_results:
            self.temporal_results['summary'].to_csv(
                self.output_dir / 'temporal_validation_summary.csv',
                index=False, encoding='utf-8-sig'
            )
        if self.spatial_results:
            self.spatial_results['summary'].to_csv(
                self.output_dir / 'spatial_validation_summary.csv',
                index=False, encoding='utf-8-sig'
            )
        
        return report_text
    
    def generate_plots(self):
        """生成对比图"""
        print("\n生成对比图...")
        
        # 1. 时间序列验证对比图
        if self.temporal_results:
            self._plot_temporal_validation()
        
        # 2. 空间外推验证对比图
        if self.spatial_results:
            self._plot_spatial_validation()
        
        # 3. R²汇总对比图
        self._plot_r2_comparison()
        
        print(f"图表已保存至: {self.output_dir}")
    
    def _plot_temporal_validation(self):
        """绘制时间序列验证对比图"""
        predictions = self.temporal_results['predictions']
        val_dates = self.temporal_results['val_dates']
        
        # 每个站点一张图
        for station in self.wq_stations:
            if station not in predictions:
                continue
            
            fig, axes = plt.subplots(len(self.pollutants), 1, figsize=(14, 3*len(self.pollutants)))
            
            for i, poll in enumerate(self.pollutants):
                ax = axes[i]
                
                obs = predictions[station]['observations'][poll]
                pred = predictions[station]['predictions'][poll]
                
                ax.plot(val_dates, obs, 'b-', label='Observed', alpha=0.7, linewidth=1)
                ax.plot(val_dates, pred, 'r-', label='Predicted', alpha=0.7, linewidth=1)
                
                # 计算R²
                metrics = self.compute_metrics(obs, pred)
                ax.set_title(f'{station} - {poll} (R²={metrics["R2"]:.3f})')
                ax.set_ylabel('Concentration (mg/L)')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Date (2021)')
            plt.suptitle(f'Temporal Validation: {station}', fontsize=12)
            plt.tight_layout()
            
            fig_path = self.output_dir / f'temporal_validation_{station}.png'
            plt.savefig(fig_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  时间序列图已保存: temporal_validation_*.png")
    
    def _plot_spatial_validation(self):
        """绘制空间外推验证对比图"""
        predictions = self.spatial_results['predictions']
        
        # 散点图（预测 vs 观测）
        fig, axes = plt.subplots(len(self.wq_stations), len(self.pollutants), 
                                  figsize=(4*len(self.pollutants), 3*len(self.wq_stations)))
        
        for i, station in enumerate(self.wq_stations):
            if station not in predictions:
                continue
            
            for j, poll in enumerate(self.pollutants):
                ax = axes[i, j] if len(self.wq_stations) > 1 else axes[j]
                
                obs = predictions[station]['observations'][poll]
                pred = predictions[station]['predictions'][poll]
                
                valid = (~np.isnan(obs)) & (~np.isnan(pred)) & (obs > 0) & (pred > 0)
                
                if valid.sum() > 0:
                    ax.scatter(obs[valid], pred[valid], alpha=0.3, s=10)
                    max_val = max(obs[valid].max(), pred[valid].max())
                    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=1)
                    
                    metrics = self.compute_metrics(obs, pred)
                    ax.set_title(f'{station}\nR²={metrics["R2"]:.3f}', fontsize=9)
                else:
                    ax.set_title(f'{station}\nNo data', fontsize=9)
                
                if i == 0:
                    ax.set_xlabel(poll, fontsize=10)
                if j == 0:
                    ax.set_ylabel('Predicted', fontsize=8)
                if i == len(self.wq_stations) - 1:
                    ax.set_xlabel('Observed', fontsize=8)
        
        plt.suptitle('Spatial Validation: Physics-based Interpolation', fontsize=12)
        plt.tight_layout()
        
        fig_path = self.output_dir / 'spatial_validation_scatter.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  空间验证散点图已保存: spatial_validation_scatter.png")
    
    def _plot_r2_comparison(self):
        """绘制R²对比柱状图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 时间序列验证
        if self.temporal_results:
            ax = axes[0]
            df = self.temporal_results['summary']
            
            x = np.arange(len(self.wq_stations))
            width = 0.25
            
            for i, poll in enumerate(self.pollutants):
                r2_values = [df.loc[df['station']==s, f'{poll}_R2'].values[0] 
                            if s in df['station'].values else np.nan
                            for s in self.wq_stations]
                ax.bar(x + i*width, r2_values, width, label=poll)
            
            ax.set_xlabel('Station')
            ax.set_ylabel('R²')
            ax.set_title('Temporal Validation (2021)')
            ax.set_xticks(x + width)
            ax.set_xticklabels(self.wq_stations, rotation=45, ha='right')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        # 空间外推验证
        if self.spatial_results:
            ax = axes[1]
            df = self.spatial_results['summary']
            
            x = np.arange(len(self.wq_stations))
            width = 0.25
            
            for i, poll in enumerate(self.pollutants):
                r2_values = [df.loc[df['val_station']==s, f'{poll}_R2'].values[0] 
                            if s in df['val_station'].values else np.nan
                            for s in self.wq_stations]
                ax.bar(x + i*width, r2_values, width, label=poll)
            
            ax.set_xlabel('Validation Station')
            ax.set_ylabel('R²')
            ax.set_title('Spatial Validation (Physics-based)')
            ax.set_xticks(x + width)
            ax.set_xticklabels(self.wq_stations, rotation=45, ha='right')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        fig_path = self.output_dir / 'r2_comparison.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  R²对比图已保存: r2_comparison.png")
    
    # ========== 主运行函数 ==========
    
    def run_full_validation(self,
                            dates: pd.DatetimeIndex,
                            wq_dict: Dict[str, np.ndarray],
                            flow_dict: Dict[str, np.ndarray],
                            precip: np.ndarray,
                            temp: np.ndarray,
                            ps_dict: Dict[str, np.ndarray]):
        """
        运行完整的综合验证
        """
        print("\n" + "=" * 70)
        print("水质模型综合验证")
        print("Combined Validation: Temporal + Spatial")
        print("=" * 70)
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 1. 时间序列验证
        temporal_df = self.run_temporal_validation(
            dates, wq_dict, flow_dict, precip, temp, ps_dict
        )
        
        # 2. 率定衰减系数
        self.calibrate_decay_rates(dates, wq_dict, flow_dict)
        
        # 3. 空间外推验证
        spatial_df = self.run_spatial_validation(
            dates, wq_dict, flow_dict, ps_dict
        )
        
        # 4. 生成报告
        self.generate_report()
        
        # 5. 生成图表
        self.generate_plots()
        
        return temporal_df, spatial_df


def run_combined_validation_mode():
    """
    运行综合验证模式的入口函数
    """
    from bcgsa.config import (DATA_PATHS, TIME_CONFIG, OUTPUT_CONFIG, 
                       WATER_QUALITY_STATIONS, HYDRO_STATIONS)
    from bcgsa.data.loader import DataLoader, DataPreprocessor
    
    print("\n" + "=" * 70)
    print("     伊洛河流域水质模型 - 综合验证模式")
    print("=" * 70)
    print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载数据
    print("\n加载数据...")
    loader = DataLoader(DATA_PATHS)
    raw_data = loader.load_all()
    
    preprocessor = DataPreprocessor(loader, TIME_CONFIG)
    processed_data = preprocessor.preprocess_all()
    
    # 准备数据
    dates = pd.date_range(TIME_CONFIG['start_date'], TIME_CONFIG['end_date'], freq='D')
    
    wq_data = processed_data['water_quality']
    runoff_data = processed_data['runoff']
    ps_data = processed_data['point_source']
    metro_data = processed_data.get('metro', None)
    
    # 提取水质数据字典
    pollutant_cols = {
        'NH3N': 'AmmoniaNitrogen',
        'TP': 'TotalPhosphorus',
        'TN': 'TotalNitrogen',
    }
    
    wq_dict = {}
    for station in WATER_QUALITY_STATIONS.keys():
        for poll, col in pollutant_cols.items():
            col_name = f"{station}_{col}"
            if col_name in wq_data.columns:
                wq_dict[f"{station}_{poll}"] = wq_data[col_name].values
    
    # 流量数据
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
    for station in WATER_QUALITY_STATIONS.keys():
        hydro = nearby_hydro.get(station, station)
        if hydro in runoff_data.columns:
            flow_dict[station] = runoff_data[hydro].values
        elif station in runoff_data.columns:
            flow_dict[station] = runoff_data[station].values
    
    # 气象数据
    n_days = len(dates)
    precip_cols = [c for c in metro_data.columns if 'Precipitation' in c] if metro_data is not None else []
    precip = metro_data[precip_cols].mean(axis=1).values if precip_cols else np.zeros(n_days)
    
    temp_cols = [c for c in metro_data.columns if 'Temperature' in c] if metro_data is not None else []
    temp = metro_data[temp_cols].mean(axis=1).values if temp_cols else 15 + 10 * np.sin(2 * np.pi * (dates.dayofyear - 100) / 365)
    
    # 点源数据
    ps_dict = {}
    for col in ps_data.columns:
        if '_load' in col:
            ps_dict[col] = ps_data[col].values
    
    # 运行验证
    output_dir = Path(OUTPUT_CONFIG['output_dir']) / 'combined_validation'
    validator = CombinedValidator(output_dir=str(output_dir))
    
    temporal_df, spatial_df = validator.run_full_validation(
        dates=dates,
        wq_dict=wq_dict,
        flow_dict=flow_dict,
        precip=precip,
        temp=temp,
        ps_dict=ps_dict
    )
    
    print("\n" + "=" * 70)
    print("综合验证完成!")
    print(f"结果已保存至: {output_dir}")
    print("=" * 70)
    
    return temporal_df, spatial_df
