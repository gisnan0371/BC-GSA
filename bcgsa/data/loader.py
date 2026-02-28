# -*- coding: utf-8 -*-
"""
================================================================================
伊洛河流域污染溯源模型 - 数据加载模块
Yiluo River Basin Pollution Source Apportionment - Data Loader
================================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_paths: Dict[str, Path]):
        """
        初始化数据加载器
        
        Args:
            data_paths: 数据文件路径字典
        """
        self.data_paths = data_paths
        self.water_quality_df = None
        self.runoff_df = None
        self.metro_df = None
        self.point_source_df = None
        self.distance_matrix = None
        
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """加载所有数据"""
        print("=" * 60)
        print("开始加载数据...")
        print("=" * 60)
        
        self.water_quality_df = self._load_water_quality()
        self.runoff_df = self._load_runoff()
        self.metro_df = self._load_metro()
        self.point_source_df = self._load_point_source()
        self.distance_matrix = self._load_distance_matrix()
        
        print("\n" + "=" * 60)
        print("数据加载完成！")
        print("=" * 60)
        
        return {
            "water_quality": self.water_quality_df,
            "runoff": self.runoff_df,
            "metro": self.metro_df,
            "point_source": self.point_source_df,
            "distance_matrix": self.distance_matrix,
        }
    
    def _load_water_quality(self) -> pd.DataFrame:
        """加载水质数据"""
        print("\n[1/5] 加载水质数据...")
        
        path = self.data_paths["water_quality"]
        df = pd.read_csv(path)
        
        # 标准化列名
        if 'pH' in df.columns:
            df = df.rename(columns={'pH': 'Ph'})
        
        # 创建日期列
        df['datetime'] = pd.to_datetime(
            df[['Year', 'Month', 'Day', 'Hour']].astype(str).agg('-'.join, axis=1),
            format='%Y-%m-%d-%H'
        )
        df['date'] = df['datetime'].dt.date
        df['date'] = pd.to_datetime(df['date'])
        
        # 获取站点列表
        station_col = 'WaterQualityMonitoringStation'
        stations = df[station_col].unique().tolist()
        
        print(f"  - 水质站点数: {len(stations)}")
        print(f"  - 站点列表: {stations}")
        print(f"  - 时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
        print(f"  - 记录数: {len(df)}")
        
        return df
    
    def _load_runoff(self) -> pd.DataFrame:
        """加载水文（流量）数据"""
        print("\n[2/5] 加载水文数据...")
        
        path = self.data_paths["runoff"]
        df = pd.read_csv(path)
        
        # 创建日期列
        df['date'] = pd.to_datetime(
            df[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1),
            format='%Y-%m-%d'
        )
        
        # 获取站点列表
        station_col = 'RiverRunoffMonitoringStation'
        stations = df[station_col].unique().tolist()
        
        # 确保流量列为数值类型
        df['Runoff'] = pd.to_numeric(df['Runoff'], errors='coerce')
        
        print(f"  - 水文站点数: {len(stations)}")
        print(f"  - 站点列表: {stations[:10]}..." if len(stations) > 10 else f"  - 站点列表: {stations}")
        print(f"  - 时间范围: {df['date'].min()} ~ {df['date'].max()}")
        print(f"  - 记录数: {len(df)}")
        
        return df
    
    def _load_metro(self) -> pd.DataFrame:
        """加载气象数据"""
        print("\n[3/5] 加载气象数据...")
        
        path = self.data_paths["metro"]
        df = pd.read_csv(path)
        
        # 创建日期时间列
        df['datetime'] = pd.to_datetime(
            df[['Year', 'Month', 'Day', 'Hour']].astype(str).agg('-'.join, axis=1),
            format='%Y-%m-%d-%H'
        )
        df['date'] = df['datetime'].dt.date
        df['date'] = pd.to_datetime(df['date'])
        
        # 获取站点列表
        station_col = 'MetroStationName'
        stations = df[station_col].unique().tolist()
        
        print(f"  - 气象站点数: {len(stations)}")
        print(f"  - 站点列表: {stations}")
        print(f"  - 时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
        print(f"  - 记录数: {len(df)}")
        
        return df
    
    def _load_point_source(self) -> pd.DataFrame:
        """加载点源污染数据"""
        print("\n[4/5] 加载点源污染数据...")
        
        path = self.data_paths["point_source"]
        df = pd.read_csv(path)
        
        # 标准化列名
        if 'discharge(m3/day)' in df.columns:
            df = df.rename(columns={'discharge(m3/day)': 'discharge'})
        
        # 创建日期列
        df['date'] = pd.to_datetime(df['Date'])
        
        # 获取点源列表
        sources = df['N_unit'].unique().tolist()
        
        # 确保数值列为数值类型
        numeric_cols = ['discharge', 'COD', 'COD_load', 'NH4N', 'NH4N_load', 
                       'TN', 'TN_load', 'TP', 'TP_load']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"  - 点源数量: {len(sources)}")
        print(f"  - 点源列表: {sources[:10]}..." if len(sources) > 10 else f"  - 点源列表: {sources}")
        print(f"  - 时间范围: {df['date'].min()} ~ {df['date'].max()}")
        print(f"  - 记录数: {len(df)}")
        
        return df
    
    def _load_distance_matrix(self) -> pd.DataFrame:
        """加载河道距离矩阵"""
        print("\n[5/5] 加载河道距离矩阵...")
        
        path = self.data_paths["distance_matrix"]
        df = pd.read_csv(path, index_col=0)
        
        # 处理重复的列名和行名（水质站和水文站同名的情况）
        # 策略：合并重复的列/行，保留非空值
        if df.columns.duplicated().any():
            print("  - 检测到重复列名，正在合并...")
            duplicated_cols = df.columns[df.columns.duplicated()].unique().tolist()
            print(f"  - 重复列: {duplicated_cols}")
            
            # 对于每个重复的列名，合并多列为一列
            new_df_dict = {}
            processed_cols = set()
            
            for col in df.columns:
                if col in processed_cols:
                    continue
                    
                if df.columns.tolist().count(col) > 1:
                    # 有重复列，取所有同名列的第一个非空值
                    col_data = df.loc[:, col]
                    if isinstance(col_data, pd.DataFrame):
                        # 多列同名，按行取第一个非空值
                        merged_col = col_data.apply(lambda row: row.dropna().iloc[0] if row.notna().any() else np.nan, axis=1)
                    else:
                        merged_col = col_data
                    new_df_dict[col] = merged_col
                else:
                    new_df_dict[col] = df[col]
                processed_cols.add(col)
            
            df = pd.DataFrame(new_df_dict)
        
        # 同样处理重复的行名
        if df.index.duplicated().any():
            print("  - 检测到重复行名，正在合并...")
            duplicated_rows = df.index[df.index.duplicated()].unique().tolist()
            print(f"  - 重复行: {duplicated_rows}")
            
            # 对于每个重复的行名，合并多行为一行
            new_rows = {}
            processed_rows = set()
            
            for idx in df.index:
                if idx in processed_rows:
                    continue
                    
                if df.index.tolist().count(idx) > 1:
                    # 有重复行，取所有同名行的第一个非空值
                    row_data = df.loc[idx, :]
                    if isinstance(row_data, pd.DataFrame):
                        merged_row = row_data.apply(lambda col: col.dropna().iloc[0] if col.notna().any() else np.nan, axis=0)
                    else:
                        merged_row = row_data
                    new_rows[idx] = merged_row
                else:
                    new_rows[idx] = df.loc[idx, :]
                processed_rows.add(idx)
            
            df = pd.DataFrame(new_rows).T
        
        # 统计信息
        n_nodes = len(df)
        n_valid = df.notna().sum().sum() - n_nodes  # 减去对角线
        
        print(f"  - 矩阵维度: {df.shape}")
        print(f"  - 节点数: {n_nodes}")
        print(f"  - 有效距离数（非对角线）: {n_valid}")
        
        return df
    
    def get_station_list(self, station_type: str) -> list:
        """获取站点列表"""
        if station_type == "water_quality":
            return self.water_quality_df['WaterQualityMonitoringStation'].unique().tolist()
        elif station_type == "hydro":
            return self.runoff_df['RiverRunoffMonitoringStation'].unique().tolist()
        elif station_type == "metro":
            return self.metro_df['MetroStationName'].unique().tolist()
        elif station_type == "point_source":
            return self.point_source_df['N_unit'].unique().tolist()
        else:
            raise ValueError(f"未知站点类型: {station_type}")
    
    def get_date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """获取数据的公共日期范围"""
        # 找到所有数据的公共时间范围
        starts = []
        ends = []
        
        if self.water_quality_df is not None:
            starts.append(self.water_quality_df['date'].min())
            ends.append(self.water_quality_df['date'].max())
        
        if self.runoff_df is not None:
            starts.append(self.runoff_df['date'].min())
            ends.append(self.runoff_df['date'].max())
        
        if self.point_source_df is not None:
            starts.append(self.point_source_df['date'].min())
            ends.append(self.point_source_df['date'].max())
        
        common_start = max(starts)
        common_end = min(ends)
        
        return common_start, common_end


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, data_loader: DataLoader, time_config: dict):
        """
        初始化预处理器
        
        Args:
            data_loader: 数据加载器实例
            time_config: 时间配置
        """
        self.data_loader = data_loader
        self.time_config = time_config
        self.start_date = pd.to_datetime(time_config["start_date"])
        self.end_date = pd.to_datetime(time_config["end_date"])
        self.dates = pd.date_range(self.start_date, self.end_date, freq='D')
        
    def preprocess_all(self) -> Dict[str, pd.DataFrame]:
        """预处理所有数据"""
        print("\n" + "=" * 60)
        print("开始数据预处理...")
        print("=" * 60)
        print(f"目标时间范围: {self.start_date.date()} ~ {self.end_date.date()}")
        print(f"共 {len(self.dates)} 天")
        
        # 1. 水质数据聚合为日均值
        wq_daily = self._aggregate_water_quality()
        
        # 2. 水文数据对齐
        runoff_daily = self._align_runoff()
        
        # 3. 气象数据聚合为日均值/日累计
        metro_daily = self._aggregate_metro()
        
        # 4. 点源数据对齐
        ps_daily = self._align_point_source()
        
        print("\n" + "=" * 60)
        print("数据预处理完成！")
        print("=" * 60)
        
        return {
            "water_quality": wq_daily,
            "runoff": runoff_daily,
            "metro": metro_daily,
            "point_source": ps_daily,
        }
    
    def _aggregate_water_quality(self) -> pd.DataFrame:
        """聚合水质数据为日均值"""
        print("\n[1/4] 聚合水质数据为日均值...")
        
        df = self.data_loader.water_quality_df.copy()
        
        # 筛选时间范围
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        
        # 按站点和日期聚合
        numeric_cols = ['WaterTemperature', 'Ph', 'DO', 'ElectricConductance',
                       'Turbidity', 'AmmoniaNitrogen', 'TotalPhosphorus', 'TotalNitrogen']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        daily = df.groupby(['WaterQualityMonitoringStation', 'date'])[available_cols].mean()
        daily = daily.reset_index()
        
        # 转换为宽表（每个站点一列）
        result_dict = {'date': self.dates}
        stations = df['WaterQualityMonitoringStation'].unique()
        
        for station in stations:
            station_data = daily[daily['WaterQualityMonitoringStation'] == station]
            station_data = station_data.set_index('date')
            
            for col in available_cols:
                col_name = f"{station}_{col}"
                if col in station_data.columns:
                    result_dict[col_name] = station_data[col].reindex(self.dates).values
        
        result = pd.DataFrame(result_dict)
        
        # 插值填充缺失值
        result = self._interpolate_missing(result)
        
        n_missing = result.isna().sum().sum()
        print(f"  - 站点数: {len(stations)}")
        print(f"  - 指标数: {len(available_cols)}")
        print(f"  - 剩余缺失值: {n_missing}")
        
        return result
    
    def _align_runoff(self) -> pd.DataFrame:
        """对齐水文数据"""
        print("\n[2/4] 对齐水文数据...")
        
        df = self.data_loader.runoff_df.copy()
        
        # 筛选时间范围
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        
        # 转换为宽表
        stations = df['RiverRunoffMonitoringStation'].unique()
        result_dict = {'date': self.dates}
        
        for station in stations:
            station_data = df[df['RiverRunoffMonitoringStation'] == station]
            station_data = station_data.set_index('date')
            result_dict[station] = station_data['Runoff'].reindex(self.dates).values
        
        result = pd.DataFrame(result_dict)
        
        # 插值填充缺失值
        result = self._interpolate_missing(result)
        
        n_missing = result.isna().sum().sum()
        print(f"  - 站点数: {len(stations)}")
        print(f"  - 剩余缺失值: {n_missing}")
        
        return result
    
    def _aggregate_metro(self) -> pd.DataFrame:
        """聚合气象数据为日均值/日累计"""
        print("\n[3/4] 聚合气象数据...")
        
        df = self.data_loader.metro_df.copy()
        
        # 筛选时间范围
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        
        # 定义聚合方式
        agg_funcs = {
            'Temperature': 'mean',
            'Pressure': 'mean',
            'RH': 'mean',
            'Wind_Speed': 'mean',
            'Precipitation': 'sum',  # 降水量为日累计
        }
        
        # 按站点和日期聚合
        available_cols = {k: v for k, v in agg_funcs.items() if k in df.columns}
        daily = df.groupby(['MetroStationName', 'date']).agg(available_cols)
        daily = daily.reset_index()
        
        # 转换为宽表
        result_dict = {'date': self.dates}
        stations = df['MetroStationName'].unique()
        
        for station in stations:
            station_data = daily[daily['MetroStationName'] == station]
            station_data = station_data.set_index('date')
            
            for col in available_cols.keys():
                col_name = f"{station}_{col}"
                if col in station_data.columns:
                    result_dict[col_name] = station_data[col].reindex(self.dates).values
        
        result = pd.DataFrame(result_dict)
        
        # 插值填充缺失值
        result = self._interpolate_missing(result)
        
        n_missing = result.isna().sum().sum()
        print(f"  - 站点数: {len(stations)}")
        print(f"  - 剩余缺失值: {n_missing}")
        
        return result
    
    def _align_point_source(self) -> pd.DataFrame:
        """对齐点源数据"""
        print("\n[4/4] 对齐点源数据...")
        
        df = self.data_loader.point_source_df.copy()
        
        # 筛选时间范围
        df = df[(df['date'] >= self.start_date) & (df['date'] <= self.end_date)]
        
        # 转换为宽表
        sources = df['N_unit'].unique()
        pollutant_cols = ['discharge', 'NH4N', 'NH4N_load', 'TN', 'TN_load', 'TP', 'TP_load']
        available_cols = [col for col in pollutant_cols if col in df.columns]
        
        result_dict = {'date': self.dates}
        
        for source in sources:
            source_data = df[df['N_unit'] == source].copy()
            
            # 处理重复日期：按日期分组取均值
            if source_data['date'].duplicated().any():
                source_data = source_data.groupby('date').mean(numeric_only=True).reset_index()
            
            source_data = source_data.set_index('date')
            
            for col in available_cols:
                col_name = f"{source}_{col}"
                if col in source_data.columns:
                    result_dict[col_name] = source_data[col].reindex(self.dates).values
        
        result = pd.DataFrame(result_dict)
        
        # 插值填充缺失值
        result = self._interpolate_missing(result)
        
        n_missing = result.isna().sum().sum()
        print(f"  - 点源数: {len(sources)}")
        print(f"  - 剩余缺失值: {n_missing}")
        
        return result
    
    def _interpolate_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """插值填充缺失值"""
        # 先线性插值
        df = df.interpolate(method='linear', limit_direction='both')
        # 再前后填充
        df = df.ffill().bfill()
        # 仍有缺失的用均值填充
        df = df.fillna(df.mean())
        # 最后用0填充
        df = df.fillna(0)
        return df
    
    def compute_basin_precipitation(self, metro_daily: pd.DataFrame) -> pd.DataFrame:
        """计算流域平均降水量"""
        print("\n计算流域平均降水量...")
        
        # 提取所有站点的降水量列
        precip_cols = [col for col in metro_daily.columns if col.endswith('_Precipitation')]
        
        if len(precip_cols) == 0:
            print("  警告: 未找到降水量数据")
            return pd.DataFrame({'date': self.dates, 'basin_precipitation': 0})
        
        # 计算流域平均（简单平均）
        basin_precip = metro_daily[precip_cols].mean(axis=1)
        
        result = pd.DataFrame({
            'date': self.dates,
            'basin_precipitation': basin_precip.values,
        })
        
        # 添加各站点降水
        for col in precip_cols:
            station = col.replace('_Precipitation', '')
            result[f'{station}_precip'] = metro_daily[col].values
        
        print(f"  - 流域平均降水: {basin_precip.mean():.2f} mm/day")
        print(f"  - 最大日降水: {basin_precip.max():.2f} mm")
        
        return result
    
    def identify_extreme_events(self, precip_df: pd.DataFrame, 
                                percentile: float = 95,
                                min_precip: float = 25.0) -> pd.DataFrame:
        """识别极端降水事件"""
        print(f"\n识别极端降水事件 (P{percentile}, 最小{min_precip}mm)...")
        
        basin_precip = precip_df['basin_precipitation']
        
        # 计算阈值
        threshold = np.percentile(basin_precip[basin_precip > 0], percentile)
        threshold = max(threshold, min_precip)
        
        # 识别极端事件
        is_extreme = basin_precip >= threshold
        
        result = precip_df.copy()
        result['is_extreme'] = is_extreme
        result['extreme_threshold'] = threshold
        
        n_extreme = is_extreme.sum()
        print(f"  - P{percentile}阈值: {threshold:.2f} mm")
        print(f"  - 极端事件天数: {n_extreme}")
        
        return result


if __name__ == "__main__":
    from bcgsa.config import DATA_PATHS, TIME_CONFIG
    
    # 测试数据加载
    loader = DataLoader(DATA_PATHS)
    data = loader.load_all()
    
    # 测试数据预处理
    preprocessor = DataPreprocessor(loader, TIME_CONFIG)
    processed = preprocessor.preprocess_all()
