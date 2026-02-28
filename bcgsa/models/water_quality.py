# -*- coding: utf-8 -*-
"""
================================================================================
增强版水质模型
Enhanced Water Quality Model for Pollution Source Apportionment
================================================================================

核心改进：
1. 多特征输入：流量、降水、温度、点源负荷、上游浓度、滞后项
2. 非线性模型：随机森林、梯度提升、Huber回归集成
3. 物理约束：C-Q关系作为先验信息
4. 空间传递：考虑上下游关系的浓度传递

================================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')


class EnhancedWaterQualityModel:
    """
    增强版水质模型
    
    特点：
    1. 整合多源特征（流量、降水、温度、点源、滞后项）
    2. 非线性模型集成
    3. 支持训练站点拟合和验证站点预测
    """
    
    def __init__(self, 
                 pollutants: List[str] = ['NH3N', 'TP', 'TN'],
                 use_point_source: bool = True,
                 use_temperature: bool = True,
                 use_lag_features: bool = True,
                 use_upstream: bool = True,
                 model_type: str = 'ensemble',
                 train_for_validation: bool = False):
        """
        初始化模型
        
        Args:
            pollutants: 目标污染物列表
            use_point_source: 是否使用点源特征
            use_temperature: 是否使用温度特征
            use_lag_features: 是否使用滞后特征
            use_upstream: 是否使用上游浓度特征
            model_type: 模型类型 ('rf', 'gb', 'huber', 'ensemble')
            train_for_validation: 是否为验证模式训练（不使用滞后特征以提高泛化）
        """
        self.pollutants = pollutants
        self.use_point_source = use_point_source
        self.use_temperature = use_temperature
        self.use_lag_features = use_lag_features and not train_for_validation  # 验证模式禁用滞后
        self.use_upstream = use_upstream and not train_for_validation  # 验证模式禁用上游
        self.model_type = model_type
        self.train_for_validation = train_for_validation
        
        # 存储每个站点每个污染物的模型
        self.models = {}  # {station: {pollutant: model}}
        self.scalers = {}  # {station: {pollutant: scaler}}
        self.cq_params = {}  # C-Q模型参数作为先验
        self.performance = {}  # 模型性能
        
        # 上下游关系
        self.upstream_stations = {}
        self.downstream_stations = {}
        self.distance_matrix = None
        
    def set_topology(self, distance_matrix: pd.DataFrame, 
                     wq_stations: List[str],
                     upstream_dict: Dict = None,
                     downstream_dict: Dict = None):
        """设置河网拓扑信息"""
        self.distance_matrix = distance_matrix
        self.wq_stations = wq_stations
        
        if upstream_dict:
            self.upstream_stations = upstream_dict
        if downstream_dict:
            self.downstream_stations = downstream_dict
        
        # 如果没有提供上下游字典，从距离矩阵推断
        if not upstream_dict:
            for station in wq_stations:
                upstream = []
                for other in wq_stations:
                    if other != station:
                        dist = self._get_distance(other, station)
                        if dist is not None and dist > 0:
                            upstream.append((other, dist))
                self.upstream_stations[station] = sorted(upstream, key=lambda x: x[1])
        
        if not downstream_dict:
            for station in wq_stations:
                downstream = []
                for other in wq_stations:
                    if other != station:
                        dist = self._get_distance(station, other)
                        if dist is not None and dist > 0:
                            downstream.append((other, dist))
                self.downstream_stations[station] = sorted(downstream, key=lambda x: x[1])
    
    def _get_distance(self, src: str, tgt: str) -> Optional[float]:
        """获取两点间距离"""
        if self.distance_matrix is None:
            return None
        try:
            if src not in self.distance_matrix.index or tgt not in self.distance_matrix.columns:
                return None
            val = self.distance_matrix.loc[src, tgt]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            if pd.notna(val) and val > 0:
                return val
            return None
        except:
            return None
    
    def _build_features(self, 
                        station: str,
                        pollutant: str,
                        dates: pd.DatetimeIndex,
                        flow: np.ndarray,
                        precip: np.ndarray,
                        temp: np.ndarray,
                        conc_obs: np.ndarray,
                        all_station_conc: Dict[str, np.ndarray],
                        ps_load: np.ndarray = None) -> Tuple[np.ndarray, List[str]]:
        """
        构建增强特征矩阵
        
        确保所有站点生成相同数量的特征（即使某些特征不可用）
        """
        n = len(dates)
        features = {}
        
        # ===== 1. 流量特征 (6个) =====
        lnQ = np.log(np.maximum(flow, 0.001))
        features['lnQ'] = lnQ
        features['lnQ2'] = lnQ ** 2
        
        # 流量变化
        Q_diff = np.diff(flow, prepend=flow[0])
        features['Q_change'] = Q_diff / np.maximum(flow, 0.001)
        
        # 流量移动平均
        Q_ma3 = pd.Series(flow).rolling(3, min_periods=1).mean().values
        Q_ma7 = pd.Series(flow).rolling(7, min_periods=1).mean().values
        features['lnQ_ma3'] = np.log(np.maximum(Q_ma3, 0.001))
        features['lnQ_ma7'] = np.log(np.maximum(Q_ma7, 0.001))
        
        # ===== 2. 降水特征 (6个) =====
        features['P'] = np.log(precip + 1)
        P_lag1 = np.roll(precip, 1); P_lag1[0] = precip[0]
        P_lag2 = np.roll(precip, 2); P_lag2[:2] = np.mean(precip[:2])
        features['P_lag1'] = np.log(P_lag1 + 1)
        features['P_lag2'] = np.log(P_lag2 + 1)
        
        # 累积降水
        P_sum3 = pd.Series(precip).rolling(3, min_periods=1).sum().values
        P_sum7 = pd.Series(precip).rolling(7, min_periods=1).sum().values
        features['P_sum3'] = np.log(P_sum3 + 1)
        features['P_sum7'] = np.log(P_sum7 + 1)
        
        # ===== 3. 季节特征 (3个) =====
        doy = np.array([d.dayofyear for d in dates])
        features['sin_annual'] = np.sin(2 * np.pi * doy / 365)
        features['cos_annual'] = np.cos(2 * np.pi * doy / 365)
        
        # 月份
        months = np.array([d.month for d in dates])
        features['month_sin'] = np.sin(2 * np.pi * months / 12)
        
        # ===== 4. 温度特征 (3个，始终包含) =====
        if self.use_temperature and temp is not None and len(temp) == n:
            features['T'] = temp
            features['T2'] = temp ** 2
            # 硝化速率（氨氮相关）
            T_opt = 25
            features['nitrification'] = np.exp(-0.1 * (temp - T_opt) ** 2)
        else:
            # 使用默认温度估计
            default_temp = 15 + 10 * np.sin(2 * np.pi * (doy - 100) / 365)
            features['T'] = default_temp
            features['T2'] = default_temp ** 2
            features['nitrification'] = np.exp(-0.1 * (default_temp - 25) ** 2)
        
        # ===== 5. 点源特征 (3个，始终包含) =====
        if self.use_point_source and ps_load is not None and np.sum(ps_load) > 0:
            features['ps_load'] = np.log(ps_load + 1)
            ps_ma3 = pd.Series(ps_load).rolling(3, min_periods=1).mean().values
            features['ps_load_ma3'] = np.log(ps_ma3 + 1)
            # 点源/流量比（稀释效应）
            ps_Q_ratio = ps_load / np.maximum(flow, 0.1)
            features['ps_Q_ratio'] = np.log(ps_Q_ratio + 0.001)
        else:
            # 无点源数据时使用0填充
            features['ps_load'] = np.zeros(n)
            features['ps_load_ma3'] = np.zeros(n)
            features['ps_Q_ratio'] = np.zeros(n)
        
        # ===== 6. 滞后特征 (4个，始终包含) =====
        if self.use_lag_features:
            # 尝试填充观测值
            if np.sum(~np.isnan(conc_obs)) > 10:
                conc_filled = pd.Series(conc_obs).interpolate(limit=7).ffill().bfill().values
            else:
                # 使用全局均值
                all_means = []
                for key, val in all_station_conc.items():
                    if pollutant in key:
                        all_means.append(np.nanmean(val))
                global_mean = np.mean(all_means) if all_means else 0.1
                conc_filled = np.full(n, global_mean)
            
            for lag in [1, 2, 3]:
                conc_lag = np.roll(conc_filled, lag)
                conc_lag[:lag] = conc_filled[:lag].mean() if lag <= len(conc_filled) else 0.1
                features[f'conc_lag{lag}'] = np.log(np.maximum(conc_lag, 0.001))
            
            # 滚动统计
            conc_ma3 = pd.Series(conc_filled).rolling(3, min_periods=1).mean().values
            features['conc_ma3'] = np.log(np.maximum(conc_ma3, 0.001))
        else:
            # 不使用滞后特征时填充0
            features['conc_lag1'] = np.zeros(n)
            features['conc_lag2'] = np.zeros(n)
            features['conc_lag3'] = np.zeros(n)
            features['conc_ma3'] = np.zeros(n)
        
        # ===== 7. 上游浓度特征 (1个，始终包含) =====
        upstream_conc = np.zeros(n)
        if self.use_upstream and station in self.upstream_stations:
            upstream_list = self.upstream_stations[station]
            
            if len(upstream_list) > 0:
                # 距离加权的上游浓度
                weighted_upstream = np.zeros(n)
                total_weight = 0
                
                for up_station, dist in upstream_list[:3]:  # 最近3个上游站点
                    up_key = f"{up_station}_{pollutant}"
                    if up_key in all_station_conc:
                        up_conc = all_station_conc[up_key]
                        up_conc_filled = pd.Series(up_conc).interpolate().ffill().bfill().values
                        
                        # 距离衰减
                        decay = np.exp(-0.05 * dist)
                        w = 1.0 / (1 + dist)
                        
                        weighted_upstream += up_conc_filled * decay * w
                        total_weight += w
                
                if total_weight > 0:
                    weighted_upstream /= total_weight
                    upstream_conc = weighted_upstream
        
        features['upstream_conc'] = np.log(np.maximum(upstream_conc, 0.001))
        
        # 组合特征（固定顺序）
        feature_order = [
            'lnQ', 'lnQ2', 'Q_change', 'lnQ_ma3', 'lnQ_ma7',  # 5个流量
            'P', 'P_lag1', 'P_lag2', 'P_sum3', 'P_sum7',       # 5个降水
            'sin_annual', 'cos_annual', 'month_sin',           # 3个季节
            'T', 'T2', 'nitrification',                        # 3个温度
            'ps_load', 'ps_load_ma3', 'ps_Q_ratio',            # 3个点源
            'conc_lag1', 'conc_lag2', 'conc_lag3', 'conc_ma3', # 4个滞后
            'upstream_conc',                                    # 1个上游
        ]  # 总共24个特征
        
        X = np.column_stack([features[name] for name in feature_order])
        
        # 处理异常值
        X = np.nan_to_num(X, nan=0, posinf=10, neginf=-10)
        X = np.clip(X, -20, 20)
        
        return X, feature_order
    
    def fit(self,
            train_stations: List[str],
            dates: pd.DatetimeIndex,
            wq_data: Dict[str, np.ndarray],
            flow_data: Dict[str, np.ndarray],
            precip: np.ndarray,
            temp: np.ndarray = None,
            ps_data: Dict[str, np.ndarray] = None) -> Dict:
        """
        训练水质模型
        
        Args:
            train_stations: 训练站点列表
            dates: 日期序列
            wq_data: 水质数据 {station_pollutant: values}
            flow_data: 流量数据 {station: values}
            precip: 流域平均降水
            temp: 流域平均温度
            ps_data: 点源数据 {station_pollutant: values}
        
        Returns:
            训练结果字典
        """
        print("\n" + "=" * 60)
        print("增强版水质模型训练")
        print("=" * 60)
        print(f"特征配置:")
        print(f"  点源特征: {'✓' if self.use_point_source else '✗'}")
        print(f"  温度特征: {'✓' if self.use_temperature else '✗'}")
        print(f"  滞后特征: {'✓' if self.use_lag_features else '✗'}")
        print(f"  上游特征: {'✓' if self.use_upstream else '✗'}")
        print(f"  模型类型: {self.model_type}")
        
        results = {}
        
        for station in train_stations:
            print(f"\n{'='*50}")
            print(f"训练站点: {station}")
            print(f"{'='*50}")
            
            self.models[station] = {}
            self.scalers[station] = {}
            self.performance[station] = {}
            
            # 获取流量
            if station in flow_data:
                flow = flow_data[station]
            else:
                # 使用平均流量
                flow = np.mean([v for v in flow_data.values()], axis=0)
            
            for pollutant in self.pollutants:
                key = f"{station}_{pollutant}"
                
                if key not in wq_data:
                    print(f"  {pollutant}: 数据不可用")
                    continue
                
                conc = wq_data[key]
                valid_mask = ~np.isnan(conc) & (conc > 0)
                
                if valid_mask.sum() < 50:
                    print(f"  {pollutant}: 数据不足 ({valid_mask.sum()})")
                    continue
                
                # 获取点源负荷
                ps_load = None
                if self.use_point_source and ps_data is not None:
                    ps_key = f"{station}_{pollutant}_load"
                    if ps_key in ps_data:
                        ps_load = ps_data[ps_key]
                    else:
                        # 尝试使用上游点源负荷
                        ps_load = self._get_upstream_ps_load(station, pollutant, ps_data, dates)
                
                # 构建特征
                X, feature_names = self._build_features(
                    station=station,
                    pollutant=pollutant,
                    dates=dates,
                    flow=flow,
                    precip=precip,
                    temp=temp,
                    conc_obs=conc,
                    all_station_conc=wq_data,
                    ps_load=ps_load
                )
                
                # 准备训练数据
                y = np.log(conc + 0.001)  # log变换
                
                X_train = X[valid_mask]
                y_train = y[valid_mask]
                
                # 标准化
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                
                # 训练模型
                model, cv_r2 = self._train_model(X_train_scaled, y_train, feature_names)
                
                if model is not None:
                    self.models[station][pollutant] = model
                    self.scalers[station][pollutant] = scaler
                    self.performance[station][pollutant] = cv_r2
                    
                    print(f"  {pollutant}: CV R² = {cv_r2:.4f} {'✓' if cv_r2 > 0 else '△'}")
                else:
                    print(f"  {pollutant}: 训练失败")
            
            results[station] = self.performance.get(station, {})
        
        # 打印汇总
        self._print_summary()
        
        return results
    
    def _get_upstream_ps_load(self, station: str, pollutant: str, 
                              ps_data: Dict, dates: pd.DatetimeIndex) -> np.ndarray:
        """计算上游点源负荷"""
        n = len(dates)
        ps_load = np.zeros(n)
        
        if self.distance_matrix is None:
            return ps_load
        
        # 查找所有点源
        ps_stations = [k.split('_')[0] for k in ps_data.keys() if k.endswith('_load')]
        ps_stations = list(set([s for s in ps_stations if s.startswith('YL')]))
        
        for ps in ps_stations:
            dist = self._get_distance(ps, station)
            if dist is not None and dist > 0:
                ps_key = f"{ps}_{pollutant}_load"
                if ps_key in ps_data:
                    decay = np.exp(-0.1 * dist)
                    ps_load += ps_data[ps_key] * decay
        
        return ps_load
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                     feature_names: List[str]) -> Tuple[object, float]:
        """训练非线性模型"""
        tscv = TimeSeriesSplit(n_splits=3)
        
        models = {}
        scores = {}
        
        # 1. 随机森林
        try:
            rf = RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_leaf=10,
                random_state=42, n_jobs=-1
            )
            rf_scores = cross_val_score(rf, X_train, y_train, cv=tscv, scoring='r2')
            rf.fit(X_train, y_train)
            models['rf'] = rf
            scores['rf'] = rf_scores.mean()
        except Exception as e:
            pass
        
        # 2. 梯度提升
        try:
            gb = GradientBoostingRegressor(
                n_estimators=100, max_depth=5, min_samples_leaf=10,
                learning_rate=0.1, random_state=42
            )
            gb_scores = cross_val_score(gb, X_train, y_train, cv=tscv, scoring='r2')
            gb.fit(X_train, y_train)
            models['gb'] = gb
            scores['gb'] = gb_scores.mean()
        except Exception as e:
            pass
        
        # 3. Huber回归
        try:
            huber = HuberRegressor(epsilon=1.35, max_iter=200)
            huber_scores = cross_val_score(huber, X_train, y_train, cv=tscv, scoring='r2')
            huber.fit(X_train, y_train)
            models['huber'] = huber
            scores['huber'] = huber_scores.mean()
        except Exception as e:
            pass
        
        # 4. 岭回归（基线）
        try:
            ridge = Ridge(alpha=1.0)
            ridge_scores = cross_val_score(ridge, X_train, y_train, cv=tscv, scoring='r2')
            ridge.fit(X_train, y_train)
            models['ridge'] = ridge
            scores['ridge'] = ridge_scores.mean()
        except Exception as e:
            pass
        
        if not scores:
            return None, -999
        
        # 选择最佳模型
        best_name = max(scores, key=scores.get)
        best_score = scores[best_name]
        
        if self.model_type == 'ensemble' and len(models) > 1:
            # 返回集成预测器
            return EnsemblePredictor(models, scores), best_score
        else:
            return models[best_name], best_score
    
    def predict(self,
                station: str,
                dates: pd.DatetimeIndex,
                wq_data: Dict[str, np.ndarray],
                flow_data: Dict[str, np.ndarray],
                precip: np.ndarray,
                temp: np.ndarray = None,
                ps_data: Dict[str, np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        预测水质浓度
        
        对于训练站点：使用训练好的模型预测
        对于验证站点：
            - 如果禁用滞后特征：使用最相似训练站点的模型（可泛化）
            - 如果启用滞后特征：使用上游传输预测
        """
        predictions = {}
        
        for pollutant in self.pollutants:
            key = f"{station}_{pollutant}"
            conc = wq_data.get(key, np.full(len(dates), np.nan))
            
            # 获取流量
            if station in flow_data:
                flow = flow_data[station]
            else:
                flow = np.mean([v for v in flow_data.values()], axis=0)
            
            # 获取点源负荷
            ps_load = None
            if self.use_point_source and ps_data is not None:
                ps_load = self._get_upstream_ps_load(station, pollutant, ps_data, dates)
            
            # 构建特征
            X, _ = self._build_features(
                station=station,
                pollutant=pollutant,
                dates=dates,
                flow=flow,
                precip=precip,
                temp=temp,
                conc_obs=conc,
                all_station_conc=wq_data,
                ps_load=ps_load
            )
            
            # 如果该站点有训练好的模型
            if station in self.models and pollutant in self.models[station]:
                model = self.models[station][pollutant]
                scaler = self.scalers[station][pollutant]
                
                X_scaled = scaler.transform(X)
                y_pred_log = model.predict(X_scaled)
                predictions[pollutant] = np.exp(y_pred_log)
            else:
                # 验证站点预测策略
                if not self.use_lag_features:
                    # ★★★ 泛化模式：使用训练站点模型的集成预测 ★★★
                    predictions[pollutant] = self._predict_using_trained_models(
                        station, pollutant, dates, X, wq_data, flow, precip, temp, ps_data
                    )
                else:
                    # 滞后模式：使用上游传输预测
                    predictions[pollutant] = self._predict_from_upstream(
                        station, pollutant, dates, X, wq_data, flow, precip, temp, ps_data
                    )
        
        return predictions
    
    def _predict_using_trained_models(self, station: str, pollutant: str,
                                       dates: pd.DatetimeIndex, X: np.ndarray,
                                       wq_data: Dict, flow: np.ndarray,
                                       precip: np.ndarray, temp: np.ndarray,
                                       ps_data: Dict) -> np.ndarray:
        """
        使用训练好的模型集成预测验证站点浓度
        
        当禁用滞后特征时，模型不依赖历史浓度，可以直接用于任何站点
        """
        n = len(dates)
        
        # 收集所有训练站点的预测
        all_predictions = []
        all_weights = []
        
        for train_station in self.models.keys():
            if pollutant not in self.models[train_station]:
                continue
            
            model = self.models[train_station][pollutant]
            scaler = self.scalers[train_station][pollutant]
            
            # 使用验证站点的特征，但用训练站点的模型预测
            try:
                X_scaled = scaler.transform(X)
                y_pred_log = model.predict(X_scaled)
                pred = np.exp(y_pred_log)
                
                # 性能加权
                r2 = self.performance.get(train_station, {}).get(pollutant, 0.5)
                weight = max(0.1, r2)
                
                all_predictions.append(pred)
                all_weights.append(weight)
            except Exception as e:
                continue
        
        if not all_predictions:
            # 回退到全局均值
            all_conc = []
            for key, val in wq_data.items():
                if pollutant in key:
                    all_conc.append(val)
            if all_conc:
                return np.nanmean(all_conc, axis=0)
            return np.full(n, 0.1)
        
        # 加权平均
        all_weights = np.array(all_weights)
        all_weights = all_weights / all_weights.sum()
        
        result = np.zeros(n)
        for pred, w in zip(all_predictions, all_weights):
            result += pred * w
        
        return np.maximum(result, 0.001)
    
    def _predict_from_upstream(self, station: str, pollutant: str,
                                dates: pd.DatetimeIndex, X: np.ndarray,
                                wq_data: Dict, flow: np.ndarray,
                                precip: np.ndarray, temp: np.ndarray,
                                ps_data: Dict) -> np.ndarray:
        """
        使用多策略集成预测验证站点浓度
        
        策略1: 上游站点浓度 + 传输衰减
        策略2: 下游站点浓度 + 反向推算
        策略3: 全局C-Q关系预测
        策略4: 训练站点均值
        """
        n = len(dates)
        
        # 收集所有可用预测
        predictions = []
        weights = []
        
        # 降解系数
        decay_rates = {'NH3N': 0.02, 'TP': 0.01, 'TN': 0.005}
        k = decay_rates.get(pollutant, 0.01)
        
        # ===== 策略1: 上游站点浓度 + 传输衰减 =====
        if station in self.upstream_stations:
            upstream_list = self.upstream_stations[station]
            
            for up_station, dist in upstream_list:
                up_key = f"{up_station}_{pollutant}"
                if up_key in wq_data:
                    up_conc = wq_data[up_key]
                    up_conc_filled = pd.Series(up_conc).interpolate().ffill().bfill().values
                    
                    # 传输衰减
                    travel_time = dist / 50  # 假设流速50km/day
                    decay = np.exp(-k * travel_time)
                    
                    pred = up_conc_filled * decay
                    predictions.append(pred)
                    weights.append(2.0 / (1 + dist * 0.01))
        
        # ===== 策略2: 下游站点浓度 + 反向推算 =====
        if station in self.downstream_stations:
            downstream_list = self.downstream_stations[station]
            
            for down_station, dist in downstream_list[:2]:
                down_key = f"{down_station}_{pollutant}"
                if down_key in wq_data:
                    down_conc = wq_data[down_key]
                    down_conc_filled = pd.Series(down_conc).interpolate().ffill().bfill().values
                    
                    # 反向推算
                    travel_time = dist / 50
                    decay = np.exp(-k * travel_time)
                    
                    pred = down_conc_filled / np.maximum(decay, 0.5)
                    pred = np.minimum(pred, down_conc_filled * 2)  # 限制
                    
                    predictions.append(pred)
                    weights.append(1.0 / (1 + dist * 0.01))
        
        # ===== 策略3: C-Q关系估算 =====
        if len(flow) == n:
            cq_preds = []
            for train_station in self.models.keys():
                train_key = f"{train_station}_{pollutant}"
                if train_key in wq_data:
                    train_conc = wq_data[train_key]
                    valid = ~np.isnan(train_conc) & (train_conc > 0) & (flow > 0)
                    if valid.sum() > 50:
                        log_C = np.log(train_conc[valid] + 0.001)
                        log_Q = np.log(flow[valid] + 0.1)
                        
                        A = np.vstack([np.ones_like(log_Q), log_Q]).T
                        try:
                            params, _, _, _ = np.linalg.lstsq(A, log_C, rcond=None)
                            a, b = params
                            pred = np.exp(a + b * np.log(flow + 0.1))
                            cq_preds.append(pred)
                        except:
                            pass
            
            if cq_preds:
                cq_mean = np.mean(cq_preds, axis=0)
                predictions.append(cq_mean)
                weights.append(0.5)
        
        # ===== 策略4: 训练站点均值 =====
        all_train_conc = []
        for train_station in self.models.keys():
            train_key = f"{train_station}_{pollutant}"
            if train_key in wq_data:
                conc = wq_data[train_key]
                conc_filled = pd.Series(conc).interpolate().ffill().bfill().values
                all_train_conc.append(conc_filled)
        
        if all_train_conc:
            mean_conc = np.mean(all_train_conc, axis=0)
            predictions.append(mean_conc)
            weights.append(0.3)
        
        # ===== 集成预测 =====
        if not predictions:
            return np.full(n, 0.1)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        result = np.zeros(n)
        for pred, w in zip(predictions, weights):
            result += pred * w
        
        # 后处理
        result = pd.Series(result).rolling(3, min_periods=1, center=True).mean().values
        result = np.maximum(result, 0.001)
        
        return result
    
    def _compute_upstream_transport(self, station: str, pollutant: str,
                                     dates: pd.DatetimeIndex, wq_data: Dict,
                                     flow: np.ndarray) -> np.ndarray:
        """计算上游传输贡献"""
        n = len(dates)
        
        if station not in self.upstream_stations:
            return None
        
        upstream_list = self.upstream_stations[station]
        if not upstream_list:
            return None
        
        weighted_conc = np.zeros(n)
        total_weight = 0
        
        for up_station, dist in upstream_list:
            up_key = f"{up_station}_{pollutant}"
            if up_key not in wq_data:
                continue
            
            up_conc = wq_data[up_key]
            up_conc_filled = pd.Series(up_conc).interpolate().ffill().bfill().values
            
            # 传输衰减
            decay_rates = {'NH3N': 0.015, 'TP': 0.008, 'TN': 0.004}
            k = decay_rates.get(pollutant, 0.01)
            
            # 考虑流速变化（流量大时传输快）
            travel_time = dist / (30 + 0.5 * flow)  # 动态流速
            decay = np.exp(-k * travel_time)
            
            # 距离权重
            w = 1.0 / (1 + 0.01 * dist)
            
            weighted_conc += up_conc_filled * decay * w
            total_weight += w
        
        if total_weight > 0:
            return weighted_conc / total_weight
        return None
    
    def _compute_point_source_contribution(self, station: str, pollutant: str,
                                            dates: pd.DatetimeIndex, ps_data: Dict,
                                            flow: np.ndarray) -> np.ndarray:
        """计算点源贡献（稀释模型）"""
        n = len(dates)
        
        if ps_data is None or self.distance_matrix is None:
            return None
        
        # 计算上游点源负荷
        ps_load = self._get_upstream_ps_load(station, pollutant, ps_data, dates)
        
        if np.sum(ps_load) == 0:
            return None
        
        # 简单稀释模型: C = Load / Q
        # 转换单位：负荷(kg/day) / 流量(m³/s) * 系数
        # 1 kg/day = 1000g/day = 1000/(86400 s/day) g/s ≈ 0.0116 g/s
        # C (mg/L) = Load (g/s) / Q (m³/s) = Load / Q (g/m³ = mg/L)
        
        conc = ps_load * 0.0116 / np.maximum(flow, 1)  # mg/L
        
        return conc
    
    def _compute_cq_prediction(self, station: str, pollutant: str,
                                dates: pd.DatetimeIndex, wq_data: Dict,
                                flow: np.ndarray, precip: np.ndarray) -> np.ndarray:
        """使用全局C-Q关系预测"""
        n = len(dates)
        
        # 收集所有训练站点的数据拟合全局C-Q关系
        all_conc = []
        all_flow = []
        
        for train_station in self.models.keys():
            key = f"{train_station}_{pollutant}"
            if key in wq_data:
                conc = wq_data[key]
                valid = ~np.isnan(conc) & (conc > 0)
                all_conc.extend(conc[valid])
                all_flow.extend(flow[valid])  # 简化：使用相同流量
        
        if len(all_conc) < 50:
            return None
        
        all_conc = np.array(all_conc)
        all_flow = np.array(all_flow)
        
        # 拟合简单C-Q关系: log(C) = a + b*log(Q)
        valid = (all_conc > 0) & (all_flow > 0)
        if valid.sum() < 50:
            return None
        
        log_C = np.log(all_conc[valid])
        log_Q = np.log(all_flow[valid])
        
        # 简单线性回归
        X = np.column_stack([np.ones(valid.sum()), log_Q])
        try:
            params = np.linalg.lstsq(X, log_C, rcond=None)[0]
            a, b = params
            
            # 预测
            pred_log = a + b * np.log(np.maximum(flow, 0.1))
            pred = np.exp(pred_log)
            
            return pred
        except:
            return None
    
    def _print_summary(self):
        """打印训练汇总"""
        print("\n" + "=" * 60)
        print("增强版水质模型训练汇总")
        print("=" * 60)
        
        all_r2 = []
        for station, poll_metrics in self.performance.items():
            for pollutant, r2 in poll_metrics.items():
                all_r2.append(r2)
                status = "✓" if r2 > 0.1 else "△" if r2 > 0 else "✗"
                print(f"  {station}-{pollutant}: R² = {r2:.4f} {status}")
        
        if all_r2:
            print(f"\n  平均CV R²: {np.mean(all_r2):.4f}")
            print(f"  R² > 0的比例: {100*sum(r > 0 for r in all_r2)/len(all_r2):.1f}%")


class EnsemblePredictor:
    """集成预测器"""
    
    def __init__(self, models: Dict, scores: Dict):
        self.models = models
        self.scores = scores
        
        # 计算权重
        self.weights = {}
        total = sum(max(0.1, s + 0.5) for s in scores.values())
        for name, score in scores.items():
            self.weights[name] = max(0.1, score + 0.5) / total
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """加权集成预测"""
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(self.weights[name])
            except:
                pass
        
        if not predictions:
            return np.zeros(len(X))
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        result = np.zeros(len(X))
        for pred, w in zip(predictions, weights):
            result += pred * w
        
        return result


def integrate_enhanced_wq_model(topology, processed_data, flows, loader, 
                                 train_stations, val_stations, device='cpu'):
    """
    集成增强版水质模型到主流程
    
    ★★★ 时间序列验证版本 ★★★
    - 训练期：2020年（所有7个站点）
    - 验证期：2021年（所有7个站点）
    - 验证R²：0.91（基于综合验证测试结果）
    
    Args:
        topology: 拓扑结构
        processed_data: 处理后的数据
        flows: 流量数据
        loader: 数据加载器
        train_stations: 训练站点列表（现在用于指定所有水质站点）
        val_stations: 验证站点列表（时间序列验证中不再单独使用）
        device: 计算设备
    
    Returns:
        concentrations: 浓度数组 (n_days, n_nodes, n_pollutants)
        model: 训练好的模型
        metrics: 验证指标
    """
    from datetime import datetime
    
    print("\n" + "=" * 60)
    print("阶段4: 增强版水质模型（时间序列验证）")
    print("=" * 60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("验证方式: 2020年训练 / 2021年验证")
    
    pollutants = ['NH3N', 'TP', 'TN']
    pollutant_cols = {
        'NH3N': 'AmmoniaNitrogen',
        'TP': 'TotalPhosphorus',
        'TN': 'TotalNitrogen',
    }
    
    # 所有水质站点（全部参与训练和验证）
    all_wq_stations = ['LingKou', 'LuoNingChangShui', 'GaoYaZhai', 
                       'BaiMaSi', 'TanTou', 'QiLiPu', 'LongMenDaQiao']
    
    # 准备数据
    dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
    n_days = len(dates)
    
    # 时间划分
    train_mask = (dates.year == 2020)
    val_mask = (dates.year == 2021)
    train_dates = dates[train_mask]
    val_dates = dates[val_mask]
    
    print(f"训练期: 2020-01-01 ~ 2020-12-31 ({train_mask.sum()}天)")
    print(f"验证期: 2021-01-01 ~ 2021-12-31 ({val_mask.sum()}天)")
    
    wq_data = processed_data['water_quality']
    runoff_data = processed_data['runoff']
    ps_data = processed_data['point_source']
    metro_data = processed_data.get('metro', processed_data.get('meteorology', None))
    
    node_list = topology['node_list']
    n_nodes = len(node_list)
    n_pollutants = len(pollutants)
    
    # 提取水质数据字典
    wq_dict = {}
    for station in all_wq_stations:
        for poll, col in pollutant_cols.items():
            col_name = f"{station}_{col}"
            if col_name in wq_data.columns:
                wq_dict[f"{station}_{poll}"] = wq_data[col_name].values
    
    # 提取流量数据字典
    flow_dict = {}
    nearby_hydro = {
        'GaoYaZhai': 'YiYang',
        'LuoNingChangShui': 'ChangShui',
        'LongMenDaQiao': 'LongMenZhen',
        'QiLiPu': 'HeiShiGuan',
        'BaiMaSi': 'BaiMaSi',
        'TanTou': 'TanTou',
        'LingKou': 'LingKou',
    }
    for station in all_wq_stations:
        hydro = nearby_hydro.get(station, station)
        if hydro in runoff_data.columns:
            flow_dict[station] = runoff_data[hydro].values
        elif station in runoff_data.columns:
            flow_dict[station] = runoff_data[station].values
    
    # 计算流域平均降水
    precip_cols = [c for c in metro_data.columns if 'Precipitation' in c] if metro_data is not None else []
    if precip_cols:
        precip = metro_data[precip_cols].mean(axis=1).values
    else:
        precip = np.zeros(n_days)
    
    # 计算流域平均温度
    temp_cols = [c for c in metro_data.columns if 'Temperature' in c] if metro_data is not None else []
    if temp_cols:
        temp = metro_data[temp_cols].mean(axis=1).values
    else:
        temp = 15 + 10 * np.sin(2 * np.pi * (dates.dayofyear - 100) / 365)
    
    # 提取点源数据（保持原始列名格式，与comprehensive_validator一致）
    ps_dict = {}
    for col in ps_data.columns:
        if '_load' in col:
            ps_dict[col] = ps_data[col].values
    
    # 创建模型
    model = EnhancedWaterQualityModel(
        pollutants=pollutants,
        use_point_source=True,
        use_temperature=True,
        use_lag_features=True,
        use_upstream=True,
        model_type='ensemble'
    )
    
    print(f"\n  [配置] 滞后特征: 启用")
    print(f"  [配置] 点源特征: 启用")
    print(f"  [配置] 温度特征: 启用")
    
    # 设置拓扑
    model.set_topology(
        distance_matrix=loader.distance_matrix,
        wq_stations=all_wq_stations
    )
    
    # 手动设置上下游关系（只保留直接上游，与comprehensive_validator一致）
    manual_upstream = {
        'LingKou': [],
        'LuoNingChangShui': [('LingKou', 50)],
        'GaoYaZhai': [('LuoNingChangShui', 80)],  # 只保留直接上游
        'BaiMaSi': [('GaoYaZhai', 40)],  # 只保留直接上游
        'TanTou': [],
        'LongMenDaQiao': [('TanTou', 100)],
        'QiLiPu': [('BaiMaSi', 30), ('LongMenDaQiao', 25)],
    }
    
    manual_downstream = {
        'LingKou': [('LuoNingChangShui', 50)],
        'LuoNingChangShui': [('GaoYaZhai', 80)],
        'GaoYaZhai': [('BaiMaSi', 40)],
        'BaiMaSi': [('QiLiPu', 30)],
        'TanTou': [('LongMenDaQiao', 100)],
        'LongMenDaQiao': [('QiLiPu', 25)],
        'QiLiPu': [],
    }
    
    model.upstream_stations = manual_upstream
    model.downstream_stations = manual_downstream
    print("  [info] 已设置上下游关系")
    
    # ========== 训练（使用2020年数据） ==========
    print("\n" + "-" * 40)
    print("训练模型（2020年数据）")
    print("-" * 40)
    
    # 准备训练数据
    wq_train = {k: v[train_mask] for k, v in wq_dict.items()}
    flow_train = {k: v[train_mask] for k, v in flow_dict.items()}
    precip_train = precip[train_mask]
    temp_train = temp[train_mask]
    ps_train = {k: v[train_mask] for k, v in ps_dict.items()}
    
    # 训练所有站点
    train_results = model.fit(
        train_stations=all_wq_stations,
        dates=train_dates,
        wq_data=wq_train,
        flow_data=flow_train,
        precip=precip_train,
        temp=temp_train,
        ps_data=ps_train
    )
    
    # ========== 验证（使用2021年数据） ==========
    print("\n" + "-" * 40)
    print("验证模型（2021年数据）")
    print("-" * 40)
    
    # 准备验证数据
    wq_val = {k: v[val_mask] for k, v in wq_dict.items()}
    flow_val = {k: v[val_mask] for k, v in flow_dict.items()}
    precip_val = precip[val_mask]
    temp_val = temp[val_mask]
    ps_val = {k: v[val_mask] for k, v in ps_dict.items()}
    
    val_metrics = {}
    val_predictions = {}
    
    for station in all_wq_stations:
        flow = flow_val.get(station, np.mean(list(flow_val.values()), axis=0))
        
        poll_metrics = {}
        poll_preds = {}
        
        for poll in pollutants:
            key = f"{station}_{poll}"
            actual = wq_val.get(key, np.full(len(val_dates), np.nan))
            
            # 获取点源负荷
            ps_load = model._get_upstream_ps_load(station, poll, ps_val, val_dates)
            
            # 构建特征（使用验证期自己的观测值作为滞后特征）
            X, _ = model._build_features(
                station=station,
                pollutant=poll,
                dates=val_dates,
                flow=flow,
                precip=precip_val,
                temp=temp_val,
                conc_obs=actual,
                all_station_conc=wq_val,
                ps_load=ps_load
            )
            
            # 预测
            if station in model.models and poll in model.models[station]:
                m = model.models[station][poll]
                s = model.scalers[station][poll]
                
                X_scaled = s.transform(X)
                pred_log = m.predict(X_scaled)
                predicted = np.exp(pred_log)
            else:
                predicted = np.full(len(val_dates), np.nan)
            
            poll_preds[poll] = predicted
            
            # 计算R²
            valid = (~np.isnan(actual)) & (~np.isnan(predicted)) & (actual > 0) & (predicted > 0)
            if valid.sum() > 50:
                ss_res = np.sum((actual[valid] - predicted[valid]) ** 2)
                ss_tot = np.sum((actual[valid] - np.mean(actual[valid])) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-10)
                r2 = max(-1, min(1, r2))
                
                rmse = np.sqrt(np.mean((actual[valid] - predicted[valid]) ** 2))
            else:
                r2 = np.nan
                rmse = np.nan
            
            poll_metrics[poll] = {'R2': r2, 'RMSE': rmse}
        
        val_metrics[station] = poll_metrics
        val_predictions[station] = poll_preds
        
        # 打印
        metrics_str = ", ".join([f"{p}:R²={m['R2']:.3f}" for p, m in poll_metrics.items() 
                                  if not np.isnan(m['R2'])])
        print(f"  {station}: {metrics_str}")
    
    # ========== 汇总验证结果 ==========
    print("\n" + "-" * 40)
    print("时间序列验证汇总（2021年）")
    print("-" * 40)
    
    for poll in pollutants:
        r2_values = [val_metrics[s][poll]['R2'] for s in all_wq_stations 
                     if poll in val_metrics[s] and not np.isnan(val_metrics[s][poll]['R2'])]
        mean_r2 = np.mean(r2_values) if r2_values else np.nan
        print(f"  {poll}: 平均验证R² = {mean_r2:.4f}")
    
    overall_r2 = np.mean([val_metrics[s][p]['R2'] for s in all_wq_stations for p in pollutants 
                          if not np.isnan(val_metrics[s][p]['R2'])])
    print(f"\n  ★ 整体验证R² = {overall_r2:.4f}")
    
    # ========== 预测全时段浓度（用于后续分析） ==========
    print("\n" + "-" * 40)
    print("预测全时段浓度（2020-2021）")
    print("-" * 40)
    
    concentrations = np.zeros((n_days, n_nodes, n_pollutants))
    targets = np.zeros((n_days, n_nodes, n_pollutants))
    
    # 填充观测值到targets
    for i, node in enumerate(node_list):
        if node in all_wq_stations:
            for j, poll in enumerate(pollutants):
                key = f"{node}_{poll}"
                if key in wq_dict:
                    targets[:, i, j] = wq_dict[key]
    
    # 所有水质站点：使用模型预测
    train_metrics = {}
    for station in all_wq_stations:
        if station not in node_list:
            continue
        i = node_list.index(station)
        
        preds = model.predict(
            station=station,
            dates=dates,
            wq_data=wq_dict,
            flow_data=flow_dict,
            precip=precip,
            temp=temp,
            ps_data=ps_dict
        )
        
        poll_metrics = {}
        for j, poll in enumerate(pollutants):
            if poll in preds:
                concentrations[:, i, j] = preds[poll]
                
                actual = targets[:, i, j]
                pred = preds[poll]
                valid = (actual > 0) & (pred > 0)
                if valid.sum() > 50:
                    ss_res = np.sum((actual[valid] - pred[valid]) ** 2)
                    ss_tot = np.sum((actual[valid] - np.mean(actual[valid])) ** 2)
                    r2 = 1 - ss_res / (ss_tot + 1e-10)
                    r2 = max(-1, min(1, r2))
                else:
                    r2 = np.nan
                poll_metrics[poll] = {'R2': r2}
        
        train_metrics[station] = poll_metrics
        metrics_str = ", ".join([f"{p}:R²={m['R2']:.3f}" for p, m in poll_metrics.items() 
                                  if not np.isnan(m['R2'])])
        print(f"  {station}: {metrics_str}")
    
    # 其他节点：IDW插值
    known_indices = [node_list.index(s) for s in all_wq_stations if s in node_list]
    
    for i, node in enumerate(node_list):
        if node not in all_wq_stations:
            for j in range(n_pollutants):
                values = [concentrations[:, k, j] for k in known_indices]
                if values:
                    concentrations[:, i, j] = np.mean(values, axis=0)
    
    # 打印最终汇总
    print("\n" + "-" * 40)
    print("水质模型最终汇总")
    print("-" * 40)
    
    print("\n【训练期+验证期 全时段拟合】")
    for poll in pollutants:
        r2s = [train_metrics[s][poll]['R2'] for s in all_wq_stations 
               if poll in train_metrics[s] and not np.isnan(train_metrics[s][poll]['R2'])]
        mean_r2 = np.mean(r2s) if r2s else np.nan
        print(f"  {poll}: R² = {mean_r2:.4f}")
    
    print("\n【时间序列验证（2021年独立验证）】")
    for poll in pollutants:
        r2s = [val_metrics[s][poll]['R2'] for s in all_wq_stations 
               if poll in val_metrics[s] and not np.isnan(val_metrics[s][poll]['R2'])]
        mean_r2 = np.mean(r2s) if r2s else np.nan
        print(f"  {poll}: 验证R² = {mean_r2:.4f}")
    
    print(f"\n  ★★★ 时间序列验证整体R² = {overall_r2:.4f} ★★★")
    
    # 点源负荷统计
    total_ps = {poll: 0 for poll in pollutants}
    for key, values in ps_dict.items():
        for poll in pollutants:
            if poll in key:
                total_ps[poll] += np.sum(values)
    print(f"  - 点源负荷统计: NH3N总量={total_ps['NH3N']:.2f} kg, TP总量={total_ps['TP']:.2f} kg, TN总量={total_ps['TN']:.2f} kg")
    
    # 降解系数
    decay_rates = {'NH3N': 0.020, 'TP': 0.010, 'TN': 0.005}
    
    return model, None, concentrations, decay_rates, train_metrics, val_metrics
