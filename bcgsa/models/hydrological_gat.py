# -*- coding: utf-8 -*-
"""
================================================================================
伊洛河流域污染溯源模型 - 水文模型
Yiluo River Basin Pollution Source Apportionment - Hydrological Model
================================================================================

基于GNN的全流域流量估算模型，包含水量平衡约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset, DataLoader


class HydroGATLayer(nn.Module):
    """水文图注意力层"""
    
    def __init__(self, in_features: int, out_features: int, 
                 n_heads: int = 4, dropout: float = 0.1):
        super(HydroGATLayer, self).__init__()
        
        self.n_heads = n_heads
        self.out_features = out_features
        head_dim = out_features // n_heads
        
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * head_dim))
        nn.init.xavier_uniform_(self.a.data)
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor, 
                edge_weights: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 节点特征 [batch, n_nodes, in_features]
            adj: 邻接矩阵 [n_nodes, n_nodes]
            edge_weights: 边权重 [n_nodes, n_nodes]
            
        Returns:
            out: 输出特征 [batch, n_nodes, out_features]
            attention: 注意力权重 [batch, n_nodes, n_nodes]
        """
        batch_size, n_nodes, _ = x.shape
        head_dim = self.out_features // self.n_heads
        
        # 线性变换
        h = self.W(x)  # [batch, n_nodes, out_features]
        h = h.view(batch_size, n_nodes, self.n_heads, head_dim)
        
        # 计算注意力分数
        h_i = h.unsqueeze(2).expand(-1, -1, n_nodes, -1, -1)  # [batch, n_nodes, n_nodes, n_heads, head_dim]
        h_j = h.unsqueeze(1).expand(-1, n_nodes, -1, -1, -1)
        
        cat_h = torch.cat([h_i, h_j], dim=-1)  # [batch, n_nodes, n_nodes, n_heads, 2*head_dim]
        
        # 注意力系数
        e = (cat_h * self.a).sum(dim=-1)  # [batch, n_nodes, n_nodes, n_heads]
        e = self.leaky_relu(e)
        
        # 应用邻接掩码
        mask = adj.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, self.n_heads)
        e = e.masked_fill(mask == 0, float('-inf'))
        
        # 数值稳定的Softmax归一化
        e_max = e.max(dim=2, keepdim=True)[0]
        e_max = torch.where(torch.isinf(e_max), torch.zeros_like(e_max), e_max)
        e_stable = e - e_max
        e_stable = torch.where(torch.isinf(e_stable), torch.full_like(e_stable, -100), e_stable)
        
        attention = F.softmax(e_stable, dim=2)  # [batch, n_nodes, n_nodes, n_heads]
        # 处理全为-inf的行（没有邻居的节点）
        attention = torch.where(torch.isnan(attention), torch.zeros_like(attention), attention)
        attention = self.dropout(attention)
        
        # 应用边权重
        if edge_weights is not None:
            edge_w = edge_weights.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, self.n_heads)
            attention = attention * edge_w
        
        # 聚合
        h = h.permute(0, 2, 1, 3)  # [batch, n_heads, n_nodes, head_dim]
        attention_t = attention.permute(0, 3, 1, 2)  # [batch, n_heads, n_nodes, n_nodes]
        
        out = torch.matmul(attention_t, h)  # [batch, n_heads, n_nodes, head_dim]
        out = out.permute(0, 2, 1, 3).contiguous()  # [batch, n_nodes, n_heads, head_dim]
        out = out.view(batch_size, n_nodes, -1)  # [batch, n_nodes, out_features]
        
        # 返回平均注意力
        avg_attention = attention.mean(dim=-1)  # [batch, n_nodes, n_nodes]
        
        return out, avg_attention


class HydroGNN(nn.Module):
    """水文GNN模型 - 改进版（带残差连接和层归一化）"""
    
    def __init__(self, n_nodes: int, input_dim: int, hidden_dim: int = 64,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1,
                 use_residual: bool = True, use_layer_norm: bool = True):
        """
        初始化水文GNN模型
        
        Args:
            n_nodes: 节点数
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: GAT层数
            num_heads: 注意力头数
            dropout: Dropout比例
            use_residual: 是否使用残差连接
            use_layer_norm: 是否使用层归一化
        """
        super(HydroGNN, self).__init__()
        
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # 节点嵌入
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # GAT层
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        
        for i in range(num_layers):
            in_dim = hidden_dim
            self.gat_layers.append(
                HydroGATLayer(in_dim, hidden_dim, num_heads, dropout)
            )
            if use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(hidden_dim))
        
        # 输出层（多层MLP提高表达能力）
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # 确保流量非负
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # 时序编码（增强时间模式学习）
        self.temporal_encoding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor,
                edge_weights: Optional[torch.Tensor] = None,
                known_mask: Optional[torch.Tensor] = None,
                known_values: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, n_nodes, input_dim]
            adj: 邻接矩阵 [n_nodes, n_nodes]
            edge_weights: 边权重 [n_nodes, n_nodes]
            known_mask: 已知节点掩码 [n_nodes] 或 [batch, n_nodes] (1=已知, 0=未知)
            known_values: 已知节点的流量值 [batch, n_nodes]
            
        Returns:
            outputs: 包含预测流量和注意力的字典
        """
        batch_size = x.shape[0]
        
        # 节点嵌入
        h = self.node_embedding(x)
        
        # 添加时序编码
        temporal_feat = self.temporal_encoding(x)
        h = h + temporal_feat  # 融合时序特征
        
        h = self.dropout(h)
        
        # GAT传播（带残差连接和层归一化）
        attention_weights = []
        for i, gat_layer in enumerate(self.gat_layers):
            h_residual = h  # 保存残差
            h, attn = gat_layer(h, adj, edge_weights)
            h = F.elu(h)
            h = self.dropout(h)
            
            # 残差连接
            if self.use_residual:
                h = h + h_residual
            
            # 层归一化
            if self.use_layer_norm and self.layer_norms is not None:
                h = self.layer_norms[i](h)
            
            attention_weights.append(attn)
        
        # 预测流量
        flow_pred = self.output_layer(h).squeeze(-1)  # [batch, n_nodes]
        
        # 如果有已知值，用已知值替换预测值
        if known_mask is not None and known_values is not None:
            # 处理known_mask的维度
            if known_mask.dim() == 1:
                # [n_nodes] -> [batch, n_nodes]
                known_mask_exp = known_mask.unsqueeze(0).expand(batch_size, -1)
            elif known_mask.dim() == 2:
                # 已经是 [batch, n_nodes]
                known_mask_exp = known_mask
            else:
                # [batch, ..., n_nodes] 取第一个样本的mask
                known_mask_exp = known_mask[0] if known_mask.dim() > 2 else known_mask
                if known_mask_exp.dim() == 1:
                    known_mask_exp = known_mask_exp.unsqueeze(0).expand(batch_size, -1)
            
            flow_pred = torch.where(known_mask_exp.bool(), known_values, flow_pred)
        
        return {
            'flow': flow_pred,
            'attention': torch.stack(attention_weights, dim=1).mean(dim=1),
            'hidden': h,
        }


class WaterBalanceLoss(nn.Module):
    """水量平衡损失函数"""
    
    def __init__(self, topology_info: Dict, node_to_idx: Dict):
        """
        初始化水量平衡损失
        
        Args:
            topology_info: 拓扑信息
            node_to_idx: 节点到索引的映射
        """
        super(WaterBalanceLoss, self).__init__()
        
        self.topology_info = topology_info
        self.node_to_idx = node_to_idx
        
        # 构建平衡方程索引
        self.balance_indices = self._build_balance_indices()
        
    def _build_balance_indices(self) -> List[Dict]:
        """构建平衡方程的索引"""
        equations = []
        
        # 这里需要根据实际拓扑构建方程
        # 简化版：只考虑直接上下游关系
        
        return equations
    
    def forward(self, flow_pred: torch.Tensor, 
                ps_discharge: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算水量平衡损失
        
        Args:
            flow_pred: 预测流量 [batch, n_nodes]
            ps_discharge: 点源排放量 [batch, n_nodes]
            
        Returns:
            loss: 水量平衡损失
        """
        # 简化版：暂时返回0
        # 完整版需要根据拓扑计算各河段的水量平衡误差
        return torch.tensor(0.0, device=flow_pred.device)


class HydroDataset(Dataset):
    """水文数据集"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray,
                 known_mask: np.ndarray, dates: pd.DatetimeIndex):
        """
        初始化数据集
        
        Args:
            features: 节点特征 [n_days, n_nodes, n_features]
            targets: 目标流量 [n_days, n_nodes]
            known_mask: 已知节点掩码 [n_nodes]
            dates: 日期索引
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.known_mask = torch.tensor(known_mask, dtype=torch.float32)
        self.dates = dates
        # 将日期转换为索引，避免DataLoader collate问题
        self.date_indices = np.arange(len(dates))
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'targets': self.targets[idx],
            'known_mask': self.known_mask,
            'date_idx': self.date_indices[idx],  # 返回索引而非Timestamp
        }
    
    def get_date(self, idx):
        """根据索引获取日期"""
        return self.dates[idx]


class HydroModelTrainer:
    """水文模型训练器"""
    
    def __init__(self, model: HydroGNN, adj: np.ndarray, edge_weights: np.ndarray,
                 device: str = 'cpu', learning_rate: float = 0.001,
                 weight_decay: float = 1e-5, balance_weight: float = 0.3):
        """
        初始化训练器
        
        Args:
            model: 水文GNN模型
            adj: 邻接矩阵
            edge_weights: 边权重矩阵
            device: 计算设备
            learning_rate: 学习率
            weight_decay: 权重衰减
            balance_weight: 水量平衡损失权重
        """
        self.model = model.to(device)
        self.device = device
        self.balance_weight = balance_weight
        
        # 转换为张量
        self.adj = torch.tensor(adj, dtype=torch.float32).to(device)
        self.edge_weights = torch.tensor(edge_weights, dtype=torch.float32).to(device)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        
        # 训练历史
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader: DataLoader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in train_loader:
            features = batch['features'].to(self.device)
            targets = batch['targets'].to(self.device)
            known_mask = batch['known_mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播 - 训练时不用已知值替换预测值
            outputs = self.model(
                features, self.adj, self.edge_weights,
                known_mask=None,  # 训练时不替换
                known_values=None
            )
            
            flow_pred = outputs['flow']
            
            # 计算损失（只在已知节点上计算）
            # 处理known_mask维度：可能是[batch, n_nodes]或[n_nodes]
            if known_mask.dim() == 2:
                known_mask_exp = known_mask
            elif known_mask.dim() == 1:
                known_mask_exp = known_mask.unsqueeze(0).expand_as(flow_pred)
            else:
                # 取第一个batch的mask
                known_mask_exp = known_mask[0]
                if known_mask_exp.dim() == 1:
                    known_mask_exp = known_mask_exp.unsqueeze(0).expand_as(flow_pred)
            
            pred_known = flow_pred[known_mask_exp.bool()]
            target_known = targets[known_mask_exp.bool()]
            
            loss = self.mse_loss(pred_known, target_known)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """验证模型"""
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        all_preds = []
        all_targets = []
        saved_known_mask = None
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                targets = batch['targets'].to(self.device)
                known_mask = batch['known_mask'].to(self.device)
                
                # 验证时也不用已知值替换
                outputs = self.model(
                    features, self.adj, self.edge_weights,
                    known_mask=None,
                    known_values=None
                )
                
                flow_pred = outputs['flow']
                
                # 处理known_mask维度
                if known_mask.dim() == 2:
                    known_mask_exp = known_mask
                    # 保存一个1D版本用于后续指标计算
                    if saved_known_mask is None:
                        saved_known_mask = known_mask[0].cpu().numpy()
                elif known_mask.dim() == 1:
                    known_mask_exp = known_mask.unsqueeze(0).expand_as(flow_pred)
                    if saved_known_mask is None:
                        saved_known_mask = known_mask.cpu().numpy()
                else:
                    known_mask_exp = known_mask[0]
                    if known_mask_exp.dim() == 1:
                        known_mask_exp = known_mask_exp.unsqueeze(0).expand_as(flow_pred)
                    if saved_known_mask is None:
                        saved_known_mask = known_mask[0].cpu().numpy()
                        if saved_known_mask.ndim > 1:
                            saved_known_mask = saved_known_mask[0]
                
                # 计算损失
                pred_known = flow_pred[known_mask_exp.bool()]
                target_known = targets[known_mask_exp.bool()]
                
                loss = self.mse_loss(pred_known, target_known)
                total_loss += loss.item()
                n_batches += 1
                
                all_preds.append(flow_pred.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # 计算评估指标
        preds = np.concatenate(all_preds, axis=0)
        targets_np = np.concatenate(all_targets, axis=0)
        
        metrics = self._compute_metrics(preds, targets_np, saved_known_mask)
        
        return total_loss / n_batches, metrics
    
    def _compute_metrics(self, preds: np.ndarray, targets: np.ndarray,
                        known_mask: np.ndarray) -> Dict:
        """计算评估指标"""
        # 只在已知节点上计算
        known_indices = np.where(known_mask > 0)[0]
        
        metrics = {}
        for idx in known_indices:
            pred = preds[:, idx]
            target = targets[:, idx]
            
            # R²
            ss_res = np.sum((target - pred) ** 2)
            ss_tot = np.sum((target - np.mean(target)) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            
            # NSE
            nse = 1 - ss_res / (ss_tot + 1e-10)
            
            # RMSE
            rmse = np.sqrt(np.mean((pred - target) ** 2))
            
            metrics[idx] = {'R2': r2, 'NSE': nse, 'RMSE': rmse}
        
        # 平均指标
        metrics['avg'] = {
            'R2': np.mean([m['R2'] for m in metrics.values() if isinstance(m, dict) and 'R2' in m]),
            'NSE': np.mean([m['NSE'] for m in metrics.values() if isinstance(m, dict) and 'NSE' in m]),
            'RMSE': np.mean([m['RMSE'] for m in metrics.values() if isinstance(m, dict) and 'RMSE' in m]),
        }
        
        return metrics
    
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
            epochs: int = 200, patience: int = 30, verbose: bool = True):
        """训练模型"""
        best_val_loss = float('inf')
        best_model_state = None
        no_improve_count = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            if val_loader is not None:
                val_loss, metrics = self.validate(val_loader)
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = self.model.state_dict().copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] "
                          f"Train Loss: {train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}, "
                          f"Avg R²: {metrics['avg']['R2']:.4f}")
                
                if no_improve_count >= patience:
                    print(f"早停于第 {epoch+1} 轮")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.6f}")
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return best_val_loss
    
    def predict(self, features: np.ndarray, known_mask: np.ndarray,
                known_values: np.ndarray, use_known: bool = True) -> np.ndarray:
        """
        预测流量
        
        Args:
            features: 输入特征
            known_mask: 已知节点掩码
            known_values: 已知节点的真实值
            use_known: 是否用已知值替换预测值（默认True，用于后续计算；False用于验证）
            
        Returns:
            predictions: 预测流量
        """
        self.model.eval()
        
        features_t = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            if use_known:
                known_mask_t = torch.tensor(known_mask, dtype=torch.float32).to(self.device)
                known_values_t = torch.tensor(known_values, dtype=torch.float32).to(self.device)
                outputs = self.model(
                    features_t, self.adj, self.edge_weights,
                    known_mask=known_mask_t,
                    known_values=known_values_t
                )
            else:
                # 纯预测模式，不替换
                outputs = self.model(
                    features_t, self.adj, self.edge_weights,
                    known_mask=None,
                    known_values=None
                )
        
        return outputs['flow'].cpu().numpy()


def prepare_hydro_features(runoff_df: pd.DataFrame, metro_df: pd.DataFrame,
                           node_list: List[str], dates: pd.DatetimeIndex,
                           metro_stations: Dict) -> np.ndarray:
    """
    准备水文模型输入特征
    
    Args:
        runoff_df: 流量数据（宽表）
        metro_df: 气象数据（宽表）
        node_list: 节点列表
        dates: 日期索引
        metro_stations: 气象站点配置
        
    Returns:
        features: 节点特征 [n_days, n_nodes, n_features]
    """
    n_days = len(dates)
    n_nodes = len(node_list)
    
    # 特征维度：流量(1) + 降水(1) + 温度(1) + 时间特征(2) = 5
    n_features = 5
    features = np.zeros((n_days, n_nodes, n_features))
    
    for i, node in enumerate(node_list):
        # 流量特征（如果有）
        if node in runoff_df.columns:
            features[:, i, 0] = runoff_df[node].values
        
        # 气象特征（需要空间插值到节点位置）
        # 简化：使用流域平均值
        precip_cols = [col for col in metro_df.columns if col.endswith('_Precipitation')]
        temp_cols = [col for col in metro_df.columns if col.endswith('_Temperature')]
        
        if precip_cols:
            features[:, i, 1] = metro_df[precip_cols].mean(axis=1).values
        if temp_cols:
            features[:, i, 2] = metro_df[temp_cols].mean(axis=1).values
    
    # 时间特征
    day_of_year = dates.dayofyear.values / 365.0
    month = dates.month.values / 12.0
    
    for i in range(n_nodes):
        features[:, i, 3] = day_of_year
        features[:, i, 4] = month
    
    return features


if __name__ == "__main__":
    # 测试代码
    print("水文模型模块加载成功")
