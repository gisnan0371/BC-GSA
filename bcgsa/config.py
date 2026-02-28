# -*- coding: utf-8 -*-
"""
================================================================================
伊洛河流域污染溯源模型 - 配置文件
Yiluo River Basin Pollution Source Apportionment - Configuration
================================================================================

v8.3 Bug修复版 (2024)
- 修复R²=1.000虚假验证问题：验证站点使用IDW插值预测
- 修复PBIAS 705-844%物质平衡误差：允许净面源为负值
"""

import os
from pathlib import Path

# =============================================================================
# 1. 数据路径配置
# =============================================================================

# 基础路径（根据实际情况修改）
BASE_DIR = Path(r"./data")
INPUT_DIR = BASE_DIR / "input_data"
OUTPUT_DIR = BASE_DIR / "output_apportionment"

# 数据文件路径
DATA_PATHS = {
    "water_quality": INPUT_DIR / "input_water_quality_data.csv",
    "runoff": INPUT_DIR / "input_runoff_data.csv",
    "metro": INPUT_DIR / "input_metro_data.csv",
    "point_source": INPUT_DIR / "point_source" / "input_source.csv",
    "distance_matrix": INPUT_DIR / "point_source" / "station_distance_matrix_directed.csv",
}

# =============================================================================
# 2. 时间配置
# =============================================================================

TIME_CONFIG = {
    "start_date": "2020-01-01",      # 开始日期（点源数据起始）
    "end_date": "2021-12-31",        # 结束日期
    "train_end": "2020-12-31",       # 训练集结束
    "test_start": "2021-01-01",      # 测试集开始
    "time_resolution": "daily",       # 时间分辨率
}

# =============================================================================
# 3. 站点配置
# =============================================================================

# 水质站点（7个）
WATER_QUALITY_STATIONS = {
    "LingKou": {"river": "LuoRiver", "lon": 110.5439, "lat": 34.0767, "order": 1},
    "LuoNingChangShui": {"river": "LuoRiver", "lon": 111.4467, "lat": 34.3302, "order": 2},
    "GaoYaZhai": {"river": "LuoRiver", "lon": 112.3856, "lat": 34.5992, "order": 3},
    "BaiMaSi": {"river": "LuoRiver", "lon": 112.5978, "lat": 34.7088, "order": 4},
    "TanTou": {"river": "YiRiver", "lon": 111.7392, "lat": 33.9896, "order": 1},
    "LongMenDaQiao": {"river": "YiRiver", "lon": 112.4751, "lat": 34.5295, "order": 2},
    "QiLiPu": {"river": "YiLuoRiver", "lon": 113.0567, "lat": 34.8265, "order": 1},
}

# 水文站点（21个）- 包含河流断面和取水渠
HYDRO_STATIONS = {
    # 伊洛河
    "HeiShiGuan": {"river": "YiLuoRiver", "type": 1, "lon": 112.9333, "lat": 34.7167, "area": 18563},
    # 伊河 - 河流断面 (type=1)
    "LuanChuan": {"river": "YiRiver", "type": 1, "lon": 111.60, "lat": 33.7833, "area": 340},
    "TanTou": {"river": "YiRiver", "type": 1, "lon": 111.7333, "lat": 33.9833, "area": 1695},
    "DongWan": {"river": "YiRiver", "type": 1, "lon": 111.9833, "lat": 34.05, "area": 2623},
    "LuHun": {"river": "YiRiver", "type": 1, "lon": 112.1833, "lat": 34.2, "area": 3492},
    "LuHunQu": {"river": "YiRiver", "type": 1, "lon": 112.1833, "lat": 34.2, "area": 3492},
    "LongMenZhen": {"river": "YiRiver", "type": 1, "lon": 112.4667, "lat": 34.55, "area": 5318},
    # 伊河 - 取水渠 (type=2)
    "BoYunLingQu": {"river": "YiRiver", "type": 2, "lon": 111.75, "lat": 33.9167, "area": 0},
    "YueJinQu": {"river": "YiRiver", "type": 2, "lon": 111.7333, "lat": 33.9833, "area": 0},
    "MaoZhuangQu": {"river": "YiRiver", "type": 2, "lon": 112.1833, "lat": 34.2, "area": 0},
    "LuHunGuangaiQu": {"river": "YiRiver", "type": 2, "lon": 112.1833, "lat": 34.2, "area": 0},
    "YiDongQu": {"river": "YiRiver", "type": 2, "lon": 112.4667, "lat": 34.55, "area": 0},
    # 洛河 - 河流断面 (type=1)
    "LingKou": {"river": "LuoRiver", "type": 1, "lon": 110.4667, "lat": 34.0833, "area": 2476},
    "LuShi": {"river": "LuoRiver", "type": 1, "lon": 111.0667, "lat": 34.05, "area": 4623},
    "ChangShui": {"river": "LuoRiver", "type": 1, "lon": 111.4333, "lat": 34.3167, "area": 6244},
    "YiYang": {"river": "LuoRiver", "type": 1, "lon": 112.1667, "lat": 34.5167, "area": 9713},
    "BaiMaSi": {"river": "LuoRiver", "type": 1, "lon": 112.5833, "lat": 34.7167, "area": 11891},
    # 洛河 - 取水渠 (type=2)
    "LuShiLuoBeiQu": {"river": "LuoRiver", "type": 2, "lon": 111.0667, "lat": 34.05, "area": 0},
    "LuoNingLuoBeiQu": {"river": "LuoRiver", "type": 2, "lon": 111.4333, "lat": 34.3167, "area": 0},
    "YiLuoQu": {"river": "LuoRiver", "type": 2, "lon": 112.1667, "lat": 34.5167, "area": 0},
    "ZhongZhouQu": {"river": "LuoRiver", "type": 2, "lon": 112.4667, "lat": 34.6667, "area": 0},
}

# 气象站点（8个）
METRO_STATIONS = {
    "LuoNing": {"lon": 111.67, "lat": 34.40},
    "XinAn": {"lon": 112.12, "lat": 34.73},
    "MengJin": {"lon": 112.43, "lat": 34.82},
    "YiChuan": {"lon": 112.42, "lat": 34.42},
    "YanShi": {"lon": 112.78, "lat": 34.73},
    "RuYang": {"lon": 112.47, "lat": 34.15},
    "SongXian": {"lon": 112.07, "lat": 34.13},
    "LuoNan": {"lon": 110.15, "lat": 34.10},
}

# 点源（20个）
POINT_SOURCES = [f"YL{10001+i}" for i in range(20)]

# =============================================================================
# 4. 河网拓扑配置
# =============================================================================

# 河流定义
RIVERS = {
    "LuoRiver": {
        "name": "洛河",
        "stations_order": ["LingKou", "LuShi", "ChangShui", "LuoNingChangShui", 
                          "YiYang", "GaoYaZhai", "ZhongZhouQu", "BaiMaSi"],
    },
    "YiRiver": {
        "name": "伊河", 
        "stations_order": ["LuanChuan", "BoYunLingQu", "TanTou", "YueJinQu",
                          "DongWan", "LuHun", "LuHunQu", "MaoZhuangQu", 
                          "LuHunGuangaiQu", "LongMenZhen", "YiDongQu", "LongMenDaQiao"],
    },
    "YiLuoRiver": {
        "name": "伊洛河",
        "stations_order": ["HeiShiGuan", "QiLiPu"],
    },
}

# =============================================================================
# 河段定义 - 精细到节点级别（用于面源分配）
# =============================================================================

# 粗粒度河段定义（用于汇总展示）
RIVER_SEGMENTS = {
    # 洛河河段
    "Luo_Upper": {
        "name": "洛河上游",
        "river": "LuoRiver",
        "start": "source",
        "end": "LuoNingChangShui",
        "stations": ["LingKou", "LuShi", "LuShiLuoBeiQu", "ChangShui", "LuoNingLuoBeiQu", "LuoNingChangShui"],
    },
    "Luo_Middle": {
        "name": "洛河中游",
        "river": "LuoRiver",
        "start": "LuoNingChangShui",
        "end": "GaoYaZhai",
        "stations": ["YiYang", "YiLuoQu", "GaoYaZhai"],
    },
    "Luo_Lower": {
        "name": "洛河下游",
        "river": "LuoRiver",
        "start": "GaoYaZhai",
        "end": "BaiMaSi",
        "stations": ["ZhongZhouQu", "BaiMaSi"],
    },
    # 伊河河段
    "Yi_Upper": {
        "name": "伊河上游",
        "river": "YiRiver",
        "start": "source",
        "end": "TanTou",
        "stations": ["LuanChuan", "BoYunLingQu", "TanTou", "YueJinQu"],
    },
    "Yi_Middle": {
        "name": "伊河中游",
        "river": "YiRiver",
        "start": "TanTou",
        "end": "LuHun",
        "stations": ["DongWan", "LuHun", "LuHunQu", "MaoZhuangQu", "LuHunGuangaiQu"],
    },
    "Yi_Lower": {
        "name": "伊河下游",
        "river": "YiRiver",
        "start": "LuHun",
        "end": "LongMenDaQiao",
        "stations": ["LongMenZhen", "YiDongQu", "LongMenDaQiao"],
    },
    # 伊洛河河段
    "YiLuo": {
        "name": "伊洛河",
        "river": "YiLuoRiver",
        "start": "confluence",
        "end": "QiLiPu",
        "stations": ["HeiShiGuan", "QiLiPu"],
    },
}

# =============================================================================
# 节点级河段定义 - 每两个相邻节点之间定义一个河段（用于精细面源计算）
# =============================================================================

# 洛河节点级河段（上游 → 下游）
LUO_RIVER_NODE_SEGMENTS = [
    {"id": "Luo_seg01", "name": "洛河-灵口上游", "upstream": "source", "downstream": "LingKou", 
     "river": "LuoRiver", "coarse_segment": "Luo_Upper"},
    {"id": "Luo_seg02", "name": "洛河-灵口至卢氏", "upstream": "LingKou", "downstream": "LuShi",
     "river": "LuoRiver", "coarse_segment": "Luo_Upper"},
    {"id": "Luo_seg03", "name": "洛河-卢氏取水渠", "upstream": "LuShi", "downstream": "LuShiLuoBeiQu",
     "river": "LuoRiver", "coarse_segment": "Luo_Upper", "is_diversion": True},
    {"id": "Luo_seg04", "name": "洛河-卢氏至长水", "upstream": "LuShi", "downstream": "ChangShui",
     "river": "LuoRiver", "coarse_segment": "Luo_Upper"},
    {"id": "Luo_seg05", "name": "洛河-洛宁取水渠", "upstream": "ChangShui", "downstream": "LuoNingLuoBeiQu",
     "river": "LuoRiver", "coarse_segment": "Luo_Upper", "is_diversion": True},
    {"id": "Luo_seg06", "name": "洛河-长水至洛宁", "upstream": "ChangShui", "downstream": "LuoNingChangShui",
     "river": "LuoRiver", "coarse_segment": "Luo_Upper"},
    {"id": "Luo_seg07", "name": "洛河-洛宁至宜阳", "upstream": "LuoNingChangShui", "downstream": "YiYang",
     "river": "LuoRiver", "coarse_segment": "Luo_Middle"},
    {"id": "Luo_seg08", "name": "洛河-宜洛取水渠", "upstream": "YiYang", "downstream": "YiLuoQu",
     "river": "LuoRiver", "coarse_segment": "Luo_Middle", "is_diversion": True},
    {"id": "Luo_seg09", "name": "洛河-宜阳至高崖寨", "upstream": "YiYang", "downstream": "GaoYaZhai",
     "river": "LuoRiver", "coarse_segment": "Luo_Middle"},
    {"id": "Luo_seg10", "name": "洛河-高崖寨至中州渠", "upstream": "GaoYaZhai", "downstream": "ZhongZhouQu",
     "river": "LuoRiver", "coarse_segment": "Luo_Lower", "is_diversion": True},
    {"id": "Luo_seg11", "name": "洛河-高崖寨至白马寺", "upstream": "GaoYaZhai", "downstream": "BaiMaSi",
     "river": "LuoRiver", "coarse_segment": "Luo_Lower"},
]

# 伊河节点级河段（上游 → 下游）
YI_RIVER_NODE_SEGMENTS = [
    {"id": "Yi_seg01", "name": "伊河-栾川上游", "upstream": "source", "downstream": "LuanChuan",
     "river": "YiRiver", "coarse_segment": "Yi_Upper"},
    {"id": "Yi_seg02", "name": "伊河-栾川至博云岭渠", "upstream": "LuanChuan", "downstream": "BoYunLingQu",
     "river": "YiRiver", "coarse_segment": "Yi_Upper", "is_diversion": True},
    {"id": "Yi_seg03", "name": "伊河-栾川至潭头", "upstream": "LuanChuan", "downstream": "TanTou",
     "river": "YiRiver", "coarse_segment": "Yi_Upper"},
    {"id": "Yi_seg04", "name": "伊河-潭头跃进渠", "upstream": "TanTou", "downstream": "YueJinQu",
     "river": "YiRiver", "coarse_segment": "Yi_Upper", "is_diversion": True},
    {"id": "Yi_seg05", "name": "伊河-潭头至东湾", "upstream": "TanTou", "downstream": "DongWan",
     "river": "YiRiver", "coarse_segment": "Yi_Middle"},
    {"id": "Yi_seg06", "name": "伊河-东湾至陆浑", "upstream": "DongWan", "downstream": "LuHun",
     "river": "YiRiver", "coarse_segment": "Yi_Middle"},
    {"id": "Yi_seg07", "name": "伊河-陆浑渠口", "upstream": "LuHun", "downstream": "LuHunQu",
     "river": "YiRiver", "coarse_segment": "Yi_Middle", "is_diversion": True},
    {"id": "Yi_seg08", "name": "伊河-毛庄取水渠", "upstream": "LuHun", "downstream": "MaoZhuangQu",
     "river": "YiRiver", "coarse_segment": "Yi_Middle", "is_diversion": True},
    {"id": "Yi_seg09", "name": "伊河-陆浑灌溉渠", "upstream": "LuHun", "downstream": "LuHunGuangaiQu",
     "river": "YiRiver", "coarse_segment": "Yi_Middle", "is_diversion": True},
    {"id": "Yi_seg10", "name": "伊河-陆浑至龙门镇", "upstream": "LuHun", "downstream": "LongMenZhen",
     "river": "YiRiver", "coarse_segment": "Yi_Lower"},
    {"id": "Yi_seg11", "name": "伊河-伊东取水渠", "upstream": "LongMenZhen", "downstream": "YiDongQu",
     "river": "YiRiver", "coarse_segment": "Yi_Lower", "is_diversion": True},
    {"id": "Yi_seg12", "name": "伊河-龙门镇至龙门大桥", "upstream": "LongMenZhen", "downstream": "LongMenDaQiao",
     "river": "YiRiver", "coarse_segment": "Yi_Lower"},
]

# 伊洛河节点级河段
YILUO_RIVER_NODE_SEGMENTS = [
    {"id": "YiLuo_seg01", "name": "伊洛河-汇流至黑石关", "upstream": ["BaiMaSi", "LongMenDaQiao"], "downstream": "HeiShiGuan",
     "river": "YiLuoRiver", "coarse_segment": "YiLuo", "is_confluence": True},
    {"id": "YiLuo_seg02", "name": "伊洛河-黑石关至七里铺", "upstream": "HeiShiGuan", "downstream": "QiLiPu",
     "river": "YiLuoRiver", "coarse_segment": "YiLuo"},
]

# 合并所有节点级河段
NODE_LEVEL_SEGMENTS = LUO_RIVER_NODE_SEGMENTS + YI_RIVER_NODE_SEGMENTS + YILUO_RIVER_NODE_SEGMENTS

# 点源所属河段映射（基于距离矩阵自动计算）
# 格式: {点源ID: 河段ID}
POINT_SOURCE_SEGMENT_MAPPING = {
    # 伊河点源
    "YL10001": "Yi_seg10",   # 伊河: 陆浑 → 龙门镇
    "YL10005": "Yi_seg10",   # 伊河: 陆浑 → 龙门镇
    "YL10008": "Yi_seg03",   # 伊河: 栾川 → 潭头
    "YL10009": "Yi_seg03",   # 伊河: 栾川 → 潭头
    "YL10010": "Yi_seg03",   # 伊河: 栾川 → 潭头
    "YL10016": "Yi_seg10",   # 伊河: 陆浑 → 龙门镇
    "YL10019": "Yi_seg10",   # 伊河: 陆浑 → 龙门镇
    # 洛河点源
    "YL10002": "Luo_seg11",  # 洛河: 高崖寨 → 白马寺
    "YL10003": "Luo_seg11",  # 洛河: 高崖寨 → 白马寺
    "YL10004": "Luo_seg11",  # 洛河: 高崖寨 → 白马寺
    "YL10006": "Luo_seg11",  # 洛河: 高崖寨 → 白马寺
    "YL10007": "Luo_seg11",  # 洛河: 高崖寨 → 白马寺
    "YL10012": "Luo_seg07",  # 洛河: 洛宁 → 宜阳
    "YL10013": "Luo_seg07",  # 洛河: 洛宁 → 宜阳
    "YL10014": "Luo_seg07",  # 洛河: 洛宁 → 宜阳
    "YL10015": "Luo_seg11",  # 洛河: 高崖寨 → 白马寺
    "YL10017": "Luo_seg11",  # 洛河: 高崖寨 → 白马寺
    "YL10018": "Luo_seg11",  # 洛河: 高崖寨 → 白马寺
    # 伊洛河点源
    "YL10011": "YiLuo_seg01",  # 伊洛河: 汇流 → 黑石关
    "YL10020": "YiLuo_seg01",  # 伊洛河: 汇流 → 黑石关
}

# 节点级河段到粗粒度河段的映射
NODE_SEGMENT_TO_COARSE = {seg["id"]: seg["coarse_segment"] for seg in NODE_LEVEL_SEGMENTS}

# 汇流关系（上游站点 -> 下游站点）
CONFLUENCE_RELATIONS = {
    # 洛河汇流
    "LingKou_H": ["LuShi"],
    "LuShi": ["ChangShui"],
    "ChangShui": ["YiYang"],
    "YiYang": ["BaiMaSi"],
    # 伊河汇流
    "LuanChuan": ["TanTou"],
    "TanTou_H": ["DongWan"],
    "DongWan": ["LuHun"],
    "LuHun": ["LongMenZhen"],
    "LongMenZhen": ["LongMenDaQiao"],
    # 伊洛河汇合
    "BaiMaSi": ["HeiShiGuan"],
    "LongMenDaQiao": ["HeiShiGuan"],
    "HeiShiGuan": ["QiLiPu"],
}

# =============================================================================
# 5. 模型配置
# =============================================================================

# 水文模型配置 - 优化版
HYDRO_MODEL_CONFIG = {
    "hidden_dim": 128,           # 增加到128（原64）
    "num_layers": 3,             # 增加到3层（原2）
    "num_heads": 8,              # 增加到8（原4）
    "dropout": 0.2,              # 增加dropout防止过拟合（原0.1）
    "learning_rate": 0.0005,     # 降低学习率（原0.001）
    "weight_decay": 1e-4,        # 增加权重衰减（原1e-5）
    "epochs": 500,               # 增加训练轮数（原200）
    "batch_size": 64,            # 增加batch size（原32）
    "patience": 50,              # 增加耐心值（原30）
    "balance_weight": 0.3,       # 水量平衡损失权重
    "use_residual": True,        # 使用残差连接
    "use_layer_norm": True,      # 使用层归一化
}

# 水质模型配置 - 优化版v2（增强正则化）
WQ_MODEL_CONFIG = {
    "hidden_dim": 64,            # 减小到64，防止过拟合（原128）
    "num_layers": 2,             # 减少到2层（原3）
    "num_heads": 4,              # 减少到4（原8）
    "dropout": 0.3,              # 增加dropout（原0.2）
    "learning_rate": 0.001,      # 适当提高学习率（原0.0005）
    "weight_decay": 5e-4,        # 增加权重衰减（原1e-4）
    "epochs": 300,               # 减少训练轮数（原500）
    "batch_size": 32,            # 减小batch size（原64）
    "patience": 30,              # 减少耐心值（原50）
    "balance_weight": 0.3,       # 物质平衡损失权重
    "decay_reg_weight": 0.2,     # 增加降解系数正则化权重（原0.1）
    "use_residual": True,        # 使用残差连接
    "use_layer_norm": True,      # 使用层归一化
}

# 传输系数配置
TRANSPORT_CONFIG = {
    # 初始降解系数 (1/km)
    "init_decay_rate": {
        "NH3N": 0.02,    # 氨氮降解较快
        "TP": 0.01,      # 总磷中等
        "TN": 0.005,     # 总氮较稳定
    },
    # 降解系数范围约束
    "decay_rate_bounds": {
        "NH3N": (0.005, 0.05),
        "TP": (0.002, 0.03),
        "TN": (0.001, 0.02),
    },
    # 同河流降解系数相似性约束
    "same_river_similarity": 0.1,
}

# =============================================================================
# 6. 验证配置
# =============================================================================

VALIDATION_CONFIG = {
    # 水质验证站点（使用下游站点验证模型泛化能力）
    "validation_wq_stations": ["QiLiPu", "LongMenDaQiao"],
    # 水文验证站点（留出不参与训练）
    "validation_hydro_stations": ["BaiMaSi", "LongMenZhen"],
    # 验证指标阈值
    "r2_threshold": 0.6,
    "nse_threshold": 0.5,
    "mass_balance_error_threshold": 0.15,  # 15%
}

# =============================================================================
# 7. 极端事件配置
# =============================================================================

EXTREME_EVENT_CONFIG = {
    # 极端降水定义
    "precipitation_percentile": 95,  # P95
    "min_precipitation": 25.0,       # 最小阈值 (mm/day)
    # 分析窗口
    "pre_event_days": 3,             # 事件前天数
    "post_event_days": 7,            # 事件后天数
    # 特定事件
    "named_events": {
        "720_rainstorm": {
            "start": "2021-07-18",
            "peak": "2021-07-20",
            "end": "2021-07-25",
            "description": "7.20特大暴雨",
        },
    },
}

# =============================================================================
# 8. 输出配置
# =============================================================================

OUTPUT_CONFIG = {
    "output_dir": OUTPUT_DIR,
    "figures_dir": OUTPUT_DIR / "figures",
    "tables_dir": OUTPUT_DIR / "tables",
    "models_dir": OUTPUT_DIR / "models",
    # 输出内容
    "save_daily_contributions": True,
    "save_monthly_summary": True,
    "save_seasonal_summary": True,
    "save_annual_summary": True,
    # 图表格式
    "figure_format": "png",
    "figure_dpi": 300,
    "table_format": "csv",
}

# =============================================================================
# 9. 目标污染物配置
# =============================================================================

TARGET_POLLUTANTS = {
    "NH3N": {
        "name": "氨氮",
        "name_en": "Ammonia Nitrogen",
        "unit": "mg/L",
        "standard": 1.0,          # III类水标准
        "column_wq": "AmmoniaNitrogen",
        "column_ps": "NH4N",
    },
    "TP": {
        "name": "总磷",
        "name_en": "Total Phosphorus",
        "unit": "mg/L",
        "standard": 0.2,          # III类水标准
        "column_wq": "TotalPhosphorus",
        "column_ps": "TP",
    },
    "TN": {
        "name": "总氮",
        "name_en": "Total Nitrogen",
        "unit": "mg/L",
        "standard": 1.0,          # III类水标准
        "column_wq": "TotalNitrogen",
        "column_ps": "TN",
    },
}

# =============================================================================
# 10. 创建输出目录
# =============================================================================

def create_output_dirs():
    """创建输出目录"""
    for key in ["output_dir", "figures_dir", "tables_dir", "models_dir"]:
        path = OUTPUT_CONFIG[key]
        if not path.exists():
            path.mkdir(parents=True)
            print(f"创建目录: {path}")

if __name__ == "__main__":
    create_output_dirs()
    print("配置加载完成！")
    print(f"数据目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
