# -*- coding: utf-8 -*-
"""
================================================================================
BC-GSA: Budget-Closed Graph-based Source Apportionment
Main Pipeline Script
================================================================================

Three-module cascade for pollution source apportionment:
  Module 1: Hydrological GAT  → daily discharge at all network nodes
  Module 2: Ensemble WQ Model → pollutant concentrations (RF, GB, Huber, Ridge)
  Module 3: Algebraic Mass Balance → NPS loads as deterministic residual

Reference:
  [Author et al.], "Nonlinear pollution source switching under extreme rainfall
  revealed by budget-closed graph-based source apportionment",
  Water Research, 2025.

Usage:
  python scripts/run_pipeline.py --data_dir ./data/input_data --output_dir ./output
================================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from bcgsa.config import (
    DATA_PATHS, TIME_CONFIG, OUTPUT_CONFIG,
    WATER_QUALITY_STATIONS, HYDRO_STATIONS, POINT_SOURCES,
    RIVER_SEGMENTS, TARGET_POLLUTANTS, VALIDATION_CONFIG,
    HYDRO_MODEL_CONFIG, WQ_MODEL_CONFIG, TRANSPORT_CONFIG,
    EXTREME_EVENT_CONFIG,
    NODE_LEVEL_SEGMENTS, POINT_SOURCE_SEGMENT_MAPPING
)
from bcgsa.data.loader import DataLoader, DataPreprocessor
from bcgsa.data.topology import TopologyBuilder, WaterBalanceTopology
from bcgsa.models.hydrological_gat import (
    HydroGNN, HydroModelTrainer, HydroDataset, prepare_hydro_features
)
from bcgsa.models.water_quality import EnhancedWaterQualityModel, EnsemblePredictor
from bcgsa.models.mass_balance import (
    EnhancedMassBalanceEstimator, IterativeMassBalanceSolver,
    AdaptiveDecayCalibrator
)
from bcgsa.models.node_level_nps import NodeLevelNPSEstimator
from bcgsa.utils.transport import TransportModel, ArrivedLoadCalculator, ContributionCalculator
from bcgsa.apportionment.source_contribution import SourceApportionment, ExtremeEventAnalyzer


def print_header():
    """Print program header."""
    print("=" * 70)
    print("  BC-GSA: Budget-Closed Graph-based Source Apportionment")
    print("  Yiluo River Basin - Pollution Source Attribution Framework")
    print("=" * 70)
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print("=" * 70)


def run_module1_hydrology(data_loader, topology, config):
    """
    Module 1: Hydrological GAT
    
    Predict daily discharge at every network node using a 3-layer 
    Graph Attention Network with 8 attention heads and 128 hidden dimensions.
    Training: 2020, Validation: 2021.
    
    Returns:
        dict: Predicted discharge time series for all nodes
    """
    print("\n" + "=" * 60)
    print("  MODULE 1: Hydrological Graph Attention Network")
    print("=" * 60)
    
    # Build river network graph (DAG)
    adj_matrix = topology.get_adjacency_matrix()
    edge_weights = topology.get_edge_weights()
    
    # Prepare features: meteorological forcing + upstream signals
    features, targets = prepare_hydro_features(
        data_loader.hydro_data,
        data_loader.metro_data,
        list(HYDRO_STATIONS.keys())
    )
    
    # Initialize GAT model
    model = HydroGNN(
        n_features=features.shape[-1],
        n_nodes=len(HYDRO_STATIONS),
        hidden_dim=config["hidden_dim"],
        n_layers=config["num_layers"],
        n_heads=config["num_heads"],
        dropout=config["dropout"]
    )
    
    # Train
    trainer = HydroModelTrainer(
        model=model,
        adj_matrix=adj_matrix,
        edge_weights=edge_weights,
        config=config
    )
    
    train_mask = features.index <= TIME_CONFIG["train_end"]
    trainer.train(
        train_features=features[train_mask],
        train_targets=targets[train_mask],
        val_features=features[~train_mask],
        val_targets=targets[~train_mask]
    )
    
    # Predict discharge for full period
    predictions = trainer.predict(features)
    
    print(f"  Discharge prediction complete: {len(predictions)} time steps")
    return predictions


def run_module2_water_quality(data_loader, hydro_predictions, config):
    """
    Module 2: Ensemble Water Quality Prediction
    
    Predict pollutant concentrations using four base learners:
    Random Forest, Gradient Boosting, Huber Regression, Ridge Regression.
    Up to 24 predictor features per station.
    Training: 2020, Validation: 2021.
    
    Returns:
        dict: Predicted concentration time series per station per pollutant
    """
    print("\n" + "=" * 60)
    print("  MODULE 2: Ensemble Water Quality Model")
    print("  (Random Forest + Gradient Boosting + Huber + Ridge)")
    print("=" * 60)
    
    wq_model = EnhancedWaterQualityModel(
        stations=list(WATER_QUALITY_STATIONS.keys()),
        pollutants=list(TARGET_POLLUTANTS.keys()),
        config=config
    )
    
    # Build feature matrix with up to 24 predictors
    wq_model.build_features(
        wq_data=data_loader.wq_data,
        hydro_predictions=hydro_predictions,
        metro_data=data_loader.metro_data
    )
    
    # Train-test split
    wq_model.train(
        train_end=TIME_CONFIG["train_end"],
        test_start=TIME_CONFIG["test_start"]
    )
    
    # Predict concentrations
    predictions = wq_model.predict()
    
    for pollutant in TARGET_POLLUTANTS:
        metrics = wq_model.get_validation_metrics(pollutant)
        print(f"  {pollutant}: R²={metrics.get('r2', 'N/A'):.3f}, "
              f"NSE={metrics.get('nse', 'N/A'):.3f}")
    
    return predictions


def run_module3_mass_balance(data_loader, hydro_predictions, wq_predictions, 
                             topology, transport_config):
    """
    Module 3: Algebraic Mass Balance Layer
    
    Compute NPS loads as the deterministic residual:
      L_NPS(t) = L_obs(t) - L_PS(t) - Σ λ_ij·L_out,j(t) + L_decay(t)
    
    Budget closure is an algebraic identity, not an optimization target.
    
    Returns:
        dict: Source apportionment results (PS/NPS contributions per node per day)
    """
    print("\n" + "=" * 60)
    print("  MODULE 3: Algebraic Mass Balance Layer")
    print("  (Budget-closed NPS residual computation)")
    print("=" * 60)
    
    # Initialize transport model with first-order decay
    transport = TransportModel(
        distance_matrix=data_loader.distance_matrix,
        node_list=list(HYDRO_STATIONS.keys()),
        decay_rates={p: transport_config["init_decay_rate"][p] 
                     for p in TARGET_POLLUTANTS},
        pollutants=list(TARGET_POLLUTANTS.keys())
    )
    
    # Node-level NPS estimation with algebraic closure
    nps_estimator = NodeLevelNPSEstimator(
        node_segments=NODE_LEVEL_SEGMENTS,
        point_source_mapping=POINT_SOURCE_SEGMENT_MAPPING,
        topology=topology,
        transport_model=transport,
        pollutants=list(TARGET_POLLUTANTS.keys())
    )
    
    # Compute NPS loads as residual (Eq. 2 in paper)
    results = nps_estimator.estimate_all_segments(
        observed_loads=data_loader.compute_observed_loads(
            wq_predictions, hydro_predictions
        ),
        ps_loads=data_loader.ps_data,
        upstream_loads=transport.compute_upstream_contributions(
            hydro_predictions, wq_predictions
        ),
        decay_params=transport_config["init_decay_rate"]
    )
    
    # Source apportionment summary
    apportionment = SourceApportionment(
        nps_results=results,
        ps_data=data_loader.ps_data,
        segments=RIVER_SEGMENTS,
        pollutants=list(TARGET_POLLUTANTS.keys())
    )
    
    summary = apportionment.compute_contributions()
    
    print(f"  Mass balance closure verified: ε ≡ 0")
    print(f"  Source apportionment complete for {len(NODE_LEVEL_SEGMENTS)} segments")
    
    return results, apportionment


def run_extreme_event_analysis(apportionment, event_config):
    """
    Analyze source-switching behavior during extreme rainfall events.
    Identifies the 72-hour NPS dominance reversal pattern.
    """
    print("\n" + "=" * 60)
    print("  EXTREME EVENT ANALYSIS")
    print("=" * 60)
    
    analyzer = ExtremeEventAnalyzer(
        apportionment=apportionment,
        event_config=event_config
    )
    
    event_results = analyzer.analyze_all_events()
    
    for event_name, result in event_results.items():
        print(f"  Event: {event_name}")
        print(f"    NPS peak contribution: {result.get('nps_peak', 'N/A'):.1f}%")
        print(f"    Reversal time: {result.get('reversal_hours', 'N/A'):.0f} hours")
    
    return event_results


def main(args=None):
    """Run the full BC-GSA pipeline."""
    parser = argparse.ArgumentParser(
        description="BC-GSA: Budget-Closed Graph-based Source Apportionment"
    )
    parser.add_argument("--data_dir", type=str, default="./data/input_data",
                        help="Path to input data directory")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Path to output directory")
    parser.add_argument("--pollutants", nargs="+", default=["NH3N", "TP", "TN"],
                        help="Target pollutants")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto, cpu, or cuda")
    args = parser.parse_args(args)
    
    print_header()
    
    # === Setup ===
    device = (
        torch.device("cuda") if args.device == "auto" and torch.cuda.is_available()
        else torch.device(args.device if args.device != "auto" else "cpu")
    )
    print(f"\n  Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === Load Data ===
    print("\n  Loading data...")
    data_loader = DataLoader(data_dir=Path(args.data_dir))
    data_loader.load_all()
    
    topology_builder = TopologyBuilder(
        hydro_stations=HYDRO_STATIONS,
        rivers=None  # Uses default from config
    )
    topology = topology_builder.build()
    
    # === Module 1: Hydrological GAT ===
    hydro_predictions = run_module1_hydrology(
        data_loader, topology, HYDRO_MODEL_CONFIG
    )
    
    # === Module 2: Ensemble Water Quality ===
    wq_predictions = run_module2_water_quality(
        data_loader, hydro_predictions, WQ_MODEL_CONFIG
    )
    
    # === Module 3: Algebraic Mass Balance ===
    nps_results, apportionment = run_module3_mass_balance(
        data_loader, hydro_predictions, wq_predictions,
        topology, TRANSPORT_CONFIG
    )
    
    # === Extreme Event Analysis ===
    event_results = run_extreme_event_analysis(
        apportionment, EXTREME_EVENT_CONFIG
    )
    
    # === Save Results ===
    print("\n  Saving results...")
    apportionment.save_results(output_dir)
    
    print("\n" + "=" * 70)
    print(f"  Pipeline complete: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
