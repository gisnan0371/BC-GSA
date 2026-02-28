# BC-GSA

# BC-GSA: Budget-Closed Graph-based Source Apportionment

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)

Official code repository for:

> **Nonlinear pollution source switching under extreme rainfall revealed by budget-closed graph-based source apportionment**
>
> *Water Research*, 2025

## Overview

BC-GSA is a three-module cascade framework for watershed-scale pollution source apportionment that enforces algebraic mass-balance closure. Unlike conventional machine learning approaches that treat budget closure as an optimization target, BC-GSA computes nonpoint source (NPS) loads as a deterministic residual, guaranteeing exact budget closure (ε ≡ 0) at every node and every time step.

Applied to the Yiluo River Basin (18,881 km²) during the extreme rainfall of July 2021, the framework reveals that extreme storms trigger a rapid reversal from point-source to nonpoint-source dominance within 72 hours, with basin-scale TP NPS contributions surging from a flood-season average of 41.5% to 89.2%.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     BC-GSA Framework                            │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌─────────────────────┐ │
│  │  Module 1     │──▶│  Module 2     │──▶│  Module 3           │ │
│  │  Hydrological │   │  Ensemble WQ  │   │  Algebraic Mass     │ │
│  │  GAT          │   │  Model        │   │  Balance Layer      │ │
│  │               │   │               │   │                     │ │
│  │  3-layer GAT  │   │  RF + GB +    │   │  L_NPS ≡ L_obs      │ │
│  │  8 attn heads │   │  Huber + Ridge│   │    - L_PS - ΣλL_up  │ │
│  │  128 hidden   │   │  24 features  │   │    + L_decay        │ │
│  └──────────────┘   └──────────────┘   └─────────────────────┘ │
│       ↓ Q(t)              ↓ C(t)              ↓ L_NPS(t)       │
│    Discharge         Concentration        NPS Attribution       │
└─────────────────────────────────────────────────────────────────┘
```

## Repository Structure

```
BC-GSA/
├── bcgsa/                          # Main package
│   ├── config.py                   # Basin configuration & hyperparameters
│   ├── data/
│   │   ├── loader.py               # Data loading and preprocessing
│   │   └── topology.py             # River network graph construction
│   ├── models/
│   │   ├── hydrological_gat.py     # Module 1: Graph Attention Network
│   │   ├── water_quality.py        # Module 2: Ensemble WQ prediction
│   │   ├── mass_balance.py         # Module 3: Algebraic mass balance
│   │   └── node_level_nps.py       # Node-level NPS residual computation
│   ├── validation/
│   │   ├── combined_validator.py   # Temporal + spatial cross-validation
│   │   └── loo_validator.py        # Leave-one-out spatial validation
│   ├── visualization/
│   │   └── plots.py                # Figure generation
│   ├── utils/
│   │   └── transport.py            # Upstream transport & decay coefficients
│   └── apportionment/
│       └── source_contribution.py  # Source contribution analysis
├── scripts/
│   └── run_pipeline.py             # Main entry point
├── data/
│   └── README.md                   # Data access instructions
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/[username]/BC-GSA.git
cd BC-GSA
pip install -r requirements.txt
```

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10
- scikit-learn ≥ 1.0
- NumPy, Pandas, SciPy, Matplotlib

## Usage

### Running the full pipeline

```bash
python scripts/run_pipeline.py \
    --data_dir ./data/input_data \
    --output_dir ./output \
    --pollutants NH3N TP TN
```

### Using individual modules

```python
from bcgsa.models.hydrological_gat import HydroGNN, HydroModelTrainer
from bcgsa.models.water_quality import EnhancedWaterQualityModel
from bcgsa.models.mass_balance import EnhancedMassBalanceEstimator

# Module 1: Hydrological GAT
model = HydroGNN(n_features=12, n_nodes=21, hidden_dim=128, n_layers=3, n_heads=8)

# Module 2: Ensemble WQ (4 base learners)
wq_model = EnhancedWaterQualityModel(stations=stations, pollutants=["TP"])

# Module 3: Algebraic mass balance (NPS = observed - PS - upstream + decay)
mb_estimator = EnhancedMassBalanceEstimator(topology=topo, transport=transport)
```

## Key Model Parameters

| Parameter | Module | Value | Description |
|-----------|--------|-------|-------------|
| Hidden dimensions | Module 1 (GAT) | 128 | GAT hidden layer size |
| Attention heads | Module 1 (GAT) | 8 | Multi-head attention |
| GAT layers | Module 1 (GAT) | 3 | Network depth |
| Base learners | Module 2 (WQ) | 4 | RF, GB, Huber, Ridge |
| Feature count | Module 2 (WQ) | ≤24 | Per-station predictor features |
| NH₃-N decay | Module 3 (MB) | 0.020 d⁻¹ | First-order in-stream decay |
| TP decay | Module 3 (MB) | 0.010 d⁻¹ | First-order in-stream decay |
| TN decay | Module 3 (MB) | 0.005 d⁻¹ | First-order in-stream decay |

Full hyperparameter details are provided in Text S2 of the Supplementary Information.

## Data Availability

Input monitoring data are subject to data-sharing agreements and cannot be publicly redistributed. See [`data/README.md`](data/README.md) for data format specifications and access instructions.

## Validation Protocol

The framework employs a triangulated validation approach:

1. **Signal-to-Noise Ratio (SNR)**: Block bootstrap (30-day blocks, 1000 iterations) confidence intervals
2. **Null Hypothesis Testing**: Zero NPS forcing → framework should yield L_NPS ≈ 0
3. **Physical Consistency**: Cross-validation against landscape metrics and C-Q slopes

Temporal split: 2020 (training) → 2021 (validation), with the July 2021 extreme event occurring entirely in the held-out set.

## Citation

If you use this code, please cite:

```bibtex
@article{bcgsa2025,
  title={Nonlinear pollution source switching under extreme rainfall revealed by 
         budget-closed graph-based source apportionment},
  journal={#####},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
