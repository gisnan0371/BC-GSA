# -*- coding: utf-8 -*-
"""
BC-GSA: Budget-Closed Graph-based Source Apportionment Framework

A three-module cascade for nonpoint source pollution attribution
under extreme rainfall, with algebraic mass-balance closure.

Modules:
    Module 1 - Hydrological GAT: Graph Attention Network for discharge prediction
    Module 2 - Ensemble Water Quality: RF/GB/Huber/Ridge ensemble for concentration
    Module 3 - Algebraic Mass Balance: Deterministic NPS residual computation
"""

__version__ = "1.0.0"
