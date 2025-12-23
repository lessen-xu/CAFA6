"""
Utility modules for CAFA6 competition.

Location: src/utils/__init__.py
"""

from .metrics import CAFAEvaluator, evaluate_batch
from .losses import get_loss_function, WeightedBCELoss, FocalLoss

