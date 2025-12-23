"""
Loss functions for CAFA6 competition.
Implements IA-weighted BCE loss and other losses for multi-label classification.

Location: src/utils/losses.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class WeightedBCELoss(nn.Module):
    """
    Binary Cross Entropy loss weighted by Information Accretion (IA).
    
    Terms with higher IA (more specific/rare) get higher weight in the loss.
    """
    
    def __init__(
        self,
        ia_weights: torch.Tensor,
        pos_weight: Optional[torch.Tensor] = None,
        normalize: bool = True
    ):
        """
        Initialize weighted BCE loss.
        
        Args:
            ia_weights: IA weights for each term, shape (num_terms,)
            pos_weight: Optional positive class weights for class imbalance
            normalize: Whether to normalize IA weights
        """
        super().__init__()
        
        if normalize:
            # Normalize IA weights to have mean 1
            ia_weights = ia_weights / (ia_weights.mean() + 1e-10)
        
        self.register_buffer('ia_weights', ia_weights)
        
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.
        
        Args:
            logits: Model outputs before sigmoid, shape (batch, num_terms)
            targets: Binary labels, shape (batch, num_terms)
        
        Returns:
            Scalar loss value
        """
        # Standard BCE with logits
        if self.pos_weight is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets,
                pos_weight=self.pos_weight,
                reduction='none'
            )
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets,
                reduction='none'
            )
        
        # Weight by IA
        weighted_bce = bce * self.ia_weights.unsqueeze(0)
        
        return weighted_bce.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Focuses on hard examples by down-weighting easy examples.
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ia_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Balancing factor for positive class
            gamma: Focusing parameter (higher = more focus on hard examples)
            ia_weights: Optional IA weights for term importance
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
        if ia_weights is not None:
            ia_weights = ia_weights / (ia_weights.mean() + 1e-10)
            self.register_buffer('ia_weights', ia_weights)
        else:
            self.ia_weights = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Loss.
        
        Args:
            logits: Model outputs before sigmoid, shape (batch, num_terms)
            targets: Binary labels, shape (batch, num_terms)
        
        Returns:
            Scalar loss value
        """
        probs = torch.sigmoid(logits)
        
        # Focal weight
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weight
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # BCE
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal loss
        focal_loss = alpha_weight * focal_weight * bce
        
        # Apply IA weights if provided
        if self.ia_weights is not None:
            focal_loss = focal_loss * self.ia_weights.unsqueeze(0)
        
        return focal_loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    
    Applies different focusing to positive and negative samples.
    Reference: Ridnik et al., "Asymmetric Loss For Multi-Label Classification"
    """
    
    def __init__(
        self,
        gamma_pos: float = 0,
        gamma_neg: float = 4,
        clip: float = 0.05,
        ia_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize Asymmetric Loss.
        
        Args:
            gamma_pos: Focusing parameter for positives
            gamma_neg: Focusing parameter for negatives
            clip: Probability clipping threshold
            ia_weights: Optional IA weights
        """
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        
        if ia_weights is not None:
            ia_weights = ia_weights / (ia_weights.mean() + 1e-10)
            self.register_buffer('ia_weights', ia_weights)
        else:
            self.ia_weights = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Asymmetric Loss.
        """
        probs = torch.sigmoid(logits)
        
        # Asymmetric Clipping
        probs_pos = probs
        probs_neg = (probs - self.clip).clamp(min=0)
        
        # Basic CE
        los_pos = targets * torch.log(probs_pos.clamp(min=1e-8))
        los_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=1e-8))
        loss = -(los_pos + los_neg)
        
        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt = probs_pos * targets + (1 - probs_neg) * (1 - targets)
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_weight = torch.pow(1 - pt, one_sided_gamma)
            loss = loss * one_sided_weight
        
        # Apply IA weights if provided
        if self.ia_weights is not None:
            loss = loss * self.ia_weights.unsqueeze(0)
        
        return loss.mean()


def get_loss_function(
    loss_type: str,
    ia_weights: np.ndarray = None,
    pos_weight: np.ndarray = None,
    device: torch.device = None
) -> nn.Module:
    """
    Factory function to create loss functions.
    
    Args:
        loss_type: One of 'bce', 'weighted_bce', 'focal', 'asymmetric'
        ia_weights: IA weights for each term
        pos_weight: Positive class weights
        device: Device to move tensors to
    
    Returns:
        Loss function module
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert to tensors
    ia_tensor = None
    if ia_weights is not None:
        ia_tensor = torch.tensor(ia_weights, dtype=torch.float32, device=device)
    
    pos_weight_tensor = None
    if pos_weight is not None:
        pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    
    if loss_type == 'bce':
        return nn.BCEWithLogitsLoss()
    
    elif loss_type == 'weighted_bce':
        if ia_tensor is None:
            raise ValueError("IA weights required for weighted_bce loss")
        return WeightedBCELoss(ia_tensor, pos_weight_tensor)
    
    elif loss_type == 'focal':
        return FocalLoss(ia_weights=ia_tensor)
    
    elif loss_type == 'asymmetric':
        return AsymmetricLoss(ia_weights=ia_tensor)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def compute_pos_weight(labels: np.ndarray, max_weight: float = 10.0) -> np.ndarray:
    """
    Compute positive class weights for class imbalance.
    
    Args:
        labels: Binary labels, shape (n_samples, n_classes)
        max_weight: Maximum weight to avoid extreme values
    
    Returns:
        Positive class weights, shape (n_classes,)
    """
    pos_count = labels.sum(axis=0)
    neg_count = labels.shape[0] - pos_count
    
    # Avoid division by zero
    pos_count = np.maximum(pos_count, 1)
    
    pos_weight = neg_count / pos_count
    pos_weight = np.clip(pos_weight, 1.0, max_weight)
    
    return pos_weight

