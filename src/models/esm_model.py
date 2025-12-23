"""
ESM-2 based models for protein function prediction.
ESM-2 is a state-of-the-art protein language model from Meta AI.

Location: src/models/esm_model.py

UPDATED:
- Added gradient checkpointing support for memory efficiency
- Added attention pooling option
- Better initialization
- Support for different model sizes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import EsmModel, EsmTokenizer, EsmConfig
from typing import Optional, Dict, List


class AttentionPooling(nn.Module):
    """Attention-based pooling over sequence length."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.Tanh(),
            nn.Linear(hidden_size // 4, 1)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)
        
        Returns:
            Pooled representation: (batch, hidden_size)
        """
        # Compute attention weights
        attn_weights = self.attention(hidden_states).squeeze(-1)  # (batch, seq_len)
        
        # Mask padding tokens
        attn_weights = attn_weights.masked_fill(
            attention_mask == 0, float('-inf')
        )
        attn_weights = F.softmax(attn_weights, dim=-1)  # (batch, seq_len)
        
        # Weighted sum
        pooled = torch.bmm(
            attn_weights.unsqueeze(1),  # (batch, 1, seq_len)
            hidden_states  # (batch, seq_len, hidden_size)
        ).squeeze(1)  # (batch, hidden_size)
        
        return pooled


class ESMForGOPrediction(nn.Module):
    """
    ESM-2 model with classification head for GO term prediction.
    
    ESM-2 comes in different sizes:
    - esm2_t6_8M_UR50D: 8M parameters (fast, for testing)
    - esm2_t12_35M_UR50D: 35M parameters
    - esm2_t30_150M_UR50D: 150M parameters
    - esm2_t33_650M_UR50D: 650M parameters (recommended)
    - esm2_t36_3B_UR50D: 3B parameters (requires A100+)
    """
    
    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        num_labels: int = None,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
        use_gradient_checkpointing: bool = False,
        pooling_type: str = "mean",  # 'mean', 'cls', 'attention'
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        self.pooling_type = pooling_type
        
        # Load ESM-2 model and tokenizer
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.esm = EsmModel.from_pretrained(model_name)
        
        # Get hidden size from config
        self.hidden_size = self.esm.config.hidden_size
        
        # Enable gradient checkpointing for memory efficiency
        if use_gradient_checkpointing:
            self.esm.gradient_checkpointing_enable()
            print(f"Gradient checkpointing enabled for {model_name}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.esm.parameters():
                param.requires_grad = False
            print(f"Backbone frozen for {model_name}")
        
        # Pooling layer
        if pooling_type == "attention":
            self.pooler = AttentionPooling(self.hidden_size)
        else:
            self.pooler = None
        
        # Classification head with residual connection
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_labels),
        )
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool hidden states to get sequence representation.
        
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: (batch, seq_len)
        
        Returns:
            Pooled representation: (batch, hidden_size)
        """
        if self.pooling_type == "cls":
            # Use [CLS] token representation
            return hidden_states[:, 0, :]
        
        elif self.pooling_type == "attention":
            return self.pooler(hidden_states, attention_mask)
        
        else:  # mean pooling
            mask = attention_mask.unsqueeze(-1).float()
            sum_embeddings = (hidden_states * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask
    
    def forward(
        self,
        sequences: List[str],
        labels: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            sequences: List of protein sequences (amino acid strings)
            labels: Optional tensor of shape (batch_size, num_labels)
            loss_fn: Optional custom loss function
        
        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        # Tokenize sequences
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        
        # Move to same device as model
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get ESM embeddings
        outputs = self.esm(**inputs)
        
        # Pool over sequence length
        pooled = self.pool(
            outputs.last_hidden_state,
            inputs["attention_mask"]
        )
        
        # Classification
        logits = self.classifier(pooled)
        
        result = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            labels = labels.to(device)
            
            if loss_fn is not None:
                result["loss"] = loss_fn(logits, labels)
            else:
                loss_fn_default = nn.BCEWithLogitsLoss()
                result["loss"] = loss_fn_default(logits, labels)
        
        return result
    
    def predict(
        self,
        sequences: List[str],
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Make predictions for sequences.
        
        Args:
            sequences: List of protein sequences
            threshold: Probability threshold for predictions
        
        Returns:
            Probability tensor of shape (batch_size, num_labels)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(sequences)
            probs = torch.sigmoid(outputs["logits"])
        return probs
    
    def get_embeddings(
        self,
        sequences: List[str]
    ) -> torch.Tensor:
        """
        Get pooled embeddings for sequences.
        
        Args:
            sequences: List of protein sequences
        
        Returns:
            Embeddings tensor of shape (batch_size, hidden_size)
        """
        self.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            device = next(self.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = self.esm(**inputs)
            pooled = self.pool(
                outputs.last_hidden_state,
                inputs["attention_mask"]
            )
        return pooled


class ESMLightning(nn.Module):
    """
    Lighter version using smaller ESM-2 model for faster experimentation.
    Good for initial testing on consumer GPUs.
    """
    
    def __init__(
        self,
        model_name: str = "facebook/esm2_t12_35M_UR50D",
        num_labels: int = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.esm = EsmModel.from_pretrained(model_name)
        self.hidden_size = self.esm.config.hidden_size
        
        # Simple classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, num_labels),
        )
    
    def forward(
        self,
        sequences: List[str],
        labels: Optional[torch.Tensor] = None,
        loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = self.esm(**inputs)
        
        # Mean pooling
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        embeddings = outputs.last_hidden_state * attention_mask
        pooled = embeddings.sum(dim=1) / attention_mask.sum(dim=1)
        
        logits = self.classifier(pooled)
        
        result = {"logits": logits}
        
        if labels is not None:
            labels = labels.to(device)
            if loss_fn is not None:
                result["loss"] = loss_fn(logits, labels)
            else:
                loss_fn_default = nn.BCEWithLogitsLoss()
                result["loss"] = loss_fn_default(logits, labels)
        
        return result


def get_model(config: dict, num_labels: int) -> nn.Module:
    """
    Factory function to create model based on config.
    
    Args:
        config: Model configuration dictionary
        num_labels: Number of output labels
    
    Returns:
        Model instance
    """
    model_type = config.get("type", "esm")
    
    if model_type == "esm_light":
        return ESMLightning(
            model_name=config["name"],
            num_labels=num_labels,
            dropout=config.get("dropout", 0.2),
        )
    else:
        return ESMForGOPrediction(
            model_name=config["name"],
            num_labels=num_labels,
            dropout=config.get("dropout", 0.1),
            freeze_backbone=config.get("freeze_backbone", False),
            use_gradient_checkpointing=config.get("use_gradient_checkpointing", False),
            pooling_type=config.get("pooling_type", "mean"),
        )

